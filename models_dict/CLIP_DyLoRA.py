#https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
#text.transformer.embeddings.position_ids error is version of transformers
#https://github.com/openai/CLIP
#https://github.com/openai/CLIP/blob/main/clip/model.py
#https://github.com/hitachinsk/SAMed/blob/main/sam_lora_image_encoder.py
import os
import json
import torch
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

from open_clip import create_model_from_pretrained, get_tokenizer

from.DyLoRA import dynamic

def local_clip_model(args, num_id, lora_r, classnames):

    alpha = 8
    biomedclip, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    template = 'this is an image of {}'
    
    ### obtain text features
    all_text_features = []
    if num_id == -1:
        for c in classnames:
            prompt = template.format(c.replace("_", " "))
            prompt = tokenizer(prompt)
            text_features = biomedclip.encode_text(prompt) # 1 512
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 1, C
            all_text_features.append(text_features)
        all_text_features = torch.cat(all_text_features,dim=0)#C,512
        print('Template:', template)
        print(f"Prompt size={all_text_features.size()})")
    
    image_encoder = biomedclip.visual
    if lora_r>0:
        print('LoRAing model.... ')
        lora_image_encoder = LoRA(image_encoder, lora_r, alpha)
        
        if args.method == 'FFA-LoRA':
            for n, p in lora_image_encoder.named_parameters():
                if 'linear_a_q' in n or 'linear_a_k' in n or 'linear_a_v' in n:
                    p.requires_grad = False
                    
            for n, p in lora_image_encoder.named_parameters():
                if 'linear_b_q' in n or 'linear_b_k' in n or 'linear_b_v' in n:
                    if p.requires_grad == False:
                        assert False
        
        return lora_image_encoder, all_text_features
    else:
        return image_encoder, all_text_features
    

class LoRA(nn.Module):
    def __init__(self, model, r: int, alpha, lora_layer=None):
        super(LoRA, self).__init__()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.r = r
        assert r > 0
        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(model.trunk.blocks)))

        # Create for storage, then we can init them or load weights
        # These are linear layers
        self.w_As = []
        self.w_Bs = []

        # Lets freeze first
        for param in model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(model.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.extend([w_a_linear_q,w_a_linear_k, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q,w_b_linear_k, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha,
            )
        self.reset_parameters()
        self.lora_image_encoder = model
        
    def set_rank(self, new_rank, frozen=True):
        for blk in self.lora_image_encoder.trunk.blocks:
            blk.attn.qkv.set_rank(new_rank, frozen)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_image_encoder(x)



class _LoRA_qkv(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r,
        alpha,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.scaling = alpha / r
        
        self.nd_lora_a_q = dynamic(maximum_rank=r)
        self.nd_lora_b_q = dynamic(maximum_rank=r)
        self.nd_lora_a_k = dynamic(maximum_rank=r)
        self.nd_lora_b_k = dynamic(maximum_rank=r)
        self.nd_lora_a_v = dynamic(maximum_rank=r)
        self.nd_lora_b_v = dynamic(maximum_rank=r)
        

    def set_rank(self, rank, frozen=True):
        self.nd_lora_a_q.set_rank(rank, frozen=frozen)
        self.nd_lora_b_q.set_rank(rank, frozen=frozen)
        self.nd_lora_a_k.set_rank(rank, frozen=frozen)
        self.nd_lora_b_k.set_rank(rank, frozen=frozen)
        self.nd_lora_a_v.set_rank(rank, frozen=frozen)
        self.nd_lora_b_v.set_rank(rank, frozen=frozen)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        
        #new_q = self.linear_b_q(self.linear_a_q(x))* self.scaling
        linear_a_q = self.nd_lora_a_q(self.linear_a_q.weight.T).T
        linear_b_q = self.nd_lora_b_q(self.linear_b_q.weight)
        new_q = (x @ linear_a_q.T @ linear_b_q.T) * self.scaling

        #new_k = self.linear_b_k(self.linear_a_k(x))* self.scaling
        linear_a_k = self.nd_lora_a_k(self.linear_a_k.weight.T).T
        linear_b_k = self.nd_lora_b_k(self.linear_b_k.weight)
        new_k = (x @ linear_a_k.T @ linear_b_k.T) * self.scaling
        
        #new_v = self.linear_b_v(self.linear_a_v(x))* self.scaling
        linear_a_v = self.nd_lora_a_v(self.linear_a_v.weight.T).T
        linear_b_v = self.nd_lora_b_v(self.linear_b_v.weight)
        new_v = (x @ linear_a_v.T @ linear_b_v.T) * self.scaling
        
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, self.dim:-self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v
        return qkv


def set_trainable_params(model):
    for n, p in model.parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

if __name__=="__main__":
    lora_model()