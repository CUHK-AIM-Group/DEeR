
########################hyperparamter#######################
'''
For each table, the main hyperparamter is C,
C \in [0.1, 0.2, 0.3, 0.4, 0.6]
For example: in our machine, optimal C is 

===============Table I ==================
---------------->Kvasir dataset<------------
LoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=1.0  C = 0.1
privacy budget=3.0  C = 0.2
privacy budget=6.0  C = 0.3


FFA-LoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=1.0  C = 0.3
privacy budget=3.0  C = 0.3
privacy budget=6.0  C = 0.3


DP-DyLoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=1.0  C = 0.1
privacy budget=3.0  C = 0.2
privacy budget=6.0  C = 0.3

Ours
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=1.0  C = 0.2
privacy budget=3.0  C = 0.3
privacy budget=6.0  C = 0.6



---------------->OCT dataset<------------
LoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=0.1  C = 0.1
privacy budget=0.5  C = 0.1
privacy budget=1.0  C = 0.2


FFA-LoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=0.1  C = 0.2
privacy budget=0.5  C = 0.2
privacy budget=1.0  C = 0.6

DP-DyLoRA
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=0.1  C = 0.1
privacy budget=0.5  C = 0.1
privacy budget=1.0  C = 0.1

Ours
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=0.1  C = 0.1
privacy budget=0.5  C = 0.3
privacy budget=1.0  C = 0.4
'''

###########################################################