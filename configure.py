
########################hyperparamter#######################
'''
For each table, the main hyperparamter is C,
C \in [0.1, 0.2, 0.3, 0.4, 0.6]
For example: in our machine, 

===============Table I ==================
---------------->Kvasir dataset<------------
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=1.0  C = 0.1
privacy budget=3.0  C = 0.2
privacy budget=6.0  C = 0.3

---------------->OCT dataset<------------
lora_r = 8, dirichlet_alpha = 0.1
privacy budget=0.1  C = 0.1
privacy budget=0.5  C = 0.1
privacy budget=1.0  C = 0.2


'''

###########################################################