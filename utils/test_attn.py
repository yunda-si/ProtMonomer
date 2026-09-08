# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import torch
from flash_attn import flash_attn_func
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention as DSAttn

batch = 1
column = 128
residue = 384
nhead = 8
head_dim = 32
device = torch.device('cuda:0')

# test DS4Sci
Q = torch.rand(batch, residue, column, nhead, head_dim).to(device).bfloat16()
K = torch.rand(batch, residue, column, nhead, head_dim).to(device).bfloat16()
V = torch.rand(batch, residue, column, nhead, head_dim).to(device).bfloat16()
out = DSAttn(Q, K, V, [None])

# test flash_attn_func
Q = torch.rand(batch, residue, nhead, head_dim).to(device).bfloat16()
K = torch.rand(batch, residue, nhead, head_dim).to(device).bfloat16()
V = torch.rand(batch, residue, nhead, head_dim).to(device).bfloat16()
output = flash_attn_func(Q, K, V,
                         softmax_scale=1,
                         dropout_p=0.0,
                         causal=False)