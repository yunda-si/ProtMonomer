#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:27:15 2023

@author: yunda_si
"""

import torch
from Bio import SeqIO

def check_input(file_list):
    checked_list = []
    for msa_file in file_list:
        if not msa_file.endswith('.a3m'):
            print(f'{msa_file} file_name error')
            continue
        
        count_seq = 0
        have_error = False
        for record in SeqIO.parse(msa_file, 'fasta'):
            try:
                chain_seq = str(record.seq).strip()
            except:
                print(f'{msa_file} parse error')
                have_error = True
                continue        
            
            count_seq += 1
            
        if have_error:
            continue
        
        checked_list.append(msa_file)
        
    return checked_list


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def localrigids(
                coords37,
                n_idx=0,
                ca_idx=1,
                c_idx=2,
                eps=1e-8,
                ):
    n_coords = coords37[..., n_idx, :]
    ca_coords = coords37[..., ca_idx, :]
    c_coords = coords37[..., c_idx, :]

    v1 = ca_coords - n_coords
    v2 = c_coords - ca_coords
    e1 = v1 / (eps + torch.linalg.norm(v1, dim=-1, keepdim=True)) #(torch.sqrt(torch.sum(v1 ** 2, dim=-1, keepdim=True) + eps))
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / (eps + torch.linalg.norm(u2, dim=-1, keepdim=True))
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.concat((e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)), dim=-1)
    T = ca_coords

    return R, T
