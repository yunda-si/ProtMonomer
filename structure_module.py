#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:34:17 2024

@author: yunda_si
"""

import torch
import torch.nn as nn
from modules import MSA2ATOM
from collections import OrderedDict
from utils.rigid_utils import quat_multiply_by_vec, quat_to_rot
from modules import FeedForwardNetwork as StructureModuleTransition

class IPA(nn.Module):

    def __init__(self,
                 pair_channel=128,
                 atom_channel=128,
                 num_atom=37,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 ):

        super(IPA, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.q_seq = nn.Linear(atom_channel, atom_channel, bias=False)
        self.k_seq = nn.Linear(atom_channel, atom_channel, bias=False)

        self.qp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.kp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.vp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)

        self.norm_pair = nn.RMSNorm(pair_channel)
        self.trans_pair = nn.Parameter((torch.rand(num_atom, pair_channel, atom_head) - 0.5) * 2 / (pair_channel ** 0.5))
        self.head_weights = nn.Parameter(torch.zeros(self.nhead))

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)


    def forward(self, seq, pair, coords, atom_mask, basis, atom_idx=None):

        atom_mask = torch.einsum('ai, aj -> aij', atom_mask, atom_mask)

        seq = self.ln_seq_in(seq)
        if atom_idx == None:
            pair_bias = torch.einsum('bijc, ach -> bhaij', self.norm_pair(pair), self.trans_pair)
        else:
            pair_bias = torch.einsum('bijc, ach -> bhaij', self.norm_pair(pair), self.trans_pair[atom_idx])

        batch, num_atom, num_residue, _ = seq.shape

        q = self.q_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)
        k = self.k_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)

        q = q * self.scale_q
        q_attn_weights = torch.einsum('baihc,bajhc -> bhaij', q, k)

        qp = self.qp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        kp = self.kp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        vp = self.vp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)

        qp = torch.einsum('balhpi, blji -> bhalpj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        kp = torch.einsum('balhpi, blji -> bhalpj', kp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        vp = torch.einsum('balhpi, blji -> bhalpj', vp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        

        qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                              kp ** 2).unsqueeze(
            -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * self.scale_qp * (-0.5)
        for i in range(3):
            head_weights = head_weights.unsqueeze(-1)

        qp_attn_weights = qp_attn_weights * head_weights

        attn = pair_bias + q_attn_weights + qp_attn_weights
        attn = attn * (3 ** -0.5)
        attn = attn.softmax(-1)

        o3 = torch.einsum('bhaij, bhajpc -> bhaipc', attn, vp) - coords.unsqueeze(1).unsqueeze(-2)
        o3 = torch.einsum('bhalpi, blij -> balhpj', o3, basis)
        o3 = torch.flatten(o3, -3, -1)

        seq = self.linear_out(torch.cat((o3,), dim=-1))

        return self.dropout_module(seq)


class IPAATOM(nn.Module):

    def __init__(self,
                 atom_channel=128,
                 pair_channel=128,
                 num_atom=37,
                 atom_head=8,
                 points=8,
                 eps=1e-6,
                 dropout_p=0.0,
                 ):
        super(IPAATOM, self).__init__()

        self.nhead = atom_head
        self.head_dim = atom_channel // atom_head
        self.scale_q = self.head_dim ** (-0.5)
        self.points = points
        self.scale_qp = (points * 9.0 / 2) ** (-0.5)
        self.eps = eps

        self.ln_seq_in = nn.RMSNorm(atom_channel)

        self.q_seq = nn.Linear(atom_channel, atom_channel, bias=False)
        self.k_seq = nn.Linear(atom_channel, atom_channel, bias=False)

        self.qp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.kp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)
        self.vp_seq = nn.Linear(atom_channel, atom_head * points * 3, bias=False)

        self.head_weights = nn.Parameter(torch.zeros(self.nhead))

        self.linear_out = nn.Linear(atom_head * (points * 3), atom_channel)

        self.softplus = nn.Softplus()
        self.dropout_attn = nn.Dropout(dropout_p)
        self.dropout_module = nn.Dropout(dropout_p)

    def forward(self, seq, pair, coords, atom_mask, basis):

        coords = torch.permute(coords, [0, 2, 1, 3]) #blac

        seq = self.ln_seq_in(seq)
        batch, num_atom, num_residue, _ = seq.shape

        q = self.q_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)
        k = self.k_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.head_dim)

        q = q * self.scale_q
        q_attn_weights = torch.einsum('bilhc,bjlhc -> bhlij', q, k)

        qp = self.qp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        kp = self.kp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)
        vp = self.vp_seq(seq).view(batch, num_atom, num_residue, self.nhead, self.points, 3)

        qp = torch.einsum('balhpi, blji -> bhlapj', qp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        kp = torch.einsum('balhpi, blji -> bhlapj', kp, basis) + coords.unsqueeze(1).unsqueeze(-2)
        vp = torch.einsum('balhpi, blji -> bhlapj', vp, basis) + coords.unsqueeze(1).unsqueeze(-2)

        qp_attn_weights = torch.einsum('bhlipc ->bhli', qp ** 2).unsqueeze(-2) + torch.einsum('bhlipc ->bhli',
                                                                                              kp ** 2).unsqueeze(
            -1) - 2 * torch.einsum('bhlipc, bhljpc ->bhlji', qp, kp)

        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * self.scale_qp * (-0.5)
        for i in range(3):
            head_weights = head_weights.unsqueeze(-1)

        qp_attn_weights = qp_attn_weights * head_weights

        attn = q_attn_weights + qp_attn_weights
        attn = attn * (2 ** -0.5)
        attn = attn.softmax(-1)

        o3 = torch.einsum('bhlij, bhljpc -> bhlipc', attn, vp) - coords.unsqueeze(1).unsqueeze(-2)
        o3 = torch.einsum('bhlapi, blij -> balhpj', o3, basis)
        o3 = torch.flatten(o3, -3, -1)

        seq = self.linear_out(torch.cat((o3,), dim=-1))

        return self.dropout_module(seq)


class StructureBlock(nn.Module):
    def __init__(self,
                 msa_channel,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=37,
                 dropout_p=0.0,
                 points=8,
                 eps=1e-7,
                 split_atom=None,
                 split_seq=None,
                 ):

        super(StructureBlock, self).__init__()

        self.num_atom = num_atom
        self.split_atom = split_atom

        self.msa2atom = MSA2ATOM(msa_channel=msa_channel,
                                 num_atom=num_atom,
                                 atom_channel=atom_channel,
                                 split_seq=split_seq)

        self.ipa = IPA(pair_channel=pair_channel,
                       atom_channel=atom_channel,
                       num_atom=num_atom,
                       atom_head=atom_nhead,
                       points=points,
                       eps=eps,
                       dropout_p=dropout_p
                       )

        self.ipa_atom = IPAATOM(atom_channel=atom_channel,
                                pair_channel=pair_channel,
                                num_atom=num_atom,
                                atom_head=atom_nhead,
                                points=points,
                                eps=eps,
                                dropout_p=dropout_p
                                )

        self.transition = StructureModuleTransition(in_channels=atom_channel,
                                                    dropout_p=dropout_p)

    def forward(self, struc_emb, msa, pair, coords, atom_mask, basis):

        b_seq = struc_emb.shape[0]
        b, n, l, c = msa.shape
        sel_idx = torch.multinomial(torch.ones(b_seq, n), replacement=False, num_samples=min(torch.randint(min(n, 32),n+1,(1,)).item(), 2048, n))

        struc_emb += self.msa2atom(msa[0, sel_idx])

        if self.split_atom != None:
            for atom_idx in torch.split(torch.arange(self.num_atom), self.split_atom):
                struc_emb[:, atom_idx] += self.ipa(struc_emb[:, atom_idx], pair, coords[:, atom_idx], atom_mask, basis, atom_idx=atom_idx)
        else:
            struc_emb += self.ipa(struc_emb, pair, coords, atom_mask, basis)

        struc_emb += self.ipa_atom(struc_emb, pair, coords, atom_mask, basis)

        struc_emb += self.transition(struc_emb)

        return struc_emb


class StructureModule(nn.Module):

    def __init__(self,
                 msa_channel,
                 atom_channel,
                 pair_channel,
                 atom_nhead,
                 num_atom=37,
                 block_num=8,
                 using_flash=True,
                 dropout_p=0.0,
                 plddt_bin=50,
                 split_atom=None,
                 split_seq=None
                 ):

        super(StructureModule, self).__init__()

        self.msa_channel = msa_channel
        self.dropout_p = dropout_p
        self.using_flash = using_flash
        self.split_atom = split_atom
        self.split_seq = split_seq

        self.structure_block = self._make_atom_encoder(block_num, atom_channel, pair_channel, atom_nhead, num_atom)
        self.ln_seq_out = nn.RMSNorm(atom_channel)

        self.coords_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                        nn.RMSNorm(atom_channel),
                                        nn.GELU())

        self.weights_coords = nn.Parameter((torch.rand(num_atom, atom_channel, 3) - 0.5) * 2 / (atom_channel ** 0.5))

        self.basis_out = nn.Sequential(nn.Linear(atom_channel, atom_channel, bias=True),
                                       nn.RMSNorm(atom_channel),
                                       nn.GELU(),
                                       nn.Linear(atom_channel, 3, bias=False),
                                       )

    def _make_atom_encoder(self, block_num, atom_channel, pair_channel, atom_nhead, num_atom):

        layers = []
        for index in range(block_num):
            layer = StructureBlock(msa_channel=self.msa_channel,
                                   atom_channel=atom_channel,
                                   pair_channel=pair_channel,
                                   atom_nhead=atom_nhead,
                                   num_atom=num_atom,
                                   dropout_p=self.dropout_p,
                                   split_atom=self.split_atom,
                                   split_seq=self.split_seq)

            layers.append(('struc_block' + str(index), layer))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, struc_emb, msa, pair, ori_coords, atom_mask, quat):

        # ori_coords = ori_coords * atom_mask
        quat = quat.float().detach()
        basis = quat_to_rot(quat)
        ori_coords = ori_coords.float()
        atom_mask = atom_mask.squeeze()

        for idx_layer, layer in enumerate(self.structure_block):
            struc_emb = layer(struc_emb, msa, pair, ori_coords, atom_mask, basis)

        seq = self.ln_seq_out(struc_emb.float())
        coords = torch.einsum('bali, aij -> balj', self.coords_out(seq), self.weights_coords)
        coords = ori_coords + torch.einsum('bali, blji -> balj', coords, basis)

        quat_update = self.basis_out(seq[:, 1, :, :])
        new_quat = quat + quat_multiply_by_vec(quat, quat_update)
        new_quat = new_quat / torch.linalg.norm(new_quat, dim=-1, keepdim=True)

        return coords, struc_emb, new_quat


