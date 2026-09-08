#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:03:39 2024

@author: yunda_si
"""

import copy
import torch
from torch import nn
import math
import torch.nn.functional as F
from collections import OrderedDict
from flash_attn import flash_attn_func
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention as DSAttn

class Attn(nn.Module):
    def __init__(self,
                 in_channels,
                 nhead,
                 dropout_p=0.1,
                 is_causal=False,
                 tied_attn=False
                 ):

        super(Attn, self).__init__()

        self.in_channels = in_channels
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.head_dim = self.in_channels // self.nhead
        self.scaling = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.tied_attn = tied_attn

        self.norm = nn.RMSNorm(in_channels)
        self.linear_q = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_k = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_v = nn.Linear(in_channels, in_channels, bias=False)
        self.linear_ff = nn.Linear(in_channels, in_channels, bias=True)
        self.gate = nn.Linear(in_channels, in_channels, bias=True)

        self.dropout_module = nn.Dropout(dropout_p)
        self.dropout_attn = nn.Dropout(dropout_p)
        self.sigmoid = nn.Sigmoid()

    def cal_attn(self, q, k, v, scale, attn_bias):

        q *= scale
        attn_weights = torch.einsum(f'bhic,bhjc -> bhij', q, k)
        if attn_bias is not None:
            attn_weights += attn_bias

        attn_prob = attn_weights.softmax(-1)
        attn_prob = self.dropout_attn(attn_prob)

        output = torch.einsum(f'bhij, bhjc -> bhic', attn_prob, v)

        return output

    def forward(self, x, attn_bias=None):

        batch, num_row, num_column, _ = x.shape
        comb_batch = batch*num_row

        x = torch.flatten(x,0,1)
        if attn_bias is not None:
            attn_bias = torch.flatten(attn_bias,0,1)

        x = self.norm(x)

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if attn_bias is not None:
            q = q * (2 ** -0.5)
            attn_bias = attn_bias * (2 ** -0.5)

        q = q.view(comb_batch, num_column, self.nhead, self.head_dim)
        k = k.view(comb_batch, num_column, self.nhead, self.head_dim)
        v = v.view(comb_batch, num_column, self.nhead, self.head_dim)

        if attn_bias is not None:
            if num_row>24 or num_column>24:
                output = DSAttn(q.unflatten(0,(batch,num_row)), k.unflatten(0,(batch,num_row)), v.unflatten(0,(batch,num_row)), [None, attn_bias.unflatten(0,(batch,1))])
                output = output.flatten(0,1)
            else:
                q = q.transpose(1, 2).unflatten(0,(batch,num_row))
                k = k.transpose(1, 2).unflatten(0,(batch,num_row))
                v = v.transpose(1, 2).unflatten(0,(batch,num_row))
                output = F.scaled_dot_product_attention(q, k, v,
                                                        scale=self.scaling,
                                                        dropout_p=0.0,
                                                        attn_mask=attn_bias.unflatten(0,(batch,1)),
                                                        is_causal=self.is_causal)
                output = output.flatten(0,1).transpose(1, 2)
        else:
            output = flash_attn_func(q, k, v,
                                     softmax_scale=self.scaling,
                                     dropout_p=0.0,
                                     causal=self.is_causal)

        g = self.sigmoid(self.gate(x))
        g = g.view(comb_batch, num_column, self.nhead, self.head_dim)
        output = g * output
        output = output.view(comb_batch, num_column, self.in_channels).contiguous()

        return torch.unflatten(self.dropout_module(self.linear_ff(output)),0,(batch, num_row))


class FeedForwardNetwork(nn.Module):

    def __init__(
                self,
                in_channels,
                dropout_p=0.1,
                scalen=4,
                ):
        super().__init__()

        hidden_channels = scalen*in_channels

        self.norm = nn.RMSNorm(in_channels)
        self.left = nn.Linear(in_channels, hidden_channels, bias=False)
        self.right = nn.Linear(in_channels, hidden_channels, bias=False)
        self.linear_ff = nn.Linear(hidden_channels, in_channels, bias=False)

        self.dropout_module = nn.Dropout(dropout_p)
        self.act = nn.SiLU()

    def forward(self, x):

        x = self.norm(x)

        left = self.left(x)
        right = self.right(x)
        x = self.linear_ff(self.act(left) * right)

        return x


class TriangularMultiplicative(nn.Module):
    def __init__(
            self,
            in_channels,
            scalen=1,
            dropout_p2d=0.1,
            split_res=None,
    ):

        super(TriangularMultiplicative, self).__init__()

        hidden_channels = in_channels*scalen
        self.split_res = split_res

        self.norm1 = nn.RMSNorm(in_channels)
        self.linear_left = nn.Linear(in_channels, hidden_channels)
        self.linear_right = nn.Linear(in_channels, hidden_channels)
        self.gate_left = nn.Linear(in_channels, hidden_channels)
        self.gate_right = nn.Linear(in_channels, hidden_channels)

        self.gate_out = nn.Linear(in_channels, in_channels)

        self.norm2 = nn.RMSNorm(hidden_channels)
        self.linear_out = nn.Linear(hidden_channels, in_channels)

        self.dropout_module = nn.Dropout(p=dropout_p2d)
        self.act = nn.Sigmoid()


    def forward(self, pair, mode='outgoing'):

        pair = self.norm1(pair)

        left = self.act(self.gate_left(pair)) * self.linear_left(pair)
        right = self.act(self.gate_right(pair)) * self.linear_right(pair)
        out = self.act(self.gate_out(pair))

        if mode == 'outgoing':
            if self.split_res is not None:
                for idx in range(math.ceil(pair.shape[1] / self.split_res)):
                    out[:,idx*self.split_res:(idx+1)*self.split_res] *= self.linear_out(self.norm2(torch.einsum('bilc, bjlc -> bijc', left[:,idx*self.split_res:(idx+1)*self.split_res], right)))
            else:
                out *= self.linear_out(self.norm2(torch.einsum('bilc, bjlc -> bijc', left, right)))
        else:
            if self.split_res is not None:
                for idx in range(math.ceil(pair.shape[1] / self.split_res)):
                    out[:,idx*self.split_res:(idx+1)*self.split_res] *= self.linear_out(self.norm2(torch.einsum('blic, bljc -> bijc', left[:,:,idx*self.split_res:(idx+1)*self.split_res], right)))
            else:
                out *= self.linear_out(self.norm2(torch.einsum('blic, bljc -> bijc', left, right)))

        return self.dropout_module(out)


class AxialFormer(nn.Module):

    def __init__(self,
                 in_channel,
                 nhead,
                 dropout_p=0.1,
                 tied_attn=False,
                 is_causal=False,
                 with_column_attn=True,
                 split_seq=None,
                 split_res=None,
                 ):

        super(AxialFormer, self).__init__()

        self.in_channel = in_channel
        self.nhead = nhead
        self.with_column_attn = with_column_attn
        self.split_seq = split_seq
        self.split_res = split_res

        self.row_attn = Attn(
                             in_channels=self.in_channel,
                             nhead=self.nhead,
                             dropout_p=dropout_p,
                             is_causal=is_causal,
                             tied_attn=tied_attn
                             )

        if self.with_column_attn:
            self.column_attn = Attn(
                                    in_channels=self.in_channel,
                                    nhead=self.nhead,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    tied_attn=tied_attn
                                    )

        self.ffn = FeedForwardNetwork(
                                      in_channels=self.in_channel,
                                      dropout_p=dropout_p,
                                      )

    def forward(self, x, row_bias=None, column_bias=None):

        if self.split_res is not None:
            for idx in range(math.ceil(x.shape[1] / self.split_res)):
                x[:, idx * self.split_res:(idx + 1) * self.split_res] += self.row_attn(x[:, idx * self.split_res:(idx + 1) * self.split_res], row_bias)
        else:
            x += self.row_attn(x, row_bias)

        if self.with_column_attn:
            x = x.transpose(1, 2)
            if self.split_seq is not None:
                for idx in range(math.ceil(x.shape[1] / self.split_seq)):
                    x[:, idx * self.split_seq:(idx + 1) * self.split_seq] += self.column_attn(x[:, idx * self.split_seq:(idx + 1) * self.split_seq], column_bias)
            else:
                x += self.column_attn(x, column_bias)
            x = x.transpose(1, 2)

        if self.split_res is not None:
            for idx in range(math.ceil(x.shape[1] / self.split_res)):
                x[:, idx * self.split_res:(idx + 1) * self.split_res] += self.ffn(x[:, idx * self.split_res:(idx + 1) * self.split_res])
        else:
            x += self.ffn(x)

        return x


class Transpair(nn.Module):

    def __init__(self,
                 in_channel,
                 nhead,
                 tied_attn,
                 ):
        super(Transpair, self).__init__()

        self.tied_attn = tied_attn
        self.linear = nn.Linear(in_channel, nhead, bias=False)
        self.norm = nn.RMSNorm(in_channel)

    def forward(self, pair):
        pair = self.norm(pair)
        pair = self.linear(pair)
        pair = torch.permute(pair, [0, 3, 1, 2])

        if not self.tied_attn:
            pair = pair.unsqueeze(1)

        return pair


class TransMSA(nn.Module):

    def __init__(self,
                 msa_channel=128,
                 hidden_channel=32,
                 pair_channel=128,
                 dropout_p=0.1,
                 split_res=None,
                 outnorm=False,
                 ):

        super(TransMSA, self).__init__()

        self.split_res = split_res
        self.outnorm = outnorm

        self.norm = nn.RMSNorm(msa_channel)
        self.linear1 = nn.Linear(msa_channel, hidden_channel, bias=True)
        self.linear2 = nn.Linear(msa_channel, hidden_channel, bias=True)
        self.linear3 = nn.Linear(hidden_channel ** 2, pair_channel, bias=True)

        if outnorm:
            self.norm2 = nn.RMSNorm(pair_channel)

        self.dropout_module = nn.Dropout2d(dropout_p)

    def _opm(self, a, b):

        outer = torch.einsum("...mbc,...mde->...bdce", a, b)# / num_row

        return outer.reshape(outer.shape[:-2] + (-1,))

    def forward(self, msa):
        batch_size, num_row, num_column, c = msa.shape

        msa = self.norm(msa)
        if self.split_res is not None:
            pair = 0
            for x in torch.split(msa, self.split_res, dim=1):
                left = self.linear1(x)
                right = self.linear2(x)  # b, num_aa, num_seq, channel
                pair += self._opm(left, right)
        else:
            left = self.linear1(msa)
            right = self.linear2(msa)  # b, num_aa, num_seq, channel
            pair = self._opm(left, right)
        pair = self.linear3(pair/num_row)
        if self.outnorm:
            pair = self.norm2(pair)

        return pair


class Evoformer(nn.Module):

    def __init__(self,
                 msa_channel,
                 msa_nhead,
                 pair_channel,
                 pair_nhead,
                 dropout_p=0.,
                 dropout_p2d=0.0,
                 with_column_attn=False,
                 tied_attn=False,
                 split_seq=None,
                 split_res=None,
                 is_causal=False):

        super(Evoformer, self).__init__()

        self.split_seq = split_seq
        self.split_res = split_res

        self.transmsa = TransMSA(msa_channel=msa_channel,
                                 pair_channel=pair_channel,
                                 dropout_p=dropout_p,
                                 split_res=split_res,
                                 outnorm=False)

        self.transpair = Transpair(in_channel=pair_channel,
                                   nhead=msa_nhead,
                                   tied_attn=tied_attn,
                                   )

        self.multi_outgoing = TriangularMultiplicative(in_channels=pair_channel,
                                                       dropout_p2d=dropout_p2d,
                                                       split_res=split_res)

        self.multi_incoming = TriangularMultiplicative(in_channels=pair_channel,
                                                       dropout_p2d=dropout_p2d,
                                                       split_res=split_res)

        self.triangle_startnode = Attn(in_channels=pair_channel,
                                       nhead=pair_nhead,
                                       dropout_p=dropout_p2d,
                                       is_causal=is_causal,
                                       tied_attn=tied_attn)

        self.triangle_endnode = Attn(in_channels=pair_channel,
                                     nhead=pair_nhead,
                                     dropout_p=dropout_p2d,
                                     is_causal=is_causal,
                                     tied_attn=tied_attn)

        self.bias_startnode = Transpair(in_channel=pair_channel,
                                        nhead=pair_nhead,
                                        tied_attn=tied_attn,
                                        )

        self.bias_endnode = Transpair(in_channel=pair_channel,
                                      nhead=pair_nhead,
                                      tied_attn=tied_attn,
                                      )

        self.pair_ffn = FeedForwardNetwork(in_channels=pair_channel,
                                           dropout_p=dropout_p2d,
                                           )

        self.axial = AxialFormer(in_channel=msa_channel,
                                 nhead=msa_nhead,
                                 dropout_p=dropout_p,
                                 tied_attn=tied_attn,
                                 is_causal=is_causal,
                                 with_column_attn=with_column_attn,
                                 split_res=split_res,
                                 split_seq=split_seq,
                                 )

    def forward(self, msa, pair, msa_bias=None):

        if msa_bias is None:
            msa = self.axial(msa, row_bias=self.transpair(pair))
            pair += self.transmsa(msa)

        pair += self.multi_outgoing(pair)
        pair += self.multi_incoming(pair, 'incoming')

        pair_bias = self.bias_startnode(pair)
        if self.split_res is not None:
            for idx in range(math.ceil(pair.shape[1] / self.split_res)):
                pair[:, idx*self.split_res:(idx+1)*self.split_res] += self.triangle_startnode(pair[:, idx*self.split_res:(idx+1)*self.split_res],
                                                                                                    pair_bias)
        else:
            pair += self.triangle_startnode(pair, pair_bias)

        pair_bias = self.bias_endnode(pair).transpose(-2, -1)
        pair = pair.transpose(1,2)
        if self.split_res is not None:
            for idx in range(math.ceil(pair.shape[1] / self.split_res)):
                pair[:, idx*self.split_res:(idx+1)*self.split_res] += self.triangle_endnode(pair[:, idx*self.split_res:(idx+1)*self.split_res],
                                                                                                    pair_bias)
        else:
            pair += self.triangle_endnode(pair, pair_bias)
        pair = pair.transpose(1, 2)

        if self.split_res is not None:
            for idx in range(math.ceil(pair.shape[1] / self.split_res)):
                pair[:, idx * self.split_res:(idx + 1) * self.split_res] += self.pair_ffn(pair[:, idx * self.split_res:(idx + 1) * self.split_res])
        else:
            pair += self.pair_ffn(pair)

        return msa, pair


class MSA2ATOM(nn.Module):

    def __init__(self,
                 msa_channel=128,
                 atom_channel=128,
                 num_atom=37,
                 split_res=None,
                 ):
        super(MSA2ATOM, self).__init__()

        self.atom_channel = atom_channel
        self.num_atom = num_atom
        self.split_res = split_res

        self.linear1 = nn.Linear(msa_channel, atom_channel, bias=True)
        self.linear2 = nn.Linear(msa_channel, num_atom, bias=False)
        self.linear3 = nn.Linear(atom_channel, atom_channel, bias=True)

        self.norm = nn.RMSNorm(msa_channel)

    def forward(self, msa):

        if self.split_res is not None:
            full_out = []
            for x in torch.split(msa, self.split_res, dim=2):
                x = self.norm(x)
                left = self.linear1(x)
                right = torch.softmax(self.linear2(x), 1)

                pair = torch.einsum('bnlc,bnla -> balc', left, right)
                pair = self.linear3(pair)
                full_out.append(pair)
            pair = torch.concat(full_out, dim=2)
        else:
            msa = self.norm(msa)
            left = self.linear1(msa)
            right = torch.softmax(self.linear2(msa), 1)

            pair = torch.einsum('bnlc,bnla -> balc', left, right)
            pair = self.linear3(pair)

        return pair


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self,
                 no_bins,
                 c_in,
                 ):
        super(PerResidueLDDTCaPredictor, self).__init__()

        c_hidden = c_in * 2

        self.linear_1 = nn.Linear(c_in, c_hidden)
        self.linear_2 = nn.Linear(c_hidden, c_hidden)
        self.linear_3 = nn.Linear(c_hidden, no_bins)

        self.relu = nn.GELU()

    def forward(self, s):
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class InputEmbedder(nn.Module):
    def __init__(self,
                 msa_channel,
                 pair_channel,
                 nums_aa=23,
                 nums_posclass=65,
                 ):
        super(InputEmbedder, self).__init__()

        self.embed_msa = nn.Embedding(nums_aa, msa_channel)
        self.embed_seq = nn.Embedding(nums_aa, pair_channel)
        self.embd_pos = nn.Embedding(nums_posclass, pair_channel)

    def forward(self, protein):
        msa_init = protein['label_msa'].unsqueeze(0)
        seq_init = msa_init[:, 0]
        msa_init = self.embed_msa(msa_init)

        seq_init = self.embed_seq(seq_init)
        pair_init = seq_init.unsqueeze(1) + seq_init.unsqueeze(2)

        idx = self.relpos(protein['sel_idx']).to(msa_init.device)
        pair_init += self.embd_pos(idx).unsqueeze(0)

        return msa_init, pair_init

    def relpos(self, idx, min_dis=-32, max_dis=32):

        idx = idx.unsqueeze(0) - idx.unsqueeze(1)
        idx = torch.clamp(idx, min_dis, max_dis) - min_dis

        return idx


class MSAEncoder(nn.Module):

    def __init__(self,
                 num_block,
                 msa_channel,
                 msa_nhead,
                 pair_channel,
                 pair_nhead,
                 tied_attn,
                 dropout_p,
                 dropout_p2d,
                 with_column_attn,
                 split_seq,
                 split_res,
                 ):

        super(MSAEncoder, self).__init__()

        self.msa_channel = msa_channel
        self.msa_nhead = msa_nhead
        self.pair_channel = pair_channel
        self.pair_nhead = pair_nhead
        self.tied_attn = tied_attn
        self.dropout_p = dropout_p
        self.dropout_p2d = dropout_p2d
        self.with_column_attn = with_column_attn
        self.split_seq = split_seq
        self.split_res = split_res

        self.msa_encoder = self._make_msa_encoder(num_block=num_block)

    def _make_msa_encoder(self, num_block):

        layers = []
        for index in range(num_block):
            layer = Evoformer(msa_channel=self.msa_channel,
                              msa_nhead=self.msa_nhead,
                              pair_channel=self.pair_channel,
                              pair_nhead=self.pair_nhead,
                              dropout_p=self.dropout_p,
                              dropout_p2d=self.dropout_p2d,
                              with_column_attn=self.with_column_attn,
                              tied_attn=self.tied_attn,
                              split_res=self.split_res,
                              split_seq=self.split_seq,
                              is_causal=False)

            layers.append(('msa_block' + str(index), layer))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, msa, pair):

        for layer in self.msa_encoder:
            msa, pair = layer(msa, pair)
 
        return msa, pair


class MaskedMSAHead(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.RMSNorm(embed_dim)
        self.gelu = nn.GELU(approximate='none')
        self.bef_layer_norm = nn.RMSNorm(embed_dim)

        self.linear = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = x.squeeze(0) 
        x = self.bef_layer_norm(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        x = self.linear(x)

        return x


class ConfidenceHead(nn.Module):

    def __init__(self,
                 nblock,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 dropout_p,
                 dropout_p2d,
                 split_res,
                 split_seq=None,
                 no_bins_lddt=50,
                 min_dis=3.25,
                 max_dis=20.75,
                 no_bins_dis=16,
                 np_bins_tm=64,
                 tied_attn=False,
                 with_column_attn=False,
                 ):
        super(ConfidenceHead, self).__init__()

        self.min_dis = min_dis
        self.max_dis = max_dis
        self.no_bin_dis = no_bins_dis

        self.evoformer = MSAEncoder(num_block=nblock,
                                    msa_channel=atom_channel,
                                    msa_nhead=atom_nhead,
                                    pair_channel=pair_channel,
                                    pair_nhead=pair_nhead,
                                    tied_attn=tied_attn,
                                    dropout_p=dropout_p,
                                    dropout_p2d=dropout_p2d,
                                    with_column_attn=with_column_attn,
                                    split_res=split_res,
                                    split_seq=split_seq)

        self.trans_dis = nn.Linear(no_bins_dis, pair_channel, bias=False)

        self.out_pae = nn.Sequential(
                                     nn.Linear(pair_channel, np_bins_tm, bias=False),
                                     )
        
        self.out_plddt = nn.Sequential(
                                       nn.Linear(atom_channel, no_bins_lddt, bias=False),
                                       ) 

    def forward(self, seq, pair, coords):
        seq = copy.deepcopy(seq[:, 1:2, ...].detach())
        pair = copy.deepcopy(pair.detach())
        coords = copy.deepcopy(coords[:, 1, ...].detach())

        bins = torch.linspace(self.min_dis, self.max_dis, self.no_bin_dis - 1, dtype=coords.dtype, device=coords.device,
                              requires_grad=False, )
        bins = torch.cat([bins.new_tensor([0.0]), bins], dim=-1)
        upper = torch.cat([bins[1:], bins.new_tensor([1e10])], dim=-1)

        d = torch.cdist(coords, coords).unsqueeze(-1)
        d = ((d > bins) * (d < upper)).type(coords.dtype)  # balln

        pair += self.trans_dis(d)

        for layer in self.evoformer.msa_encoder:
            seq, pair = layer(seq, pair)

        pae = self.out_pae(pair)
        plddt = self.out_plddt(seq)

        return pae, plddt
