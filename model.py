# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:28:07 2020

@author: yunda_si
"""

import torch.nn as nn
import torch
from utils.data_utils import localrigids
from modules import MSAEncoder, MaskedMSAHead, InputEmbedder, ConfidenceHead
from structure_module import StructureModule
from utils.rigid_utils import rot_to_quat

class ProtMonomer(nn.Module):

    def __init__(self,
                 blocks_msa_encoder,
                 groups_msa_encoder,
                 blocks_structure_decoder,
                 blocks_confidence,
                 msa_channel,
                 msa_nhead,
                 atom_channel,
                 atom_nhead,
                 pair_channel,
                 pair_nhead,
                 nums_aa=23,
                 num_atom=37,
                 dropout_p=0.0,
                 dropout_p2d=0.0,
                 dist_bin=64,
                 tm_bin=64,
                 plddt_bin=50,
                 split_seq=None,
                 split_atom=None,
                 ):

        super(ProtMonomer, self).__init__()

        self.dropout_p = dropout_p
        self.num_atom = num_atom
        self.msa_channel = msa_channel
        self.pair_channel = pair_channel
        self.dist_bin = dist_bin
        self.split_seq = split_seq

        self.embed_struc = nn.Embedding(num_atom + 1, atom_channel)

        self.input_embedder = InputEmbedder(
                                            msa_channel=msa_channel,
                                            pair_channel=pair_channel,
                                            nums_aa=nums_aa
                                            )

        self.trans_seq = nn.Sequential(
                                       nn.RMSNorm(atom_channel)
                                       )

        self.msa_encoder = nn.ModuleList(
                                         [MSAEncoder(num_block=blocks_msa_encoder,
                                                     msa_channel=msa_channel,
                                                     msa_nhead=msa_nhead,
                                                     pair_channel=pair_channel,
                                                     pair_nhead=pair_nhead,
                                                     tied_attn=False,
                                                     dropout_p=dropout_p,
                                                     dropout_p2d=dropout_p2d,
                                                     with_column_attn=True,
                                                     split_seq=split_seq)
                                                     for i in range(groups_msa_encoder)]
                                         )

        self.structure_block = StructureModule(
                                               msa_channel=msa_channel,
                                               atom_channel=atom_channel,
                                               pair_channel=pair_channel,
                                               atom_nhead=atom_nhead,
                                               block_num=blocks_structure_decoder,
                                               dropout_p=0.0,
                                               num_atom=num_atom,
                                               split_atom=split_atom,
                                               split_seq=split_seq
                                               )

        self.dist_out = nn.Sequential(
                                      nn.RMSNorm(pair_channel),
                                      nn.Linear(pair_channel, pair_channel),
                                      nn.GELU(),
                                      nn.RMSNorm(pair_channel),
                                      nn.Linear(pair_channel, dist_bin)
                                      )

        self.maskmsa_out = MaskedMSAHead(
                                         embed_dim=msa_channel,
                                         output_dim=nums_aa
                                         )

        self.confidence_block = ConfidenceHead(
                                               nblock=blocks_confidence,
                                               atom_channel=atom_channel,
                                               atom_nhead=atom_nhead,
                                               pair_channel=pair_channel,
                                               pair_nhead=pair_nhead,
                                               no_bins_lddt=plddt_bin,
                                               np_bins_tm=tm_bin,
                                               dropout_p=dropout_p,
                                               dropout_p2d=dropout_p2d,
                                               split_seq=split_seq
                                               )

    def forward(self, protein):

        pred_coords = []
        pred_plddts = []
        pred_pae = []

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            msa, pair = self.input_embedder(protein)
            for idx_layer, layer in enumerate(self.msa_encoder):
                msa, pair = layer(msa, pair)

        b, n, l, c = msa.shape
        coords = torch.permute(protein['peptide_coords'], [0, 2, 1, 3]).float()

        basis, _ = localrigids(torch.transpose(coords, 1, 2), svd_align=False)
        quat = rot_to_quat(basis)

        struc_emb = torch.repeat_interleave(torch.repeat_interleave(torch.arange(1, 38, device=coords.device).unsqueeze(1), l, 1).unsqueeze(0),coords.shape[0], 0)
        struc_emb = self.embed_struc((struc_emb * (protein["atom37_atom_exists"].T)).long()).float()

        for i_struc_cycle in range(8):
            atom_mask = protein["atom37_atom_exists"].T.unsqueeze(0).unsqueeze(-1)
            if i_struc_cycle > 0:
                struc_emb = self.trans_seq(struc_emb)
            coords, struc_emb, quat = self.structure_block(struc_emb, msa.float(), pair.float(), coords, atom_mask, quat)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            if self.split_seq is not None:
                pae = []
                plddt = []
                for i in range(0,coords.shape[0],1):
                    temp1, temp2 = self.confidence_block(struc_emb[i:i+1], pair, coords[i:i+1]*10)
                    # pae.append(temp1)
                    plddt.append(temp2)
                # pae = torch.concat(pae, 0)
                plddt = torch.concat(plddt, 0)
            else:
                pae, plddt = self.confidence_block(struc_emb, pair, coords*10)

            pred_plddts.append(plddt)
            # pred_pae.append(pae)
            pred_coords.append(coords * 10)

        pred_coords = torch.permute(torch.stack(pred_coords), [0, 1, 3, 2, 4])
        pred_plddts = torch.stack(pred_plddts)
        # pred_pae = torch.stack(pred_pae)

        return {'pred_coords': pred_coords,
                'pred_plddts': pred_plddts,
                'pred_pae': pred_pae,
                }

























