#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Dec  7 20:03:14 2024
@author: yunda_si
"""

import os.path
import numpy as np
import torch
import np.residue_constants as resc
from PeptideBuilder import Geometry
import PeptideBuilder
import tempfile
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset


def msa2np(msa):
    msa_np = []
    for seq in msa:
        seq_np = []
        for aa in seq:
            if aa in resc.HHBLITS_AA_TO_ID:
                seq_np.append(resc.HHBLITS_AA_TO_ID[aa])
            else:
                seq_np.append(resc.HHBLITS_AA_TO_ID['-'])
        msa_np.append(seq_np)
    msa_np = np.array(msa_np, dtype=np.int8)
    
    return msa_np


class MSAFeature(object):
    def __init__(self,
                 max_token=2 ** 16,
                 max_homolog=8192,
                 filter = True
                 ):

        self.max_token = max_token
        self.max_homolog = max_homolog
        self.filter = filter
        self.maxhamming_bin =  './bin/maxhamming'

    def __call__(self,
                 msa=None,
                 ):

        msa = self.msa_encoding(msa)

        return msa

    def msa_encoding(self, msa):

        num_row, num_column = msa.shape

        num_sel_msa = min(int(num_row), self.max_homolog) + 1

        sel_row_index = np.random.permutation(num_row)
        sel_row_index = sel_row_index[sel_row_index!=0]
        sel_row_index = np.concatenate([np.array([0]), sel_row_index])

        upper_num_msa = min(int(self.max_token/num_column), num_sel_msa)

        if self.filter:
            msa = msa[sel_row_index]
            sim_msa, left_msa = self.filter_msa(msa, 0.5, 0.95, 21, upper_num_msa)
            msa = np.concatenate([sim_msa, left_msa], 0)[:upper_num_msa]
        else:
            msa = msa[sel_row_index][:upper_num_msa]

        msa = torch.tensor(msa, dtype=torch.int64)

        return msa


    def filter_msa(self, msa, gap_cov=0.5, identity=0.95, gap_id=21, max_seq=8192*4):
        msa = torch.from_numpy(msa.copy())
        num_seq, num_aa = msa.shape
        identity_array = num_aa - torch.cdist(msa.float()[:1], msa.float(), p=0)
        sel_idx = (torch.sort(identity_array, descending=True).indices)[identity_array!= num_aa]
        sel_idx = torch.concat([torch.tensor([0]), sel_idx])
        msa = msa[sel_idx]
        num_seq, num_aa = msa.shape

        msa_idx = (torch.sum(msa == gap_id, dim=-1) < num_aa * gap_cov)

        total_idx = self.maxhamming(msa[msa_idx], identity, max_seq)

        mask = torch.ones(num_seq, dtype=torch.bool)
        mask[total_idx] = False

        return msa[total_idx].cpu().numpy(), msa[mask].cpu().numpy()

    def maxhamming(self, msa, identity, max_seq):
        mapping = [resc.ID_TO_HHBLITS_AA[i] for i in range(22)]
        result = np.array(mapping)[msa.numpy()]

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tempdirname:
            tempdir = Path(tempdirname)
            file_name = str(id(result))
            fasta_file = tempdir / f"{file_name}.a3m"

            with open(fasta_file, 'w') as f:
                for idx, row in enumerate(result):
                    f.write(f'>{idx}\n')
                    seq = ''.join(row)
                    f.write(f'{seq}\n')

            out_file = tempdir / f"{file_name}.idx"
            os.system(f'{self.maxhamming_bin} -r -i {fasta_file} -o {out_file} -n {max_seq} -t {1-identity} -f index >/dev/null 2>&1')
            index = torch.from_numpy(np.loadtxt(out_file).astype(np.int_))

            if index.shape == torch.Size([]):
                index = index.unsqueeze(0)

        return index


class MonomerDataset(Dataset):

    def __init__(self,
                 targets_list=None,
                 max_token=2 ** 16,
                 max_homolog=8192,
                 filter_msa=True,
                 num_structure_recycle=8,
                 ):

        self.targets_list = targets_list
        self.num_structure_recycle = num_structure_recycle
        self.get_msa_feature = MSAFeature(max_token=max_token,
                                          max_homolog=max_homolog,
                                          filter=filter_msa,
                                          )

    def __getitem__(self, idx):

        msa_file = self.targets_list[idx]
        target_name = Path(msa_file).stem

        monomer_feature = {'target_name':target_name,
                           'num_structure_recycle':self.num_structure_recycle,
                           'label_msa':None,
                           'sel_idx': None,
                           'atom37_atom_exists':None,
                           'label_seq':None,
                           'peptide_coords':None,
                           }

        # msa
        parsed_msa = SeqIO.parse(msa_file, 'fasta')
        homologs_list = []
        for record in parsed_msa:
            homologs_list.append(record.seq)
        msa_np = msa2np(homologs_list)
        msa_np = self.get_msa_feature(msa=msa_np)
        monomer_feature['label_msa'] = msa_np
        
        # seq
        fasta_seq = homologs_list[0].strip()
        print(fasta_seq)
        sel_idx = torch.arange(len(fasta_seq))
        monomer_feature['sel_idx'] = sel_idx
        monomer_feature['fasta_seq'] = fasta_seq
        seq_np = np.array([resc.restype_order_with_x[i] for i in fasta_seq])
        monomer_feature['atom37_atom_exists'] = torch.from_numpy(resc.restype_atom37_mask[seq_np]).long()

        # structures
        peptide_coords = [self.buildpeptide(fasta_seq[sel_idx[0].item():sel_idx[-1].item()+1])]
        peptide_coords = torch.stack(peptide_coords)/10
        monomer_feature['peptide_coords'] = peptide_coords

        return monomer_feature

    def buildpeptide(self, seq):
        angle = np.random.rand(len(seq), 3) * 720 - 360
        geo = Geometry.geometry(seq[0])
        geo.phi = angle[0,0]
        geo.psi_im1 = angle[0,1]
        geo.omega=angle[0, 2]
        structure = PeptideBuilder.initialize_res(geo)
        for idx_aa, aa in enumerate(seq[1:], start=1):
            PeptideBuilder.add_residue(structure, aa, phi=angle[idx_aa,0], psi_im1=angle[idx_aa,1], omega=angle[idx_aa, 2])

        atom_coords = np.zeros((len(seq), resc.atom_type_num, 3))
        chain = [i for i in structure[0]][0]
        for residue_idx, residue in enumerate(chain):
            for atom in residue:
                if atom.name not in resc.atom_types:
                    continue
                atom_coords[residue_idx, resc.atom_order[atom.name]] = atom.coord

        atom_coords = torch.tensor(atom_coords).float()
        atom_coords -= torch.mean(atom_coords[:,:3,:], dim=(0,1))

        return atom_coords

    def __len__(self):
        return len(self.targets_list)