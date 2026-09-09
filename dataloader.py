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
                 filter=True,
                 maxhamming_bin='./bin/maxhamming',
                 ):

        self.max_token = max_token
        self.max_homolog = max_homolog
        self.filter = filter
        self.maxhamming_bin = maxhamming_bin

    def __call__(self,
                 msa=None,
                 ):

        msa = self.msa_encoding(msa)

        return msa

    def msa_encoding(self, msa):

        num_row, num_column = msa.shape
        num_sel_msa = min(int(num_row), self.max_homolog) + 1
        upper_num_msa = min(int(self.max_token/num_column), num_sel_msa)

        if self.filter:
            sim_msa, left_msa = self.filter_msa(msa, 0.4, 0.9, 21) #0.4
            if len(sim_msa)>=upper_num_msa:
                msa = sim_msa[:upper_num_msa]
            else:
                sim_msa, left_msa = self.filter_msa(msa, 0.5, 0.95, 21)
                if len(sim_msa) >= upper_num_msa:
                    msa = sim_msa[:upper_num_msa]
                else:
                    sim_msa, left_msa = self.filter_msa(msa, 0.6, 0.95, 21)
                    msa = np.concatenate([sim_msa, left_msa], 0)[:upper_num_msa]
        else:
            msa = msa[:upper_num_msa]

        msa = torch.tensor(msa, dtype=torch.int64)

        return msa


    def filter_msa(self, msa, gap_cov=0.5, identity=0.95, gap_id=21):
        msa = torch.from_numpy(msa.copy())

        num_seq, num_aa = msa.shape
        identity_array = num_aa - torch.cdist(msa.float()[:1], msa.float(), p=0)
        sel_idx = torch.sort(identity_array, descending=True).indices
        msa = msa[sel_idx[0]]

        sel_idx = torch.sort(torch.sum(msa != gap_id, dim=-1), descending=True).indices
        sel_idx = torch.tensor([0] + [i for i in sel_idx if i != 0])
        msa = msa[sel_idx]

        num_seq, num_aa = msa.shape
        msa_idx = (torch.sum(msa == gap_id, dim=-1) < num_aa*gap_cov)
        total_idx = self.maxhamming(msa[msa_idx], identity)


        mask = torch.ones(num_seq, dtype=torch.bool)
        mask[total_idx] = False

        return msa[total_idx].cpu().numpy(), msa[mask].cpu().numpy()

    def maxhamming(self, msa, identity):
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
            os.system(f'{self.maxhamming_bin} -i {fasta_file} -o {out_file} -n {max(len(msa)-1,1)} -t {1-identity} -f index >/dev/null 2>&1')
            if os.path.exists(out_file):
                index = torch.from_numpy(np.loadtxt(out_file).astype(np.int_))
                if index.shape == torch.Size([]):
                    index = torch.tensor([0]).int()
            else:
                index = torch.tensor([0]).int()

        return index


class MonomerDataset(Dataset):

    def __init__(self,
                 targets_list=None,
                 max_token=2 ** 16,
                 max_homolog=8192,
                 filter_msa=True,
                 num_structure_recycle=8,
                 save_last=True,
                 maxhamming_bin='./bin/maxhamming'
                 ):

        self.targets_list = targets_list
        self.num_structure_recycle = num_structure_recycle
        self.save_last = save_last
        self.get_msa_feature = MSAFeature(max_token=max_token,
                                          max_homolog=max_homolog,
                                          filter=filter_msa,
                                          maxhamming_bin=maxhamming_bin,
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
                           'plddt_idx':None,
                           }

        if self.save_last:
            monomer_feature['plddt_idx'] = [self.num_structure_recycle-1]
        else:
            monomer_feature['plddt_idx'] = [i for i in range(self.num_structure_recycle)]

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
        sel_idx = torch.arange(len(fasta_seq))
        monomer_feature['sel_idx'] = sel_idx
        monomer_feature['fasta_seq'] = fasta_seq
        seq_np = np.array([resc.restype_order_with_x[i] for i in fasta_seq])
        monomer_feature['atom37_atom_exists'] = torch.from_numpy(resc.RESTYPE_ATOM37_MASK[seq_np]).long()

        # structure
        peptide_coords = [self.buildpeptide(fasta_seq)]
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
