# -*- coding = utf-8 -*-
"""
Created on 2026/9/8
author: yunda_si@ucac.ac.cn
"""

import matplotlib.pyplot as plt
from Bio import PDB
from sklearn.decomposition import PCA
import numpy as np
import argparse
import torch
import os

def extract_ca(structure_file, ftype, atom='CA'):
    if ftype == 'pdb':
        parser = PDB.PDBParser()
    else:
        parser = PDB.MMCIFParser()
    structure = parser.get_structure('structure', structure_file)

    b_factors = []
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if atom in residue:
                    ca_atom = residue[atom]
                    b_factors.append(ca_atom.get_bfactor())
                    coords.append(ca_atom.coord)
                else:
                    raise
    return np.array(b_factors), np.array(coords)


def cal_contact(preds_path, ftype='pdb', dis_cutoff=8):
    all_plddts = []
    all_contacts = []

    for name in os.listdir(preds_path):
        pred_file = os.path.join(preds_path, name)

        if not name.endswith(ftype):
            continue

        bfactor, coords = extract_ca(pred_file, ftype)
        contact_map = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords)).numpy()
        contact_map[contact_map < dis_cutoff] = 1
        contact_map[contact_map != 1] = 0
        all_contacts.append(contact_map)
        all_plddts.append(np.mean(bfactor))

    return all_contacts, all_plddts

def visualization(pdb_path, save_img_file, ftype):

    fontdict = {'weight': 'book', 'family': 'sans-serif',
                'stretch': 'ultra-expanded', 'size': 'large',
                'style': 'normal'}

    gridspec_kw = {'left': 0.17, 'bottom': 0.18, 'right': 0.99, 'top': 0.99}

    def scatter():
        fig, ax = plt.subplots(figsize=(5.0, 4.8), dpi=600, gridspec_kw=gridspec_kw, layout=None)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        return fig, ax

    all_contacts, all_plddts = cal_contact(pdb_path, ftype=ftype, dis_cutoff=8)

    np.random.seed(42)
    pca = PCA(n_components=2)
    plt.style.use('default')
    a = np.stack(all_contacts)
    pca_result = pca.fit_transform(a.reshape(a.shape[0], -1))

    fig, ax = scatter()
    im = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=np.array(all_plddts)/100, vmin=0.5, vmax=0.9, cmap='bwr', alpha=0.9)
    cax = ax.inset_axes([0.1, 0.85, 0.3, 0.04])
    cb = fig.colorbar(im, cax=cax, fraction=0.04, pad=0.02, orientation="horizontal", anchor=(0.6, 2.9), aspect=5)
    cb.set_label(label='pLDDT', labelpad=-35, rotation=0)
    cb.set_ticks([0.6, 0.8])

    plt.xlabel('PC1', fontdict=fontdict)
    plt.ylabel('PC2', fontdict=fontdict)

    plt.savefig(save_img_file)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PCA')
    parser.add_argument('-pdb_path', '--pdb_path',
                        default=None,
                        type=str,
                        help="/mnt/example",
                        required=False)

    parser.add_argument('-save_img_file', '--img_file',
                        default=None,
                        type=str,
                        help='/mnt/example.png',
                        required=False)

    parser.add_argument('-ftype', '--ftype',
                        default='pdb',
                        type=str,
                        help='pdb or cif',
                        required=False)

    args = parser.parse_args()

    pdb_path = args.pdb_path
    save_img_file = args.save_img_file
    ftype = args.ftype

    visualization(pdb_path, save_img_file, ftype)