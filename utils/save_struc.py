# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
from np import residue_constants as resc

pdbio = PDBIO()
mmcifio = MMCIFIO()


def set_residue(coords, bfactor, res_name_3, save_idx, segid=' '):
    residue = Residue((' ', save_idx, ' '), res_name_3, segid)
    for atom_name in resc.residue_atoms[res_name_3]:
        atom_posi = coords[resc.atom_order[atom_name]]
        residue.add(Atom(atom_name, atom_posi, bfactor, 1.0, ' ', atom_name.center(4), 0, element=atom_name[:1]))

    return residue

def to_pdb(atom_coords, bfactors, fasta_seq, sel_idx, save_file, ftype='cif', chain_id='A'):

    new_structure = Structure('Structure')
    new_model = Model(1)
    atom_coords = atom_coords.squeeze()

    new_chain = Chain(chain_id)
    for coords, bfactor, res_name1, save_idx in zip(atom_coords, bfactors, fasta_seq, sel_idx):
        res_name_3 = resc.restype_1to3[res_name1]
        
        residue = set_residue(coords.numpy(), bfactor.numpy(), res_name_3, save_idx.item()+1, segid=' ')
        new_chain.add(residue)

    new_model.add(new_chain)

    new_structure.add(new_model)
    if ftype == 'pdb':
        try:
            save_file = save_file.replace('.cif', '.pdb')
            pdbio.set_structure(new_structure)
            pdbio.save(save_file)
        except:
            raise
    else:
        save_file = save_file.replace('.pdb', '.cif')
        mmcifio.set_structure(new_structure)
        mmcifio.save(save_file)