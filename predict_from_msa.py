# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

import torch
from torch.utils.data import DataLoader
import time
from config import get_cfg
import json
import os
from itertools import zip_longest
import argparse
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataloader import MonomerDataset
from model import ProtMonomer
from utils.data_utils import compute_plddt, check_input
from utils.save_struc import to_pdb

def cpu_store(atom_coords, bfactors, seq_list, idx_list, save_file,ftype):
    to_pdb(atom_coords, bfactors, seq_list, idx_list, save_file, ftype)

def predict(cfg, dataloader):

    model = ProtMonomer(blocks_msa_encoder=cfg.model.blocks_msa_encoder,
                        groups_msa_encoder=cfg.model.groups_msa_encoder,
                        blocks_structure_decoder=cfg.model.blocks_structure_decoder,
                        blocks_confidence=cfg.model.blocks_confidence,
                        msa_channel=cfg.model.msa_channel,
                        msa_nhead=cfg.model.msa_nhead,
                        atom_channel=cfg.model.atom_channel,
                        atom_nhead=cfg.model.atom_nhead,
                        pair_channel=cfg.model.pair_channel,
                        pair_nhead=cfg.model.pair_nhead,
                        dropout_p=cfg.model.dropout_p,
                        dropout_p2d=cfg.model.dropout_p2d,
                        num_atom=cfg.model.num_atom,
                        split_seq=cfg.split_seq,
                        split_atom=cfg.split_atom,
                        )

    checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    weight_name = Path(weight_file).stem

    pending = []
    for d, monomer_dict in enumerate(dataloader):

        entry_path = os.path.join(cfg.save_path, monomer_dict['target_name'])
        if not os.path.exists(entry_path):
            os.mkdir(entry_path)
        json_file = os.path.join(entry_path, f'log_{weight_name}_{seed}.json')
        log_dict = {}
        
        for key, value in monomer_dict.items():
            if type(value) is torch.Tensor:
                monomer_dict[key] = value.to(device)

        num_msa, len_seq = monomer_dict['label_msa'].shape
        t1 = time.time()
        preds = model(monomer_dict)
        t2 = time.time()

        for key, value in monomer_dict.items():
            if type(value) == torch.Tensor:
                monomer_dict[key] = value.cpu()
        monomer_dict['sel_idx'] = monomer_dict['sel_idx'].cpu().numpy()

        pred_coords = preds['pred_coords'].cpu()
        pred_lddts = compute_plddt(preds['pred_plddts']).cpu()
        
        for idx, (coord_iter, plddt_iter) in enumerate(zip_longest(pred_coords, pred_lddts)):
            for idx_seed, (coords, plddt) in enumerate(zip_longest(coord_iter, plddt_iter)):
                
                save_file = os.path.join(entry_path, f'{seed}_{num_msa}_{weight_name}.pdb')

                f = cpu_pool.submit(cpu_store,
                                 	coords,
                                 	plddt.squeeze(),
                                 	monomer_dict['fasta_seq'],
                                 	monomer_dict['sel_idx'],
                                 	save_file,
                                    cfg.ftype)
                pending.append(f)
            
        log_dict['timing'] = f'{t2-t1:6.2f}'
        log_dict['plddt'] = [f'{i.item():6.1f}' for i in torch.mean(pred_lddts, dim=[1, 2, 3])]
        log_dict['len_seq'] = f"{len(monomer_dict['fasta_seq']):6d}"
        log_dict['target'] = monomer_dict['target_name']
        json.dump(log_dict, open(json_file,'w'))

        print(log_dict)
    
    for future in as_completed(pending):
        result = future.result()
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ProtMonomer')
    parser.add_argument('-msa_path', '--msa_path',
                        default=None,
                        type=str,
                        help="/mnt/example",
                        required=False)
    
    parser.add_argument('-msa_file', '--msa_file', 
                        default=None,
                        type=str, 
                        help='/mnt/example.a3m',
                        required=False)

    parser.add_argument('-save_path', '--save_path',
                        default=None,
                        type=str,
                        help='path to the save directory',
                        required=True)
    
    parser.add_argument('-weight', '--weight_file',
                        default=None,
                        type=str,
                        help='path to model weights',
                        required=True)

    parser.add_argument('-ftype', '--file_type',
                        default='cif',
                        type=str,
                        help='pdb or cif',
                        required=True)

    parser.add_argument('-device', '--device',
                        default='cuda:0',
                        type=str,
                        help='device to run the model',
                        required=False)
    
    parser.add_argument('-seed', '--random_seed',
                        default=42,
                        type=int,
                        help='random seed',
                        required=False)

    parser.add_argument('-ncpu', '--num_cpu',
                        default=8,
                        type=int,
                        help='threads',
                        required=False)
    
    parser.add_argument('-split_seq', '--split_seq',
                        default=0,
                        type=int,
                        help='split_seq',
                        required=False)
    
    parser.add_argument('-split_atom', '--split_atom',
                        default=0,
                        type=int,
                        help='split_atom',
                        required=False)
    
    parser.add_argument('-num_iter', '--num_iter',
                        default=8,
                        type=int,
                        help='num_iter',
                        required=False)

    parser.add_argument('-filter', '--filter_msa', 
                        default=False, 
                        type=bool, 
                        help='Filter MSA with maxhamming')

    parser.add_argument('-max_homolog', '--max_homolog',
                        default=8192,
                        type=int,
                        help='max homolog',
                        required=False)
    
    args = parser.parse_args()
    cfg = get_cfg()

    msa_path = args.msa_path
    msa_file = args.msa_file
    save_path = args.save_path
    weight_file = args.weight_file
    ftype = args.file_type
    device = torch.device(args.device)
    seed = args.random_seed
    ncpu = args.num_cpu
    split_seq = args.split_seq
    split_atom = args.split_atom
    num_iter = args.num_iter
    filter_msa = args.filter_msa
    max_homolog = args.max_homolog
    
    # init config
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.set_num_threads(ncpu)
    cpu_pool = ProcessPoolExecutor(max_workers=ncpu)
    
    cfg.weight_file = weight_file
    cfg.save_path = save_path
    cfg.ftype = ftype
    cfg.device = device
    
    # inference
    if split_seq>0:
        cfg.split_seq = split_seq
    if split_atom>0:
        cfg.split_atom = split_atom

    # dataset
    if msa_path and msa_file:
        raise ValueError('Only msa_path or msa_file')

    if msa_file:
        targets_list=[msa_file]
    elif msa_path:
        targets_list=[os.path.join(msa_path,i) for i in os.listdir(msa_path)]
    else:
        raise ValueError('At least msa_path or msa_file')
        
    targets_list = check_input(targets_list)
    if len(targets_list)<1:
        raise ValueError('number of inputs==0')
        
    print('inputs:\n', '\n'.join(targets_list))

    dataset = MonomerDataset(targets_list=targets_list,
                             max_token=4000*8192,
                             max_homolog=max_homolog,
                             filter_msa=filter_msa,
                             num_structure_recycle=num_iter)

    dataloader = DataLoader(dataset, batch_size=None)
    if not os.path.exists(save_path):
        print(f'mkdir {save_path}')
        os.mkdir(save_path)
        
    predict(cfg, dataloader)