#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:11:56 2024

@author: yunda_si
"""

from ml_collections import config_dict


def get_cfg():
    config = config_dict.ConfigDict()

    config.weight_file = None
    config.init_seed = 42

    config.model = config_dict.ConfigDict()    
    config.model.blocks_msa_encoder = 2
    config.model.groups_msa_encoder = 16
    config.model.blocks_structure_decoder = 6
    config.model.msa_channel = 256
    config.model.msa_nhead = 8
    config.model.atom_channel = 256
    config.model.atom_nhead = 8
    config.model.pair_channel = 128
    config.model.pair_nhead = 4
    config.model.num_structure_recycle = 8
    config.model.num_atom = 37
    config.model.dropout_p = 0.15
    config.model.dropout_p2d = 0.25

    config.model.blocks_confidence = 3
    
    config.split_seq = None
    config.split_atom = None
    
    return config
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
