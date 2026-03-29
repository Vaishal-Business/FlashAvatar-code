# -*- coding: utf-8 -*-
# Configuration for FLAME-NumPy Engine
# Optimized for the FLAME 2023 Open Model (CC-BY-4.0)

import argparse
from pathlib import Path
from yacs.config import CfgNode as CN

cfg = CN()

# --- Model Paths ---
# Updated to point to the commercially usable open model
cfg.model_path = "model_data/flame2023_Open.pkl"
cfg.flame_lmk_path = "model_data/flame_static_embedding.pkl"
cfg.flame_mediapipe_path = "flame/mediapipe/mediapipe_landmark_embedding.npz"

# --- Model Dimensions ---
# Standard FLAME 2023 Open settings
cfg.num_shape_params = 100 
cfg.num_exp_params = 50
cfg.tex_params = 140

# --- Execution Settings ---
cfg.actor = ''
cfg.config_name = ''
cfg.image_size = [512, 512]  # height, width
cfg.fps = 25
cfg.save_folder = './output/'

# --- Optimization Parameters (If using an optimizer with flame_numpy) ---
cfg.rotation_lr = 0.2
cfg.translation_lr = 0.003
cfg.optimize_shape = False
cfg.optimize_jaw = False

# --- Weights for Loss Functions ---
cfg.w_pho = 350
cfg.w_lmks = 7000
cfg.w_lmks_68 = 1000
cfg.w_lmks_lid = 1000
cfg.w_lmks_mouth = 15000
cfg.w_lmks_iris = 1000
cfg.w_lmks_oval = 2000

cfg.w_exp = 0.02
cfg.w_shape = 0.3
cfg.w_tex = 0.04
cfg.w_jaw = 0.05

def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    """
    Standard argument parser for the NumPy-based pipeline.
    """
    parser = argparse.ArgumentParser(description="FLAME-NumPy Inference")
    parser.add_argument('--cfg', type=str, help='Path to override config file', default=None)
    parser.add_argument('--actor', type=str, help='Actor name/folder', default='')
    
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = args.cfg
        cfg.config_name = Path(args.cfg).stem
    
    if args.actor:
        cfg.actor = args.actor

    return cfg

def parse_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg.cfg_file = cfg_file
    cfg.config_name = Path(cfg_file).stem
    return cfg