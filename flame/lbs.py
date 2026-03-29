# -*- coding: utf-8 -*-
# This configuration module is part of the FLAME-NumPy project.
# Optimized for use with the FLAME 2023 Open Model (CC-BY-4.0).

import argparse
from pathlib import Path
from yacs.config import CfgNode as CN

# Initialize config node
cfg = CN()

# -----------------------------------------------------------------------------
# Model Paths
# -----------------------------------------------------------------------------
# Defaults point to the commercially usable open-source assets
cfg.model_path = "models/flame2023_open.pkl"
cfg.static_lmk_path = "models/flame_static_embedding.pkl"
cfg.mediapipe_lmk_path = "models/mediapipe_landmark_embedding.npz"

# -----------------------------------------------------------------------------
# Model Dimensions (Adjusted for FLAME 2023 Open Defaults)
# -----------------------------------------------------------------------------
cfg.num_shape_params = 100
cfg.num_exp_params = 50
cfg.num_tex_params = 140

# -----------------------------------------------------------------------------
# Inference & Image Settings
# -----------------------------------------------------------------------------
cfg.image_size = [512, 512]  # [height, width]
cfg.fps = 25
cfg.actor_name = ''
cfg.output_dir = './output/'

# -----------------------------------------------------------------------------
# Optimization Hyperparameters (Optional for NumPy-based solvers)
# -----------------------------------------------------------------------------
cfg.learning_rate_rotation = 0.2
cfg.learning_rate_translation = 0.003
cfg.optimize_shape = False
cfg.optimize_jaw = False

# -----------------------------------------------------------------------------
# Loss Weights (Retained for optimization logic)
# -----------------------------------------------------------------------------
cfg.w_photometric = 350.0
cfg.w_landmarks_68 = 1000.0
cfg.w_landmarks_mediapipe = 7000.0
cfg.w_landmarks_mouth = 15000.0
cfg.w_landmarks_iris = 1000.0
cfg.w_shape_reg = 0.3
cfg.w_exp_reg = 0.02
cfg.w_tex_reg = 0.04

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return cfg.clone()

def update_cfg(config, cfg_file):
    """Update config from a yaml file."""
    config.merge_from_file(cfg_file)
    return config.clone()

def parse_args():
    """Argument parser for CLI overrides."""
    parser = argparse.ArgumentParser(description="FLAME NumPy Inference Configuration")
    parser.add_argument('--cfg', type=str, help='Path to a .yaml config file', default=None)
    parser.add_argument('--actor', type=str, help='Actor name override', default=None)
    
    args = parser.parse_args()
    
    current_cfg = get_cfg_defaults()
    
    if args.cfg:
        current_cfg = update_cfg(current_cfg, args.cfg)
        current_cfg.config_name = Path(args.cfg).stem
    
    if args.actor:
        current_cfg.actor_name = args.actor
        
    return current_cfg

def load_config_by_path(cfg_file):
    """Load config directly from a file path."""
    current_cfg = get_cfg_defaults()
    current_cfg = update_cfg(current_cfg, cfg_file)
    current_cfg.config_name = Path(cfg_file).stem
    return current_cfg