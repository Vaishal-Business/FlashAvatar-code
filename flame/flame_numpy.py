# flame_numpy.py
# Pure Python + NumPy reimplementation of core FLAME functionality.
# No PyTorch, no PyTorch3D. CPU friendly, fully vectorized.
# Compatible with FLAME 2023 Open model dict/pickle structure.

import os
import pickle
import numpy as np
from typing import Tuple, Optional

# -------------------------
# Rotation utilities (6D <-> rotation matrix)
# -------------------------
def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)

# -------------------------
# Linear Blend Skinning (Fully Vectorized)
# -------------------------
def lbs_numpy(betas: np.ndarray,
              full_pose_rot6d: np.ndarray,
              v_template: np.ndarray,
              shapedirs: np.ndarray,
              posedirs: np.ndarray,
              J_regressor: np.ndarray,
              parents: np.ndarray,
              weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    N = betas.shape[0]
    V = v_template.shape[0]
    J = parents.shape[0]

    # 1) Shape + expression deformation
    if shapedirs.ndim == 3:
        shapedelta = np.tensordot(shapedirs, betas.T, axes=([2], [0]))  # (V,3,N)
        shapedelta = np.transpose(shapedelta, (2, 0, 1))  # (N,V,3)
    else:
        shapedelta = (shapedirs @ betas.T).T.reshape(N, V, 3)

    v_shaped = v_template[None, :, :] + shapedelta  # (N, V, 3)

    # 2) Joint locations in rest pose
    joints = J_regressor @ v_shaped.transpose(0, 2, 1)  # (J, N, 3)
    joints = joints.transpose(1, 0, 2)  # (N, J, 3)

    # 3) Pose blend shapes
    rot_mats = rotation_6d_to_matrix(full_pose_rot6d.reshape(-1, 6)).reshape(N, -1, 3, 3)
    I3 = np.eye(3, dtype=rot_mats.dtype)[None, None, :, :]
    rel_rot = rot_mats - I3
    rel_rot_flat = rel_rot.reshape(N, -1)

    if posedirs is not None and posedirs.size > 0:
        pose_offsets = (posedirs @ rel_rot_flat.T).T.reshape(N, V, 3)
    else:
        pose_offsets = np.zeros_like(v_shaped)

    v_posed = v_shaped + pose_offsets

    # 4) Compute global transforms via forward kinematics (Vectorized over Batch)
    transforms = np.zeros((N, J, 4, 4), dtype=v_shaped.dtype)
    
    for j in range(J):
        p = parents[j]
        T_local = np.zeros((N, 4, 4), dtype=v_shaped.dtype)
        T_local[:, :3, :3] = rot_mats[:, j]
        T_local[:, 3, 3] = 1.0
        
        if p == -1:
            # Root joint absolute translation
            T_local[:, :3, 3] = joints[:, j]
            transforms[:, j] = T_local
        else:
            # Child joint relative translation
            T_local[:, :3, 3] = joints[:, j] - joints[:, p]
            transforms[:, j] = transforms[:, p] @ T_local

    # 5) Compute inverse rest transforms efficiently
    # Equivalent to T_global @ T_rest^-1 (subtracts rotated rest joint pos)
    transforms_rel = transforms.copy()
    R_global = transforms[:, :, :3, :3]
    t_global = transforms[:, :, :3, 3]
    
    j_rest_transformed = np.matmul(R_global, joints[:, :, :, None]).squeeze(-1)
    transforms_rel[:, :, :3, 3] = t_global - j_rest_transformed

    # 6) Skinning
    v_posed_h = np.concatenate([v_posed, np.ones((N, V, 1), dtype=v_posed.dtype)], axis=2)
    vertices = np.zeros_like(v_posed)
    
    for j in range(J):
        Tj = transforms_rel[:, j]  # (N,4,4)
        Tv = np.matmul(v_posed_h, Tj.transpose(0, 2, 1))  # (N,V,4)
        w = weights[None, :, j:j+1]  # (1,V,1)
        vertices += (Tv[..., :3] * w)

    return vertices, joints

# -------------------------
# Barycentric landmark interpolation (Vectorized)
# -------------------------
def vertices_to_landmarks(vertices: np.ndarray,
                          faces: np.ndarray,
                          lmk_faces_idx: np.ndarray,
                          lmk_bary_coords: np.ndarray) -> np.ndarray:
    N = vertices.shape[0]
    if lmk_faces_idx.ndim == 1:
        lmk_faces_idx = np.tile(lmk_faces_idx[None, :], (N, 1))
    if lmk_bary_coords.ndim == 2:
        lmk_bary_coords = np.tile(lmk_bary_coords[None, :, :], (N, 1, 1))

    # Vectorized gathering across the batch
    N_idx = np.arange(N)[:, None, None]
    face_vids = faces[lmk_faces_idx] # (N, L, 3)
    vcoords = vertices[N_idx, face_vids] # (N, L, 3, 3)
    
    landmarks = np.einsum('nlvi,nlv->nli', vcoords, lmk_bary_coords)
    return landmarks

# -------------------------
# FLAME wrapper
# -------------------------
class FlameNumpy:
    def __init__(self, flame_geom: dict, num_shape_params: int = 100, num_exp_params: int = 50):
        self.v_template = np.asarray(flame_geom['v_template']).astype(np.float32)
        self.faces = np.asarray(flame_geom['f']).astype(np.int64)
        
        shapedirs = np.asarray(flame_geom['shapedirs']).astype(np.float32)
        if shapedirs.ndim == 3:
            n_keep = num_shape_params + num_exp_params
            shapedirs = shapedirs[:, :, :n_keep]
        self.shapedirs = shapedirs

        self.posedirs = np.asarray(flame_geom['posedirs']).astype(np.float32)
        if self.posedirs.ndim == 3:
            self.posedirs = self.posedirs.reshape(-1, self.posedirs.shape[-1])

        self.J_regressor = np.asarray(flame_geom['J_regressor']).astype(np.float32)
        kintree = np.asarray(flame_geom['kintree_table'][0]).astype(np.int64)
        kintree[0] = -1
        self.parents = kintree
        self.weights = np.asarray(flame_geom['weights']).astype(np.float32)

        self.lmk_faces_idx = None
        self.lmk_bary_coords = None
        self.dynamic_lmk_faces_idx = None
        self.dynamic_lmk_bary_coords = None
        self.mp_lmk_faces_idx = None
        self.mp_lmk_bary_coords = None

        self.shape_params = np.zeros((1, shapedirs.shape[2] - num_exp_params), dtype=np.float32) if shapedirs.ndim == 3 else np.zeros((1, 0), dtype=np.float32)
        self.expression_params = np.zeros((1, num_exp_params), dtype=np.float32)

    def load_landmark_embeddings(self, lmk_dict: dict):
        self.lmk_faces_idx = np.asarray(lmk_dict['static_lmk_faces_idx']).astype(np.int64)
        self.lmk_bary_coords = np.asarray(lmk_dict['static_lmk_bary_coords']).astype(np.float32)
        self.dynamic_lmk_faces_idx = np.asarray(lmk_dict['dynamic_lmk_faces_idx']).astype(np.int64)
        self.dynamic_lmk_bary_coords = np.asarray(lmk_dict['dynamic_lmk_bary_coords']).astype(np.float32)

    def load_mediapipe_embedding(self, mp_dict: dict):
        self.mp_lmk_faces_idx = np.asarray(mp_dict['lmk_face_idx']).astype(np.int64)
        self.mp_lmk_bary_coords = np.asarray(mp_dict['lmk_b_coords']).astype(np.float32)
        self.mediapipe_idx = np.asarray(mp_dict['landmark_indices']).astype(np.int64)

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices: np.ndarray, full_pose_rot6d: np.ndarray, cameras: np.ndarray):
        N = vertices.shape[0]
        global_rot6d = full_pose_rot6d[:, 0:6]
        R = rotation_6d_to_matrix(global_rot6d)
        
        sy = np.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
        yaw = np.arctan2(-R[:, 2, 0], sy)
        yaw_deg = np.degrees(yaw)
        
        yaw_clamped = np.clip(np.round(-yaw_deg), -39, 39).astype(np.int32)
        yaw_idx = np.where(yaw_clamped < 0, (39 - yaw_clamped), yaw_clamped).astype(np.int32)
        
        if self.dynamic_lmk_faces_idx is None:
            return np.zeros((N, 0), dtype=np.int64), np.zeros((N, 0, 3), dtype=np.float32)
            
        dyn_faces = self.dynamic_lmk_faces_idx[yaw_idx]
        dyn_bcoords = self.dynamic_lmk_bary_coords[yaw_idx]
        return dyn_faces, dyn_bcoords

    def forward(self,
                shape_params: Optional[np.ndarray],
                cameras: np.ndarray,
                trans_params: Optional[np.ndarray] = None,
                rot_params_rot6d: Optional[np.ndarray] = None,
                neck_pose_rot6d: Optional[np.ndarray] = None,
                jaw_pose_rot6d: Optional[np.ndarray] = None,
                eye_pose_rot6d: Optional[np.ndarray] = None,
                expression_params: Optional[np.ndarray] = None,
                eyelid_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if shape_params is None:
            shape_params = np.tile(self.shape_params, (1, 1))
        if expression_params is None:
            expression_params = np.tile(self.expression_params, (shape_params.shape[0], 1))
        N = shape_params.shape[0]

        if trans_params is None:
            trans_params = np.zeros((N, 3), dtype=np.float32)
        if rot_params_rot6d is None:
            rot_params_rot6d = np.tile(matrix_to_rotation_6d(np.eye(3, dtype=np.float32))[None, :], (N, 1))
        if neck_pose_rot6d is None:
            neck_pose_rot6d = np.tile(matrix_to_rotation_6d(np.eye(3, dtype=np.float32))[None, :], (N, 1))
        if jaw_pose_rot6d is None:
            jaw_pose_rot6d = np.tile(matrix_to_rotation_6d(np.eye(3, dtype=np.float32))[None, :], (N, 1))
        if eye_pose_rot6d is None:
            eye_pose_rot6d = np.tile(matrix_to_rotation_6d(np.eye(3, dtype=np.float32))[None, :], (N, 2))

        betas = np.concatenate([shape_params, expression_params], axis=1)
        full_pose = np.concatenate([rot_params_rot6d, neck_pose_rot6d, jaw_pose_rot6d, eye_pose_rot6d], axis=1)

        vertices, joints = lbs_numpy(betas, full_pose, self.v_template, self.shapedirs, self.posedirs,
                                     self.J_regressor, self.parents, self.weights)

        if eyelid_params is not None and hasattr(self, 'r_eyelid') and hasattr(self, 'l_eyelid'):
            vertices = vertices + (self.r_eyelid[None, ...] * eyelid_params[:, 1:2, None]) + (self.l_eyelid[None, ...] * eyelid_params[:, 0:1, None])

        if self.lmk_faces_idx is None or self.lmk_bary_coords is None:
            lmk68 = np.zeros((N, 0, 3), dtype=vertices.dtype)
        else:
            lmk_faces_idx = np.tile(self.lmk_faces_idx[None, :], (N, 1))
            lmk_bary_coords = np.tile(self.lmk_bary_coords[None, :, :], (N, 1, 1))
            dyn_faces_idx, dyn_bcoords = self._find_dynamic_lmk_idx_and_bcoords(vertices, full_pose, cameras)
            all_faces_idx = np.concatenate([dyn_faces_idx, lmk_faces_idx], axis=1)
            all_bcoords = np.concatenate([dyn_bcoords, lmk_bary_coords], axis=1)
            lmk68 = vertices_to_landmarks(vertices, self.faces, all_faces_idx, all_bcoords)

        if self.mp_lmk_faces_idx is None or self.mp_lmk_bary_coords is None:
            mp = np.zeros((N, 0, 3), dtype=vertices.dtype)
        else:
            mp_faces_idx = np.tile(self.mp_lmk_faces_idx[None, :], (N, 1))
            mp_bcoords = np.tile(self.mp_lmk_bary_coords[None, :, :], (N, 1, 1))
            mp = vertices_to_landmarks(vertices, self.faces, mp_faces_idx, mp_bcoords)

        vertices = vertices + trans_params[:, None, :]
        lmk68 = lmk68 + trans_params[:, None, :]
        mp = mp + trans_params[:, None, :]

        return vertices, lmk68, mp

    def forward_geo(self, shape_params: np.ndarray, trans_params: Optional[np.ndarray] = None,
                    rot_params_rot6d: Optional[np.ndarray] = None, neck_pose_rot6d: Optional[np.ndarray] = None,
                    jaw_pose_rot6d: Optional[np.ndarray] = None, eye_pose_rot6d: Optional[np.ndarray] = None,
                    expression_params: Optional[np.ndarray] = None, eyelid_params: Optional[np.ndarray] = None) -> np.ndarray:
        vertices, _, _ = self.forward(shape_params, cameras=np.eye(3)[None, ...], trans_params=trans_params,
                                      rot_params_rot6d=rot_params_rot6d, neck_pose_rot6d=neck_pose_rot6d,
                                      jaw_pose_rot6d=jaw_pose_rot6d, eye_pose_rot6d=eye_pose_rot6d,
                                      expression_params=expression_params, eyelid_params=eyelid_params)
        return vertices

# -------------------------
# Simple texture PCA reconstructor (Vectorized)
# -------------------------
class FlameTexNumpy:
    def __init__(self, tex_space: dict, image_size=(256,256)):
        if 'tex_dir' in tex_space:
            mu_key, pc_key, scale = 'mean', 'tex_dir', 1.0
        else:
            mu_key, pc_key, scale = 'MU', 'PC', 255.0
            
        self.texture_mean = np.asarray(tex_space[mu_key]).astype(np.float32).reshape(-1) * scale
        self.texture_basis = np.asarray(tex_space[pc_key]).astype(np.float32)
        self.image_size = image_size
        self.texture = None

    def set_actor_texture(self, img_array: np.ndarray):
        self.texture = img_array.astype(np.float32)
        if self.texture.max() <= 1.0:
            self.texture = (self.texture * 255.0).astype(np.float32)

    def forward(self, texcode: np.ndarray) -> np.ndarray:
        if self.texture is not None:
            H, W = self.image_size
            img = self.texture
            h0, w0 = img.shape[:2]
            
            if (h0, w0) == (H, W):
                return (img[None, ...] / 255.0)

            # Vectorized Bilinear Interpolation (Replaces the slow nested Python loops)
            ys = np.clip((np.arange(H) + 0.5) * (h0 / H) - 0.5, 0, h0 - 1)
            xs = np.clip((np.arange(W) + 0.5) * (w0 / W) - 0.5, 0, w0 - 1)
            
            y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
            y1, x1 = np.clip(y0 + 1, 0, h0 - 1), np.clip(x0 + 1, 0, w0 - 1)
            
            wy, wx = ys - y0, xs - x0
            
            c00 = img[y0[:, None], x0[None, :]]
            c01 = img[y0[:, None], x1[None, :]]
            c10 = img[y1[:, None], x0[None, :]]
            c11 = img[y1[:, None], x1[None, :]]
            
            wx_grid = wx[None, :, None]
            wy_grid = wy[:, None, None]
            
            c0 = c00 * (1 - wx_grid) + c01 * wx_grid
            c1 = c10 * (1 - wx_grid) + c11 * wx_grid
            out = c0 * (1 - wy_grid) + c1 * wy_grid
            
            return out[None, ...] / 255.0

        P = self.texture_mean.shape[0]
        N = texcode.shape[0]
        tex = self.texture_mean[None, :] + (self.texture_basis[:, :texcode.shape[1]] @ texcode.T).T
        
        try:
            side = int(np.sqrt(P // 3))
            tex = tex.reshape(N, side, side, 3)
        except Exception:
            tex = tex.reshape(N, 1, P, 1)
            
        tex = tex[..., ::-1]
        return tex / 255.0

def load_flame_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if hasattr(data, '__dict__') and not isinstance(data, dict):
        data = data.__dict__
    return data