import cv2
import torch
import trimesh
import numpy as np
from notebook.utils import setup_sam_3d_body

estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

def get_tpose_mesh_final(estimator, outputs):
    mhr_layer = estimator.model.head_pose.mhr
    device = next(mhr_layer.parameters()).device
    raw_shape = outputs[0]['shape_params']
    id_coeffs = torch.tensor(raw_shape).float().to(device).unsqueeze(0)
    pose_coeffs = torch.zeros((1, 204)).float().to(device)
    face_coeffs = torch.zeros((1, 72)).float().to(device)
    with torch.no_grad():
        res = mhr_layer(
            id_coeffs,      
            pose_coeffs,    
            face_coeffs,    
            True            
        )
    if isinstance(res, tuple):
        if isinstance(res[0], (tuple, list)):
            verts = res[0][0]
            joints = res[0][1]
        else:
            verts = res[0]
            joints = res[1]
    else:
        verts = res
        joints = None
    try:
        if hasattr(mhr_layer, 'character_torch'):
            faces = mhr_layer.character_torch.mesh.faces
        else:
    
            faces = estimator.model.head_pose.mhr.character_torch.mesh.faces
        faces_np = faces.cpu().numpy()
    except:
        faces_np = None
    verts_np = verts.cpu().numpy()[0]
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    if joints is not None:
        joints_np = joints.cpu().numpy()[0]
    else:
        joints_np = None
    return mesh, joints_np

