import cv2
import torch
import numpy as np
from utils import setup_sam_3d_body
from measure import get_tpose_mesh_final, get_measurements

def pipe(model, file, height):
    img_bgr = cv2.imread(file)
    outputs = model.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    try:
        # 1. Run the generator
        tpose_mesh, tpose_joints = get_tpose_mesh_final(estimator, outputs)
        print(f"\nSUCCESS: Generated T-Pose Mesh.")
        print(f"Vertices: {len(tpose_mesh.vertices)}")
        print(f"Joints: {len(tpose_joints) if tpose_joints is not None else 0}")

    except Exception as e:
        print(f"Error: {e}")
    
    get_measurements(tpose_mesh, tpose_joints, target_height_cm=height)

if __name__ == "__main__":
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
    pipe(estimator, "/r.png", 163)