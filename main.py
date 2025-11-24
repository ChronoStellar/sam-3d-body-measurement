import cv2
import torch
import numpy as np
from utils import setup_sam_3d_body
from measure import get_tpose_mesh_final, get_measurements

# Global variable to hold the model in memory
estimator = None

def load_model():
    """Initializes the model if not already loaded."""
    global estimator
    if estimator is None:
        estimator = setup_sam_3d_body()
    return estimator

def pipe(model, img_path_or_array, height_cm):
    """
    Runs the inference pipeline.
    Args:
        model: The loaded estimator.
        img_path_or_array: Path to image string OR numpy array (RGB).
        height_cm: Float target height.
    Returns:
        dict: The measurement results.
    """
    # Handle input (File path vs Numpy Array)
    if isinstance(img_path_or_array, str):
        img_bgr = cv2.imread(img_path_or_array)
        if img_bgr is None:
            raise ValueError(f"Could not load image from {img_path_or_array}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already an array (e.g., decoded from base64 in handler)
        img_rgb = img_path_or_array

    # Run SAM 3D Body
    outputs = model.process_one_image(img_rgb)

    try:
        tpose_mesh, tpose_joints = get_tpose_mesh_final(model, outputs)
    except Exception as e:
        print(f"Error extracting T-Pose: {e}")
        return {"error": str(e)}
    
    # Calculate measurements
    # Note: get_measurements returns a dict like {'Chest': 90.5, 'Waist': 70.0}
    results = get_measurements(tpose_mesh, tpose_joints, target_height_cm=height_cm)
    
    return results

# if __name__ == "__main__":
#     # Local testing only
#     model = load_model()
#     res = pipe(model, "r.png", 163)
#     print(res)