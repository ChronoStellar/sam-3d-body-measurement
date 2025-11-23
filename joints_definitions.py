import numpy as np

MHR_JOINT_MAP = {
    "pelvis": 1, "spine1": 37, "spine2": 36, "spine3": 35,
    "neck_base": 110, "r_shoulder": 69, "r_elbow": 73, "r_wrist": 60
}

MEASUREMENT_CONFIG = {
    # --- CIRCUMFERENCES ---
    "Chest": {
        "type": "circumference",
        "landmarks": ["spine1", "spine2"],
        "func": np.mean,
        "offset": 3.0,
        "color": "blue"
    },
    "Waist": {
        "type": "circumference",
        "landmarks": ["spine2", "spine3"],
        "func": np.mean,
        "offset": 0.0,
        "color": "green"
    },
    "Hips": {
        "type": "circumference",
        "landmarks": ["pelvis"],
        "func": np.mean,
        "offset": 0.0,
        "color": "purple"
    },
    
    # --- NEW: POLYLINE (Curved Measurements) ---
    "Spinal Length": {
        "type": "polyline", 
        # Define the path the tape measure should follow
        "landmarks": ["neck_base", "spine1", "spine2", "spine3", "pelvis"], 
        "color": "orange"
    },
    
    # --- LINEAR (Straight Lines) ---
    "Arm Length": {
        "type": "polyline", # Changed to polyline to include elbow bend
        "landmarks": ["r_shoulder", "r_elbow", "r_wrist"],
        "color": "cyan"
    }
}