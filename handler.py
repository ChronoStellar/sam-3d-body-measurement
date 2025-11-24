import runpod
import base64
import numpy as np
import cv2
import os
from main import load_model, pipe

# --- 1. Warm Start: Load Model Once ---
print("--- Handler: Loading Model ---")
# This loads the model globally so we don't reload it for every request
MODEL = load_model()
print("--- Handler: Model Loaded ---")

def decode_image(image_input):
    """Decodes base64 string or handles local paths for testing."""
    try:
        # Check if it's a local file path (for debugging)
        if isinstance(image_input, str) and os.path.isfile(image_input):
            return cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)

        # Handle Base64
        if ',' in image_input:
            image_input = image_input.split(',')[1]
        
        decoded_string = base64.b64decode(image_input)
        nparr = np.frombuffer(decoded_string, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Image decoding failed (result is None)")
            
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def format_response(raw_measurements, height):
    """
    Maps the keys from your `measure.py` (Title Case) to 
    the requested snake_case JSON format.
    """
    # Keys found in your log -> Keys required in output
    key_map = {
        "Chest": "chest_circumference",
        "Waist": "waist_circumference",
        "Spinal Length": "torso_length",
        "Arm Length": "arms_length",
        "Inside Leg": "inside_leg_length",
        "Hips": "hips_circumference"
    }

    formatted = {}
    
    # 1. Map values
    for raw_key, value in raw_measurements.items():
        # Convert numpy floats to standard python floats for JSON serialization
        val_float = float(value)
        
        if raw_key in key_map:
            formatted[key_map[raw_key]] = round(val_float, 2)
        else:
            # Fallback for unknown keys: "Key Name" -> "key_name"
            clean_key = raw_key.lower().replace(" ", "_")
            formatted[clean_key] = round(val_float, 2)

    # 2. Ensure Height is included
    formatted["height"] = height

    return formatted

def handler(job):
    """
    Entry point for RunPod.
    Expected Input: { "input": { "image": "base64_...", "height": 165 } }
    """
    job_input = job.get('input', {})
    
    # 1. Parse Inputs
    image_data = job_input.get('image')
    height_cm = float(job_input.get('height', 165)) # Default to 165 if missing

    if not image_data:
        return {"error": "No image provided in input"}

    try:
        # 2. Decode Image
        img_rgb = decode_image(image_data)

        # 3. Run Inference (Using the global MODEL)
        raw_results = pipe(MODEL, img_rgb, height_cm)

        # Check for pipeline errors
        if "error" in raw_results:
            return {"error": raw_results["error"]}

        # 4. Format Output to JSON spec
        final_json = format_response(raw_results, height_cm)
        
        return final_json

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Start the worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})