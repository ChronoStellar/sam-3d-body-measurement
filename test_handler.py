import base64
import json
from handler import handler

# 1. Read your image and convert to Base64
img_path = "r.png" # Make sure this file exists
with open(img_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 2. Create the mock payload
mock_event = {
    "input": {
        "image": encoded_string,
        "height": 163
    }
}

# 3. Call the handler directly
print("Sending request to handler...")
response = handler(mock_event)

# 4. Print the JSON result
print("\n--- FINAL JSON RESPONSE ---")
print(json.dumps(response, indent=2))