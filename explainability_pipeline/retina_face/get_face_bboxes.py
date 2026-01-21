from retinaface import RetinaFace
from tqdm import tqdm
import os
import json
from PIL import Image, ImageDraw
import numpy as np
import random

# helper functions
def visualize_retinaface_output(
        image: Image.Image,
        detection: dict
    ) -> Image.Image:
    """
    Helper to visualize RetinaFace outputs; useful for manual checking
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # use the RetinaFace keys to draw bboxes
    for face_key, face_data in detection.items():
        if "facial_area" not in face_data or "landmarks" not in face_data:
            continue
        
        # outline the total face area in red
        x1, y1, x2, y2 = face_data["facial_area"]
        draw.rectangle([x1, y1, x2, y2], outline = "red", width = 3)
        
        # draw dots for the facial landmarks
        for _, (lx, ly) in face_data["landmarks"].items():
            r = 2
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill = "blue", outline = "blue")
    
    return annotated_img


def convert_to_serializable(obj):
    """
    Helper to parse the output of RetinaFace (which can be output as a string in dictionary form)
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

    
def get_retina_face_outputs(
       input_dir: str,
       output_dir: str,
       output_json: str,
       save_images: bool
    ):
    """
    Generates a JSON file containing facial area information for each person 
    in an image using RetinaFace.
    
    :param input_dir: path to directory of images
    :type input_dir: str
    :param output_dir: path to directory to output images and JSON file
    :type output_dir: str
    :param output_json: name of the JSON file to output information to
    :type output_json: str
    :param save_images: indicator to save annotated images
    :type save_images: bool
    """

    # first, see if previous run got interrupted
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}
    try:
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                results_dict = json.load(f)
            print(f"Resuming. Loaded {len(results_dict)} existing entries from {output_json}")
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode {output_json}. Starting from scratch.")
    except FileNotFoundError:
        print("No existing JSON file found. Starting from scratch.")

    all_files_to_process = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    new_files_to_process = []

    # only process files that haven't been processed already
    for filename in all_files_to_process:
        image_key = os.path.splitext(filename)[0]
        if image_key not in results_dict:
            new_files_to_process.append(filename)
    print(f"Found {len(new_files_to_process)} new images to process.")

    # run RetinaFace on all new images and save the outputs
    for filename in tqdm(new_files_to_process, desc = "Processing images"):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)

            try:
                # run RetinaFace
                detection = RetinaFace.detect_faces(image_path)
                results_dict[os.path.splitext(filename)[0]] = detection

                # visualize the outputs if specified
                if save_images:
                    img = Image.open(image_path).convert("RGB")
                    annotated = visualize_retinaface_output(img, detection)
                    annotated.save(os.path.join(output_dir, filename))
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    serializable_results = convert_to_serializable(results_dict)

    with open(output_json, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nDone. Outputs saved to:\n- {output_dir}\n- {output_json}")

