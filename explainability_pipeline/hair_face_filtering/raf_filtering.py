import json
import os
from tqdm import tqdm

def get_filtered_entries(
        facial_annotation_file: str,
        hair_segmentation_file: str,
        output_dir: str,
        category: str
):
    """
    Collects per-image facial annotations for RAF images.
    Each image contains exactly one face, with gender already provided
    by the RetinaFace output. No body matching is required.
    
    :param facial_annotation_file: file with body facial area coordinates
    :type facial_annotation_file: str
    :param hair_segmentation_file: file with paths to hair segmentation .npy files
    :type hair_segmentation_file: str
    :param output_dir: path to directory to output
    :type output_dir: str
    :param category: category name, such as "happiness", "anger", etc.
    :type category: str
    """

    with open(facial_annotation_file) as f:
        face_annotations = json.load(f)

    with open(hair_segmentation_file) as f:
        hair_masks = json.load(f)

    filtered_output = {}

    for img_id, img_data in tqdm(face_annotations.items(), desc = "Processing images"):
        if img_id not in hair_masks:
            continue

        # raf only has one face per image
        for face_id, face_data in img_data.items():
            if not face_id.startswith("face_"):
                continue

            filtered_output[img_id] = {
                "face_1": {
                    "face_coords": face_data["face_coords"],
                    "hair_mask_path": hair_masks[img_id].get(face_id),
                    "gender": face_data.get("gender", "unsure"),
                    "emotion": category
                }
            }

            break  # ensure only one face

    os.makedirs(output_dir, exist_ok = True)
    output_path = os.path.join(output_dir, f"{category}_all_filtered_faces_and_hair.json")

    with open(output_path, "w") as f:
        json.dump(filtered_output, f, indent=2)

    print("Done.")
