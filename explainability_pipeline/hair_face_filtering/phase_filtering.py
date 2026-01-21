import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# helper functions
def box_to_rect(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxA_area + boxB_area - inter_area
    return inter_area / union if union > 0 else 0

def horizontal_center_distance(face_rect, body_rect):
    """
    Helper that computes the absolute horizontal distance between the center of a face box
    and the center of a body box. 

    Useful when multiple body boxes fully contain a face.
    """
    face_center_x = (face_rect[0] + face_rect[2]) / 2
    body_center_x = (body_rect[0] + body_rect[2]) / 2
    return abs(face_center_x - body_center_x)

def find_category_and_gender_for_body(body_box, original_persons, category_type, iou_thresh = 0.95):
    """
    Helper to get person-level annotations for each body.
    """
    for person_key, person_data in original_persons.items():
        if not person_key.startswith("person"):
            continue
        region = person_data.get("region", [])
        if len(region) == 4:
            person_box = box_to_rect(region)
            iou = compute_iou(box_to_rect(body_box), person_box)
            if iou >= iou_thresh:
                category = person_data.get("annotations", {}).get(category_type, None)
                gender = person_data.get("annotations", {}).get("gender", None)
                return category, gender
    return None


def get_filtered_entries(
        body_annotation_file: str,
        facial_annotation_file: str,
        hair_segmentation_file: str,
        phase_annotation_file: str,
        input_dir: str,
        output_dir: str,
        category: str,
        category_type: str,
        denote_multiple_faces: bool = False
):
    """
    Matches person-level annotations to facial annotations for relevant category.
    This is important because in Phase images, not all people in a given image are categorized the same,
    but the facial outputs are for every person in an image. Thus, we must connect the correct
    facial annotations to only the relevant people in an image.
    
    :param body_annotation_file: file with body bounding box coordinates
    :type body_annotation_file: str
    :param facial_annotation_file: file with body facial area coordinates
    :type facial_annotation_file: str
    :param hair_segmentation_file: file with paths to hair segmentation .npy files
    :type hair_segmentation_file: str
    :param phase_annotation_file: original Phase annotations
    :type phase_annotation_file: str
    :param input_dir: path to root directory with impages
    :type input_dir: str
    :param output_dir: path to directory to output
    :type output_dir: str
    :param category: category name, such as "work", "anger", etc.
    :type category: str
    :param category_type: type of category, either "activity" or "emotion"
    :type category_type: str
    :param denote_multiple_faces: indicates whether to save images with multiple faces separately
    :type denote_multiple_faces: bool
    """

    with open(body_annotation_file) as f:
        body_annotations = json.load(f)

    with open(facial_annotation_file) as f:
        face_annotations = json.load(f)

    with open(hair_segmentation_file) as f:
        hair_masks = json.load(f)

    with open(phase_annotation_file) as f:
        original_annotations = json.load(f)

    filtered_output = {}

    visual_dir = output_dir + "/all_bbox_vis"
    os.makedirs(visual_dir, exist_ok = True)

    if denote_multiple_faces:
        visual_dir_review = output_dir + "/annotated_multiple_faces"
        os.makedirs(visual_dir_review, exist_ok = True)

    for img_id, body_boxes in tqdm(body_annotations.items(), desc = "Processing images"):
        if img_id not in face_annotations:
            continue
        elif img_id not in hair_masks:
            continue

        body_rects = [box_to_rect(b) for b in body_boxes]
        filtered_faces = {}
        image_path = os.path.join(input_dir, f"{img_id}.png")

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for rect in body_rects:
            draw.rectangle(rect, outline="blue", width=2)

        font = ImageFont.load_default() 

        for face_idx, (face_id, face_data) in enumerate(face_annotations[img_id].items(), start=1):
            face_rect = face_data["facial_area"]

            valid_candidates = [
                (idx, body_rect) for idx, body_rect in enumerate(body_rects)
                if (
                    face_rect[1] >= body_rect[1] and face_rect[3] <= body_rect[3] and  # vertical
                    face_rect[0] >= body_rect[0] and face_rect[2] <= body_rect[2]      # horizontal
                )
            ]

            if valid_candidates:
                best_body_idx, best_body_rect = min(
                    valid_candidates,
                    key = lambda item: horizontal_center_distance(face_rect, item[1])
                )

                # align it s.t. the body and face don't overlap
                adjusted_body_rect = (
                    best_body_rect[0],
                    face_rect[3], 
                    best_body_rect[2], 
                    best_body_rect[3]
                )

                gender = None
                found_category = None
                if img_id in original_annotations:
                    found_category, gender = find_category_and_gender_for_body(body_boxes[best_body_idx], original_annotations[img_id], category_type=category_type)
                
                if found_category == category:
                    filtered_faces[face_id] = {
                        "face_coords": face_rect,
                        "hair_mask_path": hair_masks[img_id].get(face_id, None),
                        "body_coords": adjusted_body_rect,
                        "gender": gender,
                        category_type: found_category
                    }

                draw.rectangle(face_rect, outline="red", width=2)
                label = f"face_{face_idx}"
                x, y = face_rect[0], face_rect[1]
                draw.text((x + 3, y + 3), label, fill="white", font=font)
        
        if filtered_faces:
            filtered_output[img_id] = filtered_faces

            for face_idx, (face_id, data) in enumerate(filtered_faces.items(), start=1):
                face_rect = data["face_coords"]
                gender = data.get("gender", "unknown")
                label = f"face_{face_idx} ({gender})"
                x, y = face_rect[0], face_rect[1]
                draw.text((x + 3, y + 3), label, fill="white", font=font)

            image.save(os.path.join(visual_dir, f"{img_id}_annotated.png"))

            if len(filtered_faces) > 1 and denote_multiple_faces:
                image.save(os.path.join(visual_dir_review, f"{img_id}_annotated.png"))
    
    if denote_multiple_faces:
        multi_face_output = {img_id: faces for img_id, faces in filtered_output.items() if len(faces) > 1}

        multi_face_json_path = output_dir + "/multiple_faces_filtered_output.json"
        with open(multi_face_json_path, "w") as f:
            json.dump(multi_face_output, f, indent = 2)

    with open(os.path.join(output_dir, f"{category}_all_filtered_faces_and_hair.json"), "w") as f:
        json.dump(filtered_output, f, indent=2)
    
    print("Done.")