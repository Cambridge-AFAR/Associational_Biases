import os
import json
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

# helper functions
def expand_bbox(
        bbox,
        shape,
        scale = 1.6
    ):
    """
    Helper to expand a given bounding box. This helps to make sure all hair is contained within
    the bbox so that it can properly be detected.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_x1 = max(0, cx - new_w // 2)
    new_y1 = max(0, cy - new_h // 2)
    new_x2 = min(shape[1], cx + new_w // 2)
    new_y2 = min(shape[0], cy + new_h // 2)
    return new_x1, new_y1, new_x2, new_y2


def get_hair_seg_outputs_raf(
        input_dir: str,
        output_dir: str,
        retinaface_json: str,
        output_json: str,
        hair_model: str = "hair_segmenter.tflite",
        visualize: bool = False
    ):
    """
    Detects hair for each specified face in an image and saves it.
    
    :param input_dir: path to directory of images
    :type input_dir: str
    :param output_dir: path to directory to output .npy files and JSON file
    :type output_dir: str
    :param retinaface_json: path to the JSON file outputted by get_retina_face_outputs()
    :type retinaface_json: str
    :param output_json: name of the JSON file to output information to
    :type output_json: str
    :param hair_model: path to the .tflite file to be used by MediaPipe
    :type hair_model: str
    :param visualize: indicates visualization for masks
    :type visualize: bool
    """
    
    os.makedirs(output_dir, exist_ok = True)

    # load RetinaFace output
    with open(retinaface_json, "r") as f:
        face_data = json.load(f)

    # initialize MediaPipe for hair segmentation
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options = BaseOptions(model_asset_path = hair_model),
        running_mode = VisionRunningMode.IMAGE,
        output_category_mask = True
    )

    output_map = {}

    with ImageSegmenter.create_from_options(options) as segmenter:
        # RAF has one face per image
        for image_key, data in tqdm(face_data.items(), desc = "Segmenting hair"):
            img_path = input_dir + "/" + data["image_path"]  # will have a prefix ("train", etc), not just image name
            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            output_map[image_key] = {}

            for face_key, face_data in data.items():
                if not face_key.startswith("face_"):
                    continue
                
                x1, y1, x2, y2 = expand_bbox(face_data["face_coords"], (h, w), scale = 2.0)
                face_crop = image_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # get segmentation for hair
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = face_crop.astype(np.uint8))
                result = segmenter.segment(mp_image)
                mask = result.category_mask.numpy_view()
                binary_mask = (mask > 0).astype(np.uint8)

                # make sure the mask is accurate to the original image
                resized_mask = cv2.resize(binary_mask, (x2 - x1, y2 - y1), interpolation = cv2.INTER_NEAREST)
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = resized_mask

                if visualize:
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                    plt.title("Full Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(binary_mask, cmap="gray")
                    plt.title("Hair Mask (Crop)")
                    plt.axis("off")

                    overlay = image_rgb.copy()
                    overlay[y1:y2, x1:x2][resized_mask > 0] = [255, 0, 0] 
                    plt.subplot(1, 3, 3)
                    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    plt.title("Overlay on Image")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()

                save_path = os.path.join(output_dir, f"{image_key}_{face_key}.npy")
                np.save(save_path, full_mask)

                output_map[image_key][face_key] = {
                    "hair_mask_path": os.path.relpath(save_path, output_dir)
                }

    with open(output_json, "w") as f:
        json.dump(output_map, f, indent=2)

    print(f"\nHair masks saved to: {output_dir}")



def get_hair_seg_outputs_phase(
        input_dir: str,
        output_dir: str,
        retinaface_json: str,
        output_json: str,
        hair_model: str = "hair_segmenter.tflite"
    ):
    """
    Detects hair for each specified face in an image and saves it.
    
    :param input_dir: path to directory of images
    :type input_dir: str
    :param output_dir: path to directory to output .npy files and JSON file
    :type output_dir: str
    :param retinaface_json: path to the JSON file outputted by get_retina_face_outputs()
    :type retinaface_json: str
    :param output_json: name of the JSON file to output information to
    :type output_json: str
    :param hair_model: path to the .tflite file to be used by MediaPipe
    :type hair_model: str
    """
    
    os.makedirs(output_dir, exist_ok = True)

    # load RetinaFace output
    with open(retinaface_json, "r") as f:
        face_data = json.load(f)

    # initialize MediaPipe for hair segmentation
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options = BaseOptions(model_asset_path = hair_model),
        running_mode = VisionRunningMode.IMAGE,
        output_category_mask = True
    )

    output_map = {}

    with ImageSegmenter.create_from_options(options) as segmenter:
        for image_name in tqdm(face_data, desc = "Segmenting hair"):
            image_path = os.path.join(input_dir, f"{image_name}.png")
            if not os.path.exists(image_path):
                continue

            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            face_dict = face_data[image_name]
            output_map[image_name] = {}

            # Phase images may have multiple faces per image
            for face_key, face_data in face_dict.items():
                x1, y1, x2, y2 = expand_bbox(face_data["face_coords"], (h, w), scale = 2.3)

                face_crop = image_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # get segmentation for hair
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop.astype(np.uint8))
                result = segmenter.segment(mp_image)
                mask = result.category_mask.numpy_view()
                binary_mask = (mask > 0).astype(np.uint8)

                # make sure the mask is accurate to the original image
                resized_mask = cv2.resize(binary_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = resized_mask

                save_path = os.path.join(output_dir, f"{image_name}_{face_key}.npy")
                np.save(save_path, full_mask)

                output_map[image_name][face_key] = os.path.relpath(save_path, output_dir)


    with open(output_json, "w") as f:
        json.dump(output_map, f, indent=2)

    print(f"\nHair masks saved to: {output_dir}")