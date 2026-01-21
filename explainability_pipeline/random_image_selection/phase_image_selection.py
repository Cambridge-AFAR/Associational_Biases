import random
import os
import shutil
import json
from typing import Literal

def get_phase_images(
    category: str,
    category_type: Literal["activity", "emotion"],
    root_dir: str = "../phase_images/val/",
    out_dir: str = "random_images/",
    out_suffix: str = "",
    annotations_path: str = "../phase_images/val_anno.json",
    num_to_copy: int | str = "max" # int or "max"
):
    """
    Docstring for get_phase_images
    
    :param category: category name, such as "work", "anger", etc.
    :type category: str
    :param category_type: type of category, either an activity or emotion
    :type category_type: Literal["activity", "emotion"]
    :param root_dir: root directory of Phase images
    :type root_dir: str
    :param out_dir: output directory path
    :type out_dir: str
    :param out_suffix: an optional suffix to add onto the output directory
    :type out_suffix: str
    :param annotations_path: path to Phase annotations
    :type annotations_path: str
    :param num_to_copy: a number of images to copy, or the maximum possible
    :type num_to_copy: int | str
    """

    directories = out_dir + f"phase_{category}_images"
    directories += ("/" + out_suffix) if out_suffix != "" else ""
    os.makedirs(directories, exist_ok = True)
    bbox_dict = {}

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    all_images = {
        k: v for k, v in data.items()
        if any(
            person.get("annotations", {}).get(category_type) == category
            for key, person in v.items()
            if key.startswith("person")
        )
    }

    print(f"Total number of images with this {category_type}:", len(all_images))

    if num_to_copy == "max":
        num_to_copy = len(all_images)
    elif isinstance(num_to_copy, str):
        raise ValueError("Please specify `num_to_copy` as an integer or 'max'.")
    
    # sample the given number of images
    sample_keys = random.sample(list(all_images.keys()), min(num_to_copy, len(all_images)))

    # copy the given sample into the specified output directory
    # and create a JSON file containing the relevant bbox annotation data
    for key in sample_keys:
        item = all_images[key]
        image_filename = f"{key}.png"
        image_path = os.path.join(root_dir, image_filename)
        out_image_path = os.path.join(directories, image_filename)

        if os.path.exists(image_path):
            shutil.copy(image_path, out_image_path)
        else:
            print(f"image not found - {image_path}")
            continue

        bbox_dict[key] = []

        for subkey, person in item.items():
            if subkey.startswith("person"):
                if person.get("annotations", {}).get(category_type) == category:
                    # gets the body bounding box coordinates for the person in the image
                    region = person.get("region", [])
                    if len(region) == 4:
                        bbox_dict[key].append(region)

    bbox_path = os.path.join(directories, "bbox_annotations.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_dict, f, indent=2)

    print(f"Copied {len(bbox_dict)} images and saved bbox JSON to {bbox_path}")