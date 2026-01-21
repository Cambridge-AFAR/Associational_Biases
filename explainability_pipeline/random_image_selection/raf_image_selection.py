import os
import random
import shutil
import pathlib

# RAF emotion mapping from docs
RAF_LABELS = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

def get_raf_images(
    emotion: str,
    root_dir: str,
    subset: str = "train",          # "train", "test", or "all"
    out_suffix: str = "",
    num_to_copy: int | str = "max" # int or "max"
):
    """
    Copies all, or a random subset of, RAF-DB images for the specified emotion.
    
    :param emotion: given emotion per RAF_LABELS
    :type emotion: str
    :param root_dir: root directory of RAF-DB images
    :type root_dir: str
    :param subset: specification for train, test, or all
    :type subset: str
    :param out_suffix: an optional suffix to add onto the output directory
    :type out_suffix: str
    :param num_to_copy: a number of images to copy, or the maximum possible
    :type num_to_copy: int | str
    """
    
    root = pathlib.Path(root_dir)
    img_dir = root / "aligned"  # use aligned to make sure it's accurate to the annotations
    label_file = root / "list_patition_label.txt"
    out_dir = root / f"raf_{emotion.lower()}_images"
    if out_suffix:
        out_dir = out_dir / out_suffix
    out_dir.mkdir(parents = True, exist_ok = True)

    # check to make sure we have a valid emotion input
    reverse_labels = {v.lower(): k for k, v in RAF_LABELS.items()}
    if emotion.lower() not in reverse_labels:
        raise ValueError(f"Invalid emotion: {emotion}. Must be one of {list(reverse_labels.keys())}")

    emotion_id = reverse_labels[emotion.lower()]

    # take the RAF-DB label file (that partitions emotions) to get all relevant images
    selected = []
    with open(label_file, "r") as f:
        for line in f:
            name, label_str = line.strip().split()
            label = int(label_str)
            if label == emotion_id:
                if subset == "train" and not name.startswith("train_"):
                    continue
                if subset == "test" and not name.startswith("test_"):
                    continue
                selected.append(name)

    print(f"Total images for {emotion}: {len(selected)}")

    # sample the given number of images
    if num_to_copy == "max":
        sample = selected
    else:
        n = min(int(num_to_copy), len(selected))
        sample = random.sample(selected, n)

    # copy the given sample into the specified output directory
    copied = 0
    for name in sample:
        src_path = img_dir / name
        dst_path = out_dir / name
        if not src_path.exists():
            print(f"Missing image: {src_path}")
            continue
        shutil.copy(src_path, dst_path)
        copied += 1

    print(f"Copied {copied} images into {out_dir}.")