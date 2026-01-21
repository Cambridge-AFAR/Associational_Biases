from collections import defaultdict
from pathlib import Path
import pandas as pd
import json

COCO_CSV_COLS = [
    "id",
    "skin",  # one per person in image
    "ratio",  # bounding box ration of largest person
    "bb_skin",  # largest person's skin
    "bb_gender",  # largest person's gender (can be both though)
    "split",  # val or train
]

COCO_SKIN_TYPES = ["Light", "Dark"]

COCO_GENDERS = ["Male", "Female", "Both"]


def load_annos():
    csv = Path("coco_annos/images_val2014.csv")
    df = pd.read_csv(csv.open("r"), index_col="id")
    return df


def extract_caps():
    base = Path("coco_caps")
    cs = base / "captions_val2014.json"
    with cs.open("r") as f:
        data: dict = json.load(f)
    annos = data["annotations"]
    del data
    data = defaultdict(list)
    for cap in annos:
        data[cap["image_id"]].append(cap["caption"].strip())
    with (base / "val_caps_clean.json").open("w") as f:
        json.dump(data, f)


def prune_unneeded_imgs():
    annos = load_annos()
    base = Path("coco_ims")
    dest = base / "spare"
    dest.mkdir(exist_ok=True, parents=True)
    ids = set(annos.index)
    for img in (base / "val").glob("*.jpg"):
        img_id = int(img.stem.split("_")[-1])
        if img_id not in ids:
            img.rename(dest / img.name)


def validate_imgs():
    annos = load_annos()
    base = Path("coco_ims/val")
    num_imgs = len(list(base.glob("*.jpg")))
    num_annos = len(annos)
    assert num_imgs == num_annos, f"got {num_imgs} and {num_annos}"
    print(f"{num_imgs} images and annotations in {base}")


if __name__ == "__main__":
    # extract_caps()
    # prune_unneeded_imgs()
    validate_imgs()
