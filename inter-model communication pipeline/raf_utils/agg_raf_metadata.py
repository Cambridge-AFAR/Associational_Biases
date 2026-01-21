import json
import os
from itertools import chain
from typing import Dict, Tuple

from utils import RAF_DB_EMOTIONS

GENDERS = ["Male", "Female", "Unsure"]
RACES = ["Caucasian", "African-American", "Asian"]
AGES = ["0-3", "4-19", "20-39", "40-69", "70+"]

if __name__ == "__main__":
    label_lookup: Dict[Tuple[str, int], int] = {}

    data = {}

    with open("RAF_DB/train_labels.csv") as f, open("RAF_DB/test_labels.csv") as f2:
        for line in chain(f.readlines()[1:], f2.readlines()[1:]):
            img, label = line.split(",")
            part, img_id, *_ = img.split("_")
            label_lookup[part, int(img_id)] = int(label)

    base = "RAF_DB_FULL/basic/Annotation/manual"
    for file in os.listdir(base):
        # if "train" not in file:
        #     continue
        part, img_id, *_ = file.split("_")
        label = label_lookup[part, int(img_id)]
        with open(f"{base}/{file}") as f:
            file_data = map(lambda x: int(x.strip()), f.readlines()[-3:])
        if label not in data:
            data[label] = {
                "categories": {
                    "age": {"_metadata": {"invalid": 0}},
                    "ethnicity": {"_metadata": {"invalid": 0}},
                    "gender": {"_metadata": {"invalid": 0}},
                },
                "label": RAF_DB_EMOTIONS[label - 1],
            }
        gen_ind, eth_ind, age_ind = file_data
        age = AGES[age_ind]
        eth = RACES[eth_ind].lower()
        gen = GENDERS[gen_ind].lower()
        data[label]["categories"]["age"][age] = (
            data[label]["categories"]["age"].get(age, 0) + 1
        )
        data[label]["categories"]["age"]["_metadata"]["total"] = (
            data[label]["categories"]["age"]["_metadata"].get("total", 0) + 1
        )
        data[label]["categories"]["ethnicity"][eth] = (
            data[label]["categories"]["ethnicity"].get(eth, 0) + 1
        )
        data[label]["categories"]["ethnicity"]["_metadata"]["total"] = (
            data[label]["categories"]["ethnicity"]["_metadata"].get("total", 0) + 1
        )
        data[label]["categories"]["gender"][gen] = (
            data[label]["categories"]["gender"].get(gen, 0) + 1
        )
        data[label]["categories"]["gender"]["_metadata"]["total"] = (
            data[label]["categories"]["gender"]["_metadata"].get("total", 0) + 1
        )

    with open("raf_utils/raf_agg.json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
