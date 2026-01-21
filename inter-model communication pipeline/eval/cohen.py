import csv
import json
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from statsmodels.stats.inter_rater import cohens_kappa

from utils import RAF_DB_EMOTIONS
from utils.phase import (
    PHASE_AGES,
    PHASE_GENDERS,
    PHASE_SKIN_TONES,
)

# Define RAF categories
RAF_AGES = ["0-3", "4-19", "20-39", "40-69", "70+"]
RAF_RACES = ["caucasian", "african-american", "asian"]
RAF_GENDERS = ["male", "female", "unsure"]

ALPHA = 0.01


def cohen_kappa(pairs, categories, *, print_cont=False):
    """

    Args:
        pairs: List of tuples (orig_val, coll_val)
        categories: List of possible category values

    Returns:
        chi2_stat: The test statistic
        p_value: The p-value
        df: Degrees of freedom
    """
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    contingency = np.zeros((k, k))

    for orig, coll in pairs:
        if orig in cat_to_idx and coll in cat_to_idx:
            i = cat_to_idx[orig]
            j = cat_to_idx[coll]
            contingency[i, j] += 1

    if print_cont:
        print(list(cat_to_idx.keys()))
        print(contingency)

    res = cohens_kappa(contingency, return_results=False)

    return res


def load_phase_pairs(task: str):
    """
    Load paired data for phase task, grouped by activity/emotion.
    Assumes original is phase_anno/train.json (with multiple people per image),
    collected is phase_{task}s/train_annos.json (single per image).
    For pairing, take the first person from original if multiple.
    """
    short_task = f"{task[:3]}s"

    pairs = defaultdict(lambda: defaultdict(list))

    tot = 0
    skipped = 0

    for split in ("val", "train"):
        orig_path = Path(f"phase_anno/{split}_reduced.json")
        coll_path = Path(f"phase_{short_task}/{split}_annos.json")

        with orig_path.open() as f:
            orig_data = json.load(f)

        with coll_path.open() as f:
            coll_data = json.load(f)

        for img_id, img_data in orig_data.items():
            assert img_id in coll_data, f"{img_id} not found in coll_data"
            if img_id in coll_data:
                orig_annos = img_data["annotations"]
                coll_annos = coll_data[img_id]

                label = orig_annos[task].lower()
                if label not in ("disagreement", "unsure"):
                    for bias in ["age", "gender", "skin tone"]:
                        orig_val = orig_annos[bias.replace(" ", "").lower()].lower()
                        coll_val = coll_annos[bias].lower()
                        if (
                            orig_val
                            and coll_val
                            and orig_val not in ("disagreement", "unsure")
                            and coll_val not in ("disagreement", "unsure")
                        ):
                            if orig_val == "children":
                                orig_val = "child"
                            if coll_val == "young adult":
                                coll_val = "young"
                            pairs[label][bias].append((orig_val, coll_val))
                        else:
                            skipped += 1

                        tot += 1
                else:
                    skipped += 1
                    tot += 3  # since 3 biases

    print(f"found {tot - skipped} pairs, skipped {(skipped / tot) * 100:.2f}%")

    return pairs


def phase_kappa(task: str):
    pairs = load_phase_pairs(task)

    bias_vals = {
        "age": sorted(
            [
                x.lower()
                for x in PHASE_AGES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
        "skin tone": sorted(
            [
                x.lower()
                for x in PHASE_SKIN_TONES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
        "gender": sorted(
            [
                x.lower()
                for x in PHASE_GENDERS
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
    }

    for label, bias_pairs in sorted(pairs.items(), key=lambda item: item[0]):
        print(label)
        for bias_type, categories in bias_vals.items():
            print(bias_type)
            if bias_type in bias_pairs and bias_pairs[bias_type]:
                k = cohen_kappa(
                    bias_pairs[bias_type],
                    categories,
                    print_cont=(label == "household"),
                )
                print(f"  {bias_type}: kappa={k:.3f}")
            else:
                print(f"{task}: {label} {bias_type} has no paired data")


def load_raf_pairs():
    # Load original annotations from RAF_DB_FULL

    label_lookup: Dict[Tuple[str, int], int] = {}

    with open("RAF_DB/train_labels.csv") as f, open("RAF_DB/test_labels.csv") as f2:
        for line in chain(f.readlines()[1:], f2.readlines()[1:]):
            img, label = line.split(",")
            part, img_id, *_ = img.split("_")
            label_lookup[part, int(img_id)] = int(label)

    orig_dict = {}
    base = "RAF_DB_FULL/basic/Annotation/manual"
    for file in os.listdir(base):
        part, img_id, *_ = file.split("_")
        img = f"{part}_{img_id}_aligned.jpg"
        label = label_lookup[part, int(img_id)]

        with open(f"{base}/{file}") as f:
            lines = f.readlines()
            gen_ind, eth_ind, age_ind = map(int, lines[-3:])
            orig_dict[img] = {
                "age": RAF_AGES[age_ind],
                "ethnicity": RAF_RACES[eth_ind],
                "gender": RAF_GENDERS[gen_ind],
                "emo_label": label,
            }

    pairs = defaultdict(lambda: defaultdict(list))
    tot = 0
    skipped = 0

    for split in ("train", "test"):
        with open(f"collected_raf/{split}/raf_cleaned.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["img"]
                tot += 1
                if img in orig_dict:
                    for bias in [
                        "age",
                        "ethnicity",
                        "gender",
                    ]:
                        orig_lab = orig_dict[img]["emo_label"]
                        orig_val = orig_dict[img][bias]
                        coll_val = row[bias]
                        if orig_val and coll_val:
                            pairs[orig_lab][bias].append((orig_val, coll_val))
                        else:
                            skipped += 1
                else:
                    skipped += 1

    print(f"found {tot - skipped} pairs, skipped {(skipped / tot) * 100:.2f}%")

    return pairs


def raf_kappa():
    pairs = load_raf_pairs()

    bias_vals = {
        "age": RAF_AGES,
        "ethnicity": RAF_RACES,
        "gender": RAF_GENDERS,
    }

    for emo_id, emo in enumerate(RAF_DB_EMOTIONS, start=1):
        print(emo)
        emo_pairs = pairs[emo_id]
        for bias_type, categories in bias_vals.items():
            print(bias_type)
            if bias_type in emo_pairs and emo_pairs[bias_type]:
                k = cohen_kappa(emo_pairs[bias_type], categories)
                print(f"kappa = {k:.3f}")
            else:
                print(f"  {bias_type}: No paired data")
    print()


if __name__ == "__main__":
    print("RAF")
    raf_kappa()
    print("Phase emotions")
    phase_kappa("emotion")
    print("\n\nPhase activities")
    phase_kappa("activity")
