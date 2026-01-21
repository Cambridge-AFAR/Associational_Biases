import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from statsmodels.stats.contingency_tables import SquareTable
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.stats.multitest import fdrcorrection

from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.phase import PHASE_AGES, PHASE_GENDERS, PHASE_SKIN_TONES

# Define categories
RAF_AGES = ["0-3", "4-19", "20-39", "40-69", "70+"]
RAF_RACES = ["caucasian", "african-american", "asian"]
RAF_GENDERS = ["male", "female", "unsure"]
ALPHA = 0.01


def cohen_kappa(pairs, categories):
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    contingency = np.zeros((k, k))
    for orig, coll in pairs:
        if orig in cat_to_idx and coll in cat_to_idx:
            i = cat_to_idx[orig]
            j = cat_to_idx[coll]
            contingency[i, j] += 1
    res = cohens_kappa(contingency, return_results=False)
    return res


def stuart_maxwell_test(pairs, categories):
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    contingency = np.zeros((k, k))
    for orig, coll in pairs:
        if orig in cat_to_idx and coll in cat_to_idx:
            i = cat_to_idx[orig]
            j = cat_to_idx[coll]
            contingency[i, j] += 1
    tab = SquareTable(contingency)
    res = tab.homogeneity(method="stuart_maxwell")
    stat, p, df = res.statistic, res.pvalue, res.df  # type: ignore
    if stat == np.inf:
        raise Exception("divide by zero error!")
    return stat, p, df


def load_raf_pairs():
    import csv
    import os
    from itertools import chain

    label_lookup: Dict[tuple[str, int], int] = {}
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
    for split in ("train", "test"):
        with open(f"collected_raf/{split}/raf_cleaned.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["img"]
                if img in orig_dict:
                    orig_lab = orig_dict[img]["emo_label"]
                    for bias in ["age", "ethnicity", "gender"]:
                        orig_val = orig_dict[img][bias]
                        coll_val = row[bias]
                        if orig_val and coll_val:
                            pairs[orig_lab][bias].append((orig_val, coll_val))
    return pairs


def load_phase_pairs(task: str):
    short_task = f"{task[:3]}s"
    pairs = defaultdict(lambda: defaultdict(list))
    for split in ("val", "train"):
        orig_path = Path(f"phase_anno/{split}_reduced.json")
        coll_path = Path(f"phase_{short_task}/{split}_annos.json")
        with orig_path.open() as f:
            orig_data = json.load(f)
        with coll_path.open() as f:
            coll_data = json.load(f)
        for img_id, img_data in orig_data.items():
            if img_id in coll_data:
                orig_annos = img_data["annotations"]
                coll_annos = coll_data[img_id]
                label = orig_annos[task].lower()
                if label not in ("disagreement", "unsure"):
                    for bias in ["age", "gender", "ethnicity"]:
                        bias_key = "skin tone" if bias == "ethnicity" else bias
                        orig_val = orig_annos[bias_key.replace(" ", "").lower()].lower()
                        coll_val = coll_annos[bias_key].lower()
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
    return pairs


def compute_raf_jaccard():
    orig_path = Path("raf_utils/raf_agg.json")
    coll_path = Path("collected_raf/whole_raf_agg.json")
    with orig_path.open() as f:
        orig = json.load(f)
    with coll_path.open() as f:
        coll = json.load(f)
    bias_vals = {
        "age": sorted(list(map(lambda x: x.lower(), AGES))),
        "ethnicity": sorted(list(map(lambda x: x.lower(), RACES))),
        "gender": sorted(list(map(lambda x: x.lower(), GENDERS))),
    }
    results = defaultdict(dict)
    for emotion_ind in orig:
        emotion = RAF_DB_EMOTIONS[int(emotion_ind) - 1]
        for bias in bias_vals:
            orig_bias_dict = orig[emotion_ind]["categories"][bias]
            coll_bias_dict = coll[emotion_ind]["categories"][bias]
            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias]])
            vals = np.array(vals)
            js = (
                np.sum(vals.min(axis=0)) / np.sum(vals.max(axis=0))
                if np.sum(vals.max(axis=0)) > 0
                else 0
            )
            results[emotion][bias] = js
    return results


def compute_phase_jaccard(task: str):
    short_task = f"{task[:3]}s"
    orig_path = Path(f"phase_anno/{task}_agg.json")
    coll_path = Path(f"phase_{short_task}/agg.json")
    with orig_path.open() as f:
        orig = json.load(f)
    with coll_path.open() as f:
        coll = json.load(f)
    bias_vals = {
        "age": sorted(
            [
                x.lower()
                for x in PHASE_AGES
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
        "ethnicity": sorted(
            [
                x.lower()
                for x in PHASE_SKIN_TONES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
    }
    results = defaultdict(dict)
    for category in orig:
        if category[0] == "_":
            continue
        for bias in bias_vals:
            bias_key = "skin tone" if bias == "ethnicity" else bias
            orig_bias_dict = orig[category][bias_key]
            coll_bias_dict = coll[category][bias_key]
            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias]])
            vals = np.array(vals)
            js = (
                np.sum(vals.min(axis=0)) / np.sum(vals.max(axis=0))
                if np.sum(vals.max(axis=0)) > 0
                else 0
            )
            results[category][bias] = js
    return results


# Collect data
raf_pairs = load_raf_pairs()
raf_jacc = compute_raf_jaccard()

raf_biases = ["age", "ethnicity", "gender"]
raf_categories = {
    "age": RAF_AGES,
    "ethnicity": RAF_RACES,
    "gender": RAF_GENDERS,
}

raf_data = {}
for emo_id, emo in enumerate(RAF_DB_EMOTIONS, start=1):
    raf_data[emo] = {}
    for bias in raf_biases:
        if emo_id in raf_pairs and raf_pairs[emo_id][bias]:
            kappa = cohen_kappa(raf_pairs[emo_id][bias], raf_categories[bias])
            stat, p, df = stuart_maxwell_test(
                raf_pairs[emo_id][bias], raf_categories[bias]
            )
            jacc = raf_jacc.get(emo, {}).get(bias, 0)
            raf_data[emo][bias] = (stat, p, kappa, jacc, df)

# FDR correction for RAF
all_pvals = []
for emo in raf_data:
    for bias in raf_biases:
        if bias in raf_data[emo]:
            all_pvals.append(raf_data[emo][bias][1])
rejected, _ = fdrcorrection(all_pvals, alpha=ALPHA)
idx = 0
for emo in sorted(raf_data.keys()):
    for bias in raf_biases:
        if bias in raf_data[emo]:
            stat, p, kappa, jacc, df = raf_data[emo][bias]
            sig = rejected[idx]
            raf_data[emo][bias] = (stat, p, kappa, jacc, df, sig)
            idx += 1

phase_data = {}
for task in ["emotion", "activity"]:
    pairs = load_phase_pairs(task)
    jacc = compute_phase_jaccard(task)
    bias_vals = {
        "age": sorted(
            [
                x.lower()
                for x in PHASE_AGES
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
        "ethnicity": sorted(
            [
                x.lower()
                for x in PHASE_SKIN_TONES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
    }
    phase_data[task] = {}
    for label in pairs:
        phase_data[task][label] = {}
        for bias in ["age", "gender", "ethnicity"]:
            if bias in pairs[label]:
                kappa = cohen_kappa(pairs[label][bias], bias_vals[bias])
                stat, p, df = stuart_maxwell_test(pairs[label][bias], bias_vals[bias])
                jac = jacc.get(label, {}).get(bias, 0)
                phase_data[task][label][bias] = (stat, p, kappa, jac, df)

    # FDR correction for phase task
    all_pvals = []
    for label in phase_data[task]:
        for bias in ["age", "gender", "ethnicity"]:
            if bias in phase_data[task][label]:
                all_pvals.append(phase_data[task][label][bias][1])
    rejected, _ = fdrcorrection(all_pvals, alpha=ALPHA)
    idx = 0
    for label in sorted(phase_data[task].keys()):
        for bias in ["age", "gender", "ethnicity"]:
            if bias in phase_data[task][label]:
                stat, p, kappa, jacc, df = phase_data[task][label][bias]
                sig = rejected[idx]
                phase_data[task][label][bias] = (stat, p, kappa, jacc, df, sig)
                idx += 1

# Output LaTeX table content
print("#" * 10)
for emo in raf_data.keys():
    line = f"{emo.title()}"
    for bias in raf_biases:
        if bias in raf_data[emo]:
            stat, p, kappa, jacc, df, sig = raf_data[emo][bias]
            stumax_str = f"\\bfseries {stat:.2f}" if sig else f"{stat:.2f}"
            line += f" & {stumax_str} & {kappa:.4f} & {jacc:.4f}"
        else:
            line += " & - & - & -"
    print(line + " \\\\")
print("#" * 10)
for task in ["emotion", "activity"]:
    for label in sorted(phase_data[task].keys()):
        line = f"{label.title()}"
        for bias in ["age", "ethnicity", "gender"]:
            if bias in phase_data[task][label]:
                stat, p, kappa, jacc, df, sig = phase_data[task][label][bias]
                stumax_str = f"\\bfseries {stat:.2f}" if sig else f"{stat:.2f}"
                line += f" & {stumax_str} & {kappa:.4f} & {jacc:.4f}"
            else:
                line += " & - & - & -"
        print(line + " \\\\")
    print("#" * 10)
