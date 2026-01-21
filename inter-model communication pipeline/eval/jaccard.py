import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

from raf_utils import AGES, GENDERS, RACES
from utils import RAF_DB_EMOTIONS
from utils.phase import (
    PHASE_AGES,
    PHASE_GENDERS,
    PHASE_SKIN_TONES,
)

# source: https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance


def raf_jacc():
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

    results = []

    for emotion_ind in orig:
        emotion = RAF_DB_EMOTIONS[int(emotion_ind) - 1]
        emotion_results = []
        for bias_type in orig[emotion_ind]["categories"]:
            orig_bias_dict = orig[emotion_ind]["categories"][bias_type]
            coll_bias_dict = coll[emotion_ind]["categories"][bias_type]
            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias_type]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias_type]])

            vals = np.array(vals)
            js = np.sum(vals.min(axis=0)) / np.sum(vals.max(axis=0))
            emotion_results.append(js)

        results.append([emotion.title()] + emotion_results)

    print("\nRAF Dataset Results:")
    print(
        tabulate(
            results,
            headers=["Emotion", "Age", "Ethnicity", "Gender"],
            tablefmt="grid",
            floatfmt=".4f",
        )
    )
    print()


def phase_chi(task: str):
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
        "skin tone": sorted(
            [
                x.lower()
                for x in PHASE_SKIN_TONES
                if x.lower() not in ("disagreement", "unsure")
            ]
        ),
    }

    results = []

    for category in orig:
        if category[0] == "_":
            continue
        cat_res = []
        for bias_type in orig[category]:
            orig_bias_dict = orig[category][bias_type]
            coll_bias_dict = coll[category][bias_type]

            vals = []
            vals.append([orig_bias_dict.get(b, 0) for b in bias_vals[bias_type]])
            vals.append([coll_bias_dict.get(b, 0) for b in bias_vals[bias_type]])

            vals = np.array(vals)
            js = np.sum(vals.min(axis=0)) / np.sum(vals.max(axis=0))
            cat_res.append(js)
        results.append([category.title()] + cat_res)

    print(f"\nPhase {task.title()}:")
    print(
        tabulate(
            results,
            headers=["Category", "Age", "Gender", "Skin Tone"],
            tablefmt="grid",
            floatfmt=".4f",
        )
    )
    print()


if __name__ == "__main__":
    raf_jacc()
    phase_chi("activity")
    phase_chi("emotion")
