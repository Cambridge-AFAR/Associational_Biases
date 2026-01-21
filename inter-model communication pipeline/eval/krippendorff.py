import json
from collections import defaultdict
from pathlib import Path

import krippendorff
import numpy as np

from raf_utils.agg_raf_metadata import AGES, GENDERS, RACES

MANUAL_BASE = Path("manual_annotations")

cats = ["age", "ethnicity", "gender"]

cat_to_lab_to_idx = {
    n: defaultdict(
        lambda: np.nan, {item.lower(): index for index, item in enumerate(v)}
    )
    for n, v in zip(cats, [AGES, RACES, GENDERS])
}

cat_to_lab_to_idx["gender"]["unclear"] = np.nan


def get_reliability_data_from_json(json_fp: str):
    print(f"...{json_fp}")
    with open(json_fp) as f:
        j = json.load(f)
    res = {c: [] for c in cats}
    for em, em_dict in j.items():
        if em == "empty count":
            continue
        for img_dict in em_dict["categories"].values():
            for cat in cats:
                cat_idx = cat_to_lab_to_idx[cat][img_dict[cat]]
                res[cat].append(cat_idx)

    return res


def get_all_raf_data():
    def _get_data():
        for split in ("test", "train"):
            with (Path("collected_raf") / split / "raf_cleaned.json").open() as f:
                data = json.load(f)
            for cat in data:
                if cat == "empty count":
                    continue
                yield data[cat]["categories"]

    for d in _get_data():
        for key, value in d.items():
            del value["raw"]
            yield key.split(".")[0], value


def raf_kripp(*, transform_age: bool = False):
    with (MANUAL_BASE / "raf.json").open() as f:
        manual_annos: dict[str, dict] = json.load(f)

    def _clean_dict(d):
        for key in d:
            d[key] = d[key].lower()
        if "ethnicity" in d:
            d["race"] = d.pop("ethnicity")
        if transform_age:
            if "19" in d["age"] or "20" in d["age"]:
                d["age"] = "4-39"
        return d

    for img in manual_annos:
        manual_annos[img] = _clean_dict(manual_annos[img])

    llava_annos = {
        key: _clean_dict(value)
        for key, value in get_all_raf_data()
        if key in manual_annos
    }

    keys = sorted(manual_annos.keys())

    cats = ("age", "gender", "race")

    annos = (manual_annos, llava_annos)

    def get_labels_for_cat(cat):
        cat_lists = []
        for lab in annos:
            cat_lists.append([lab[key][cat] for key in keys])
        return cat_lists

    for cat in cats:
        d = np.array(get_labels_for_cat(cat))
        alpha = krippendorff.alpha(d, level_of_measurement="nominal")
        print(f"{cat.title()} α: {alpha}")


def phase_kripp(task):
    short_task = f"{task[:3]}s"

    def _load(split):
        with Path(f"phase_{short_task}/{split}_annos.json").open() as f:
            return json.load(f)

    with (MANUAL_BASE / f"phase_{short_task}.json").open() as f:
        manual_annos: dict[str, dict] = json.load(f)

    llava_annos = {split: _load(split) for split in ("train", "val")}

    # filter full annos
    def _clean_dict(d):
        for key in d:
            if key == "age":
                if d[key] == "young adult":
                    d[key] = "young"
            d[key] = d[key].lower()
        return d

    llava_annos = {
        img: _clean_dict(llava_annos[manual_annos[img]["set"]][img])
        for img in manual_annos
    }

    assert len(manual_annos) == len(llava_annos)

    cats = ("age", "gender", "skin tone")

    annos = (manual_annos, llava_annos)

    def get_labels_for_cat(cat):
        keys = sorted(manual_annos.keys())
        cat_lists = []
        for lab in annos:
            cat_lists.append([lab[key][cat] for key in keys])
        return cat_lists

    for cat in cats:
        d = np.array(get_labels_for_cat(cat))
        alpha = krippendorff.alpha(d, level_of_measurement="nominal")
        print(f"{cat.title()} α: {alpha}")


if __name__ == "__main__":
    print("PHASE_ACTS")
    phase_kripp("activities")
    print("\n")
    print("PHASE_EMOS")
    phase_kripp("emotions")
    print("\n")
    print("RAF")
    raf_kripp(transform_age=True)
