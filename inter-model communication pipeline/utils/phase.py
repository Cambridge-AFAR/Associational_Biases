import json
import re
from collections import defaultdict
from functools import reduce
from operator import mul
from pathlib import Path

PHASE_AGES = [
    "baby",  # (0-2 years-old)
    "child",  # (3-14 years-old)
    "young",  # (15-29 years-old) aka young adult
    "adult",  # (30-64 years-old)
    "senior",  # (65 years-old or more)
    "unsure",
]

PHASE_GENDERS = ["male", "female", "unsure"]

PHASE_SKIN_TONES = [
    "lighter",
    "darker",
    "unsure",
]  # based on Fitzpatrick, but simplified to binary for higher annotator agreement

PHASE_ETHNICITIES = [
    "black",
    "east asian",
    "indian",
    "latino",
    "middle eastern",
    "southeast asian",
    "white",
    "other",
    "unsure",
]

PHASE_EMOTIONS = ["happiness", "sadness", "fear", "anger", "neutral", "unsure"]

PHASE_ACTIVITIES = [
    "helping and caring",
    "eating",
    "household",
    "dance and music",
    "personal care",
    "posing",
    "sports",
    "transportation",
    "work",
    "other",
    "unsure",
]


def shorten_label(label: str):
    if label == "helping and caring":
        return "caring"
    if label == "dance and music":
        return "music"
    if label == "personal care":
        return "personal"
    if label == "happiness":
        return "happy"
    if label == "sadness":
        return "sad"
    return label


def clean_phase_annos(file: Path):
    outf = (
        Path("phase_anno") / Path("train" if "train" in file.name else "val")
    ).with_suffix(".json")
    with file.open("r") as f:
        data: dict[str, dict] = json.load(f)

    exp = re.compile(r"person(\d+)")

    for img_data in data.values():
        people = {}
        for key in list(img_data.keys()):
            res = exp.match(key)
            if res:
                ind = int(res.group(1))
                person = img_data.pop(key)
                person["annotations"]["ethnicity"] = person["annotations"][
                    "ethnicity"
                ].replace("_", " ")
                people[ind] = person
        img_data["people"] = people

    json.dump(data, outf.open("w"))


def phase_annos_by_biggest_region():
    base = Path("phase_anno")
    for split in ("train", "val"):
        new_data = {}
        with (base / split).with_suffix(".json").open() as f:
            data = json.load(f)
        for img, img_dict in data.items():
            people = img_dict.pop("people")
            bp = max(people.values(), key=lambda p: reduce(mul, p["region"]))
            new_data[img] = img_dict | bp
        with (base / f"{split}_reduced").with_suffix(".json").open("w") as f:
            json.dump(new_data, f)


def agg_collected_phase():
    regex = re.compile(
        r"1.\s+(?:Skin tone:\s+)?(.+?)\n2.\s+(?:Gender:\s+)?(.+?)\n3.\s+(?:Age:\s+)?(.+?)$",
        re.IGNORECASE,
    )
    phase_base = Path("phase_anno")
    for task in ("acts", "emos"):
        results = defaultdict(
            lambda: {
                "age": defaultdict(int),
                "skin tone": defaultdict(int),
                "gender": defaultdict(int),
            }
        )
        results["_invalid"] = {"age": 0, "skin tone": 0, "gender": 0}  # type: ignore
        collected_base = Path(f"phase_{task}")
        for split in ("train", "val"):
            img_results = {}
            with (phase_base / f"{split}_reduced").with_suffix(".json").open() as f:
                orig = json.load(f)
            with (collected_base / split / "annotations.json").open() as f:
                annos = json.load(f)
            for img_id, a in annos.items():
                res = regex.search(a)
                if res is None:
                    for k in results["_invalid"]:
                        results["_invalid"][k] += 1
                    continue

                st = res.group(1).strip().strip("[]()").lower()
                g = res.group(2).strip().strip("[]()").lower()
                a = res.group(3).strip().strip("[]()").lower()

                img_results[img_id] = {
                    "age": a,
                    "gender": g,
                    "skin tone": st,
                }

                task_key = "activity" if task == "acts" else "emotion"
                task_val = orig[img_id]["annotations"][task_key]
                if task_val == "disagreement":
                    task_val = "_disagreement"
                if st in PHASE_SKIN_TONES:
                    results[task_val]["skin tone"][st] += 1
                else:
                    results["_invalid"]["skin tone"] += 1
                if g in PHASE_GENDERS:
                    results[task_val]["gender"][g] += 1
                else:
                    results["_invalid"]["gender"] += 1
                if a == "young adult":
                    a = "young"
                if a in PHASE_AGES:
                    results[task_val]["age"][a] += 1
                else:
                    results["_invalid"]["age"] += 1
            with (collected_base / f"{split}_annos.json").open("w") as f:
                json.dump(img_results, f, sort_keys=True, indent=4)

        with (collected_base / "agg.json").open("w") as f:
            json.dump(results, f, sort_keys=True, indent=4)


def agg_orig_phase():
    phase_base = Path("phase_anno")
    tasks = ("activity", "emotion")

    results = {
        task: defaultdict(
            lambda: {
                "age": defaultdict(int),
                "skin tone": defaultdict(int),
                "gender": defaultdict(int),
            }
        )
        for task in tasks
    }

    for split in ("train", "val"):
        with (phase_base / f"{split}_reduced").with_suffix(".json").open() as f:
            orig = json.load(f)

        for img_dict in orig.values():
            st = img_dict["annotations"]["skintone"]
            g = img_dict["annotations"]["gender"]
            a = img_dict["annotations"]["age"]

            for task in tasks:
                task_val = img_dict["annotations"][task]
                if task_val == "disagreement":
                    task_val = "_disagreement"
                results[task][task_val]["skin tone"][st] += 1
                results[task][task_val]["gender"][g] += 1
                results[task][task_val]["age"][a] += 1
    for task in tasks:
        with (phase_base / f"{task}_agg.json").open("w") as f:
            json.dump(results[task], f, sort_keys=True, indent=4)


if __name__ == "__main__":
    # clean_phase_annos(Path("phase_anno/phase_gcc_train_regions_20221101.json"))
    # clean_phase_annos(Path("phase_anno/phase_gcc_val_regions_20221101.json"))
    # phase_annos_by_biggest_region()
    agg_collected_phase()
    # agg_orig_phase()
