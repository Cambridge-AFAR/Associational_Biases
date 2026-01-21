import argparse
import json
import re

from raf_utils import AGES, GENDERS, RACES

AGE_CATEGORIES = {
    "child/teenager",
    "young adult",
    "middle-aged",
    "old",
    "other/unclear",
}

ETHNICITY_CATEGORIES = {
    "white",
    "black",
    "east asian",
    "south asian",
    "other/unclear",
}

GENDER_CATEGORIES = {"male", "female", "other/unclear"}


def clean_categories(raf_labels: bool, in_file: str, out_file: str):
    regex = re.compile(
        r"1.\s+(?:Ethnicity:\s+)?(.+?)\n2.\s+(?:Gender:\s+)?(.+?)\n3.\s+(?:Age:\s+)?(.+?)$",
        re.IGNORECASE,
    )
    with open(in_file) as f:
        dirty = json.load(f)

    clean = {}
    empty_count = 0

    for folder, folder_dict in dirty.items():
        d = {"label": folder_dict["label"], "categories": {}}
        for file, res_dict in folder_dict["results"].items():
            llava_cats = res_dict["llava categories"]
            if llava_cats == "":
                print(f"File {file} string is empty!")
                empty_count += 1
                continue
            res = regex.search(res_dict["llava categories"])
            if res is None:
                print(llava_cats)
                raise Exception(f"Something went wrong with file {file}.\n{res_dict}.")
            d["categories"][file] = {
                "ethnicity": res.group(1).strip().strip("[]()").lower(),
                "gender": res.group(2).strip().strip("[]()").lower(),
                "age": res.group(3).strip().strip("[]()").lower(),
                "raw": llava_cats.strip().strip("[]()").lower(),
            }
        clean[folder] = d

    clean["empty count"] = empty_count

    with open(out_file, "w") as f:
        json.dump(clean, f, indent=4, sort_keys=True)


def aggregate_results(raf_labels: bool, in_file: str, out_file: str):
    with open(in_file) as f:
        data = json.load(f)

    if raf_labels:
        categories = {
            "age": [x.lower() for x in AGES],
            "ethnicity": [x.lower() for x in RACES],
            "gender": [x.lower() for x in GENDERS],
        }
    else:
        categories = {
            "age": AGE_CATEGORIES,
            "ethnicity": ETHNICITY_CATEGORIES,
            "gender": GENDER_CATEGORIES,
        }

    aggregated_results = {}

    for folder, folder_dict in data.items():
        if folder == "empty count":
            continue

        age_counts: dict = {"_metadata": {"invalid": 0}}
        ethnicity_counts: dict = {"_metadata": {"invalid": 0}}
        gender_counts: dict = {"_metadata": {"invalid": 0}}

        for res_dict in folder_dict["categories"].values():
            age = res_dict["age"]
            if age in categories["age"]:
                age_counts[age] = age_counts.get(age, 0) + 1
            else:
                print("Invalid age", res_dict)
                age_counts["_metadata"]["invalid"] += 1

            # Count ethnicity
            ethnicity = res_dict["ethnicity"]
            if ethnicity in categories["ethnicity"]:
                ethnicity_counts[ethnicity] = ethnicity_counts.get(ethnicity, 0) + 1
            else:
                print("Invalid ethnicity", res_dict)
                ethnicity_counts["_metadata"]["invalid"] += 1

            # Count gender
            gender = res_dict["gender"]
            if gender in categories["gender"]:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
            else:
                print("Invalid gender", res_dict)
                gender_counts["_metadata"]["invalid"] += 1

        for d in age_counts, ethnicity_counts, gender_counts:
            d["_metadata"]["total"] = sum(
                filter(lambda x: isinstance(x, int), d.values())
            )

        aggregated_results[folder] = {
            "label": folder_dict["label"],
            "categories": {
                "age": age_counts,
                "ethnicity": ethnicity_counts,
                "gender": gender_counts,
            },
        }

    with open(out_file, "w") as f:
        json.dump(aggregated_results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a function to run.")
    parser.add_argument(
        "function",
        choices=["agg", "clean"],
        help="The function to run: 'agg' for `aggregate_results`, 'clean' for `clean_categories`",
    )

    parser.add_argument("--raf", action="store_true", required=False)

    parser.add_argument("-i", "--infile", type=str, required=True)
    parser.add_argument("-o", "--outfile", type=str, required=True)

    args = parser.parse_args()

    raf_labels = args.raf
    in_file = args.infile
    out_file = args.outfile

    if args.function == "agg":
        aggregate_results(raf_labels, in_file, out_file)
    elif args.function == "clean":
        clean_categories(raf_labels, in_file, out_file)
