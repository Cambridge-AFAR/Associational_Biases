import json
from pathlib import Path
import pandas as pd


def conv_raf_gen_results(file: Path):
    with file.open("r") as f:
        d = json.load(f)

    def get_rows():
        for folder in d:
            for img, img_vals in d[folder]["results"].items():
                yield (
                    folder,
                    d[folder]["label"],
                    img,
                    img_vals["original image"],
                    img_vals["image location"],
                    img_vals["first description"],
                    img_vals["second description"],
                )

    df = pd.DataFrame(
        get_rows(),
        columns=[
            "folder",
            "label",
            "img",
            "orig_img_loc",
            "gen_img_loc",
            "fst_desc",
            "snd_desc",
        ],
    )

    df.to_csv(file.with_suffix(".csv"), index=False)


def conv_raf_cleaned_results(file: Path):
    with file.open("r") as f:
        d = json.load(f)

    def get_rows():
        for folder in d:
            if folder == "empty count":
                continue
            for img, img_vals in d[folder]["categories"].items():
                yield (
                    folder,
                    img,
                    img_vals["age"],
                    img_vals["ethnicity"],
                    img_vals["gender"],
                    img_vals["raw"].replace("\n", "\\n"),
                )

    df = pd.DataFrame(
        get_rows(),
        columns=[
            "folder",
            "img",
            "age",
            "ethnicity",
            "gender",
            "raw",
        ],
    )

    # df = df.stack().str.replace("\n", "\\n", regex=True).unstack()
    df.to_csv(file.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    path = "collected_raf/test/results.json"
    conv_raf_gen_results(Path(path))
    path = "collected_raf/test/raf_cleaned.json"
    conv_raf_cleaned_results(Path(path))
