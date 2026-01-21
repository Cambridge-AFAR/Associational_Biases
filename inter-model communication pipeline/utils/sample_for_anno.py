import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sample_phase(num_per_class: int):
    assert num_per_class % 2 == 0
    num_per_class = num_per_class // 2  # half from val, half from train

    base = Path("phase_anno")
    with (base / "val_reduced.json").open() as f:
        val = json.load(f)
    with (base / "train_reduced.json").open() as f:
        train = json.load(f)

    tasks = ("activity", "emotion")

    def get_groups(data):
        res = {t: defaultdict(list) for t in tasks}
        for item in data:
            for task in tasks:
                anno = data[item]["annotations"][task]
                if anno.lower() in ("other", "disagreement"):
                    continue
                res[task][anno].append(item)
        return res

    groups = {"val": get_groups(val), "train": get_groups(train)}

    def sample(task):
        for gs in groups:
            for anno in groups[gs][task]:
                opts = groups[gs][task][anno]
                for img_id in np.random.choice(
                    opts, min(num_per_class, len(opts)), replace=False
                ):
                    yield img_id, f"phase_{task[:3]}s/{gs}/{img_id}.png", anno

    for task in tasks:
        df = pd.DataFrame(sample(task), columns=("img_id", "source", "label"))
        df.to_csv(f"phase_for_anno/{task}_imgs.csv", header=True, index=False)


def sample_raf(num_per_class: int):
    assert num_per_class % 2 == 0
    num_per_class = num_per_class // 2

    def sample():
        base = Path("collected_raf")
        for split in ("test", "train"):
            for folder in (base / split).iterdir():
                if not folder.is_dir():
                    continue
                for choice in np.random.choice(
                    list(folder.glob("*.jpg")),  # type: ignore
                    num_per_class,
                    replace=False,
                ):
                    yield choice.stem, str(choice), folder.stem.split("_")[1]

    df = pd.DataFrame(sample(), columns=("img_id", "source", "label"))

    df.to_csv("collected_raf/for_manual_anno.csv", index=False, header=True)


if __name__ == "__main__":
    NUM_PER_CLASS = 20
    sample_phase(NUM_PER_CLASS)
    sample_raf(NUM_PER_CLASS)
