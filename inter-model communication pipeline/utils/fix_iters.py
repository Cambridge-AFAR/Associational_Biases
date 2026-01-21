import json
import os
from itertools import groupby
import shutil

base = "img_results/iters"

iters = sorted(os.listdir(base))
groups = groupby(iters, key=lambda x: x.split("_")[0])

for key, group in groups:
    os.makedirs(f"{base}/{key}", exist_ok=True)
    res = []
    for i, f in enumerate(group):
        with open(f"{base}/{f}/{f}.json") as jsonfile:
            j = json.load(jsonfile)[0]
        new_image = f"{base}/{key}/{key}_{i}.png"
        res.append(
            {
                "iteration": i,
                "description": j["description"],
                "image": new_image,
                "prompt": j["prompt"],
            }
        )
        os.rename(j["image"], new_image)
        shutil.rmtree(f"{base}/{f}")

    with open(f"{base}/{key}/{key}.json", "w") as jsonfile:
        json.dump(res, jsonfile, indent=4, sort_keys=True)
