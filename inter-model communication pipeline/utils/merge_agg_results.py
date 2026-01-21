import json
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Callable


class KeyedDefaultDict(defaultdict):
    default_factory: Callable[[Any], Any]

    def __init__(self, default_factory: Callable[[Any], Any]):
        super().__init__()
        self.default_factory = default_factory

    def __missing__(self, key: Any) -> Any:
        self[key] = self.default_factory(key)
        return self[key]


def merge_json_files(*json_fps: str):
    json_objects = []
    with ExitStack() as stack:
        for fp in json_fps:
            file = stack.enter_context(open(fp, "r"))
            json_objects.append(json.load(file))

    merged = {}

    for obj in json_objects:
        for key, data in obj.items():
            if key not in merged:
                merged[key] = {
                    "categories": defaultdict(
                        lambda: KeyedDefaultDict(
                            lambda x: defaultdict(int) if x == "_metadata" else int()
                        )
                    ),
                    "label": data["label"],
                }

            for category, subcategories in data["categories"].items():
                for subkey, value in subcategories.items():
                    if subkey == "_metadata":
                        # update _metadata values later after summing counts
                        continue

                    merged[key]["categories"][category][subkey] += value

                # handle _metadata separately
                metadata = subcategories.get("_metadata", {})
                merged[key]["categories"][category]["_metadata"] = {
                    "invalid": merged[key]["categories"][category]["_metadata"].get(
                        "invalid", 0
                    )
                    + metadata.get("invalid", 0),
                    "total": sum(
                        v
                        for sk, v in merged[key]["categories"][category].items()
                        if sk != "_metadata"
                    ),
                }

    return merged


if __name__ == "__main__":
    merged_json = merge_json_files(
        "collected_raf/train/raf_agg.json", "collected_raf/test/raf_agg.json"
    )

    with open("collected_raf/whole_raf_agg.json", "w") as f:
        json.dump(merged_json, f, indent=4, sort_keys=True)
