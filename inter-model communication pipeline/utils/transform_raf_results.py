import json
import os


def main():
    with open("whole_raf/results.json") as f:
        data = json.load(f)

    reformatted = {}

    for d in data:
        if reformatted.get(d["folder"]) is None:
            reformatted[d["folder"]] = {
                "folder": d["folder"],
                "label": d["label"],
                "results": reformat_results(d["results"]),
            }
        else:
            reformatted[d["folder"]]["results"].update(reformat_results(d["results"]))

    with open("whole_raf/results_reformatted.json", "w") as f:
        json.dump(reformatted, f, indent=4, sort_keys=True)


def reformat_results(results):
    new_d = {}
    for r in results:
        _, tail = os.path.split(r["original image"])
        new_d[tail] = r
    return new_d


def validate():
    with open("whole_raf/results_reformatted.json") as f:
        data = json.load(f)

    for i in range(1, 8):
        assert str(i) in data, f"Data does not contain label {i}"
    for k, v in data.items():
        assert len(os.listdir(f"RAF_DB/DATASET/test/{k}")) == len(
            v["results"]
        ), f"Label {k} does not contain the correct amount of data"


if __name__ == "__main__":
    main()
    validate()
