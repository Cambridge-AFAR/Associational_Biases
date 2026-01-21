import argparse
import json
import os
import re

from PIL import Image
from tqdm import tqdm

from eval.raf_img_eval import PromptEnum
from utils.description import describe_image_llava
from utils.initialisation import init_llava

helper_dict = {
    "surprised": {"folder": "1", "label": "Surprise", "results": {}},
    "fear": {"folder": "2", "label": "Fear", "results": {}},
    "disgust": {"folder": "3", "label": "Disgust", "results": {}},
    "happy": {"folder": "4", "label": "Happiness", "results": {}},
    "sad": {"folder": "5", "label": "Sadness", "results": {}},
    "angry": {"folder": "6", "label": "Anger", "results": {}},
    "neutral": {"folder": "7", "label": "Neutral", "results": {}},
}


def main(prompt_choice: PromptEnum):
    prompt = prompt_choice.value

    processor, model = init_llava()

    base_path = "img_results/iters"

    regex = re.compile(r"(\[.*\]\s*->)")

    result_d = {}

    for category in (pb := tqdm(helper_dict, desc="iterating through categories...")):
        # print(f"Iterating through {category}...")

        with open(os.path.join(base_path, category, f"{category}.json")) as f:
            init_j = json.load(f)

        hd = helper_dict[category]
        rd = hd["results"]

        for img_dict in tqdm(init_j, desc="Iterating images", leave=False):
            img = Image.open(img_dict["image"])
            resp = describe_image_llava(processor, model, img, prompt)
            # cleanup response
            resp = regex.sub("", resp)
            img_name = img_dict["image"].split("/")[-1]
            rd[img_name] = img_dict
            rd[img_name]["llava categories"] = resp

        result_d[hd["folder"]] = hd

        pb.write(f"Writing JSON at end of {category}...")
        with open("img_results/iters/llava_annotated.json", "w") as f:
            json.dump(result_d, f, indent=4, sort_keys=True)
        pb.write("...saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=PromptEnum.from_string,
        choices=list(PromptEnum),
        help="Prompt format to choose",
        default="raf",
    )

    args = parser.parse_args()

    main(args.prompt)
