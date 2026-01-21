import argparse
import json
from tqdm import tqdm
from PIL import Image

import pathlib

from utils.description import describe_image_llava
from utils.initialisation import init_llava, init_stable_diffusion

TRAIN_SIZE = 14275
VAL_SIZE = 4614

RESULT_FILE = "results.json"

CATEGORIES = ["activities", "emotions"]


def phase_run_imgs(force: bool, output_dir: pathlib.Path, category: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    if category == "activities":
        prompt = "Focusing only on the activites that the person or people shown are doing, and ignoring other aspects of the image, describe the activities that the person or people shown are doing from one of the following categories of activities: [helping and caring, eating, household, dance and music, personal care, posing, sports, transportation, work, other, unsure]. Keep the description to at most 50-60 words."
    elif category == "emotions":
        prompt = "Focusing only on the expressions and emotions of the person or people shown, and ignoring other aspects of the image, describe the emotions and expressions of the person or people in the image from one of the following emotions: [happiness, sadness, fear, anger, neutral, unsure]. Keep the description to at most 50-60 words."
    else:
        raise ValueError(
            f"unexpected category {category}, expected one of {CATEGORIES}"
        )

    pipe = init_stable_diffusion()
    processor, model = init_llava()

    for split in ("train", "val"):
        folder = pathlib.Path("phase_imgs") / split
        out_folder = output_dir / split
        out_folder.mkdir(parents=True, exist_ok=True)
        num_files = TRAIN_SIZE if split == "train" else VAL_SIZE

        results = {}

        if not force and (out_folder / RESULT_FILE).exists():
            # load existing results
            with (out_folder / RESULT_FILE).open("r") as f:
                results = json.load(f)

        if len(results) == num_files:
            print(f"{split} fully generated, skipping...")

        for file in tqdm(folder.glob("*.png"), desc=f"{split}...", total=num_files):
            image_id = file.stem
            if not force and image_id in results:
                # skip already-generated output
                continue

            image = Image.open(file)

            fst_desc = describe_image_llava(processor, model, image, prompt=prompt)

            new_image: Image.Image = pipe(
                fst_desc,
                num_inference_steps=28,
                width=768,
                height=768,
            ).images[0]  # type: ignore

            img_path = out_folder / file.name
            new_image.save(img_path)

            results[image_id] = {
                "original image": str(file),
                "first description": fst_desc,
                "image loc": str(img_path),
            }

            with (out_folder / RESULT_FILE).open("w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force regeneration of all images"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output_dir",
        type=str,
        help="Defines the output directory",
        dest="output_dir",
    )
    parser.add_argument("-c", "--category", choices=CATEGORIES)
    args = parser.parse_args()

    phase_run_imgs(
        force=args.force,
        output_dir=pathlib.Path(args.output_dir),
        category=args.category,
    )
