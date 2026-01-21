# source: https://huggingface.co/sentence-transformers/clip-ViT-L-14
import argparse
import json
import os
import sys

import dotenv
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

load_any = dotenv.load_dotenv()

if load_any:
    print("Environment variables loaded from .env!")
else:
    print("Nothing loaded from .env")

image_and_sent_model = SentenceTransformer("clip-ViT-L-14")
sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def truncate_text(text, max_length=77):
#     return ' '.join(text.split()[:max_length])

run_env = os.environ.get("RUN_ENV")
local_run = run_env is None or run_env == "local"


def convert_hpc_fp_to_local(fp: str):
    if not local_run:
        return fp
    return fp.split("affective-ai/")[1]


def compare_image_to_desc(input_dir, output_dir):
    print("Comparing images to descriptions...")
    results = []
    for dir in (
        pb := tqdm(os.listdir(input_dir), desc="Iterating through directory...")
    ):
        pb.desc = f"Iterating through directory {dir}..."
        results.append({"prompt_name": dir})
        with open(os.path.join(input_dir, dir, f"{dir}.json")) as fp:
            j = json.load(fp)

        label = j[0].get("label")
        if label is not None:
            results[-1]["label"] = label

        for iter in tqdm(j, desc="Iterating...", leave=False):
            img_emb = image_and_sent_model.encode(
                Image.open(convert_hpc_fp_to_local(iter["image"]))  # type: ignore
            )
            text_emb = image_and_sent_model.encode([iter["description"]])
            results[-1][f"iter_{iter['iteration']}"] = util.cos_sim(
                img_emb, text_emb
            ).item()

    df = pd.json_normalize(results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "image_to_desc.csv"), index=False)


def compare_pairwise(input_dir, output_dir):
    print("Comparing pairwise...")
    img_to_img_results = []
    text_to_text_results = []
    for dir in (
        pb := tqdm(os.listdir(input_dir), desc="Iterating through directory...")
    ):
        pb.desc = f"Iterating through directory {dir}..."
        img_to_img_results.append({"prompt_name": dir})
        text_to_text_results.append({"prompt_name": dir})
        with open(os.path.join(input_dir, dir, f"{dir}.json")) as fp:
            j = json.load(fp)

        label = j[0].get("label")
        if label is not None:
            img_to_img_results[-1]["label"] = label
            text_to_text_results[-1]["label"] = label

        for fst, snd in tqdm(
            zip(j, j[1:]), desc="Iterating...", leave=False, total=len(j) - 1
        ):
            fst_img_emb = image_and_sent_model.encode(
                Image.open(convert_hpc_fp_to_local(fst["image"]))  # type: ignore
            )
            snd_img_emb = image_and_sent_model.encode(
                Image.open(convert_hpc_fp_to_local(snd["image"]))  # type: ignore
            )
            text_emb = sent_model.encode([fst["description"], snd["description"]])

            img_to_img_results[-1][f"iter_{fst['iteration']}_{snd['iteration']}"] = (
                util.cos_sim(fst_img_emb, snd_img_emb).item()
            )
            text_to_text_results[-1][f"iter_{fst['iteration']}_{snd['iteration']}"] = (
                util.cos_sim(text_emb[0], text_emb[1]).item()
            )

    os.makedirs(output_dir, exist_ok=True)

    df_ii = pd.json_normalize(img_to_img_results)
    df_ii.to_csv(os.path.join(output_dir, "image_to_image_pairwise.csv"), index=False)
    df_tt = pd.json_normalize(text_to_text_results)
    df_tt.to_csv(os.path.join(output_dir, "desc_to_desc_pairwise.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--type", choices=["image_to_desc", "pairwise", "both"], default="both"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing the input data (images and JSON files).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory to save the output CSV files.",
    )
    args = parser.parse_args()

    choice = args.type
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if choice in ["image_to_desc", "both"]:
        compare_image_to_desc(input_dir, output_dir)
    if choice in ["pairwise", "both"]:
        compare_pairwise(input_dir, output_dir)
