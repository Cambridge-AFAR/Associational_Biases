# sources for code: https://huggingface.co/stabilityai/stable-diffusion-3.5-large, https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
import argparse
import json
import os
from typing import cast

from PIL import Image
from tqdm import tqdm

from utils import RAF_DB_EMOTIONS
from utils.data import Data, DataMode, DBImage, ImageFile, PromptTuple
from utils.description import describe_image_llava
from utils.initialisation import init_llava, init_stable_diffusion


def generate_and_describe_loop(
    pipe,
    initial_prompt: str,
    prompt_name: str,
    output_dir: str,
    loops: int = 5,
    processor=None,
    model=None,
    initial_image=None,
):
    folder_path = os.path.join(output_dir, prompt_name)
    os.makedirs(folder_path, exist_ok=True)

    ps = [
        {
            "iteration": 0,
            "prompt": initial_prompt,
        }
    ]

    for i in (
        pb := tqdm(range(loops), desc="Inter-model communication...", leave=False)
    ):
        if i > 0:
            ps.append({"iteration": i, "prompt": ps[i - 1]["description"]})

        num_steps = 28

        if initial_image is None or i > 0:
            image: Image.Image = pipe(
                ps[i]["prompt"], num_inference_steps=num_steps, width=768, height=768
            ).images[0]  # type: ignore
        else:
            image_full_path, image_label = initial_image
            image = Image.open(image_full_path)
            if i == 0:
                del ps[0]["prompt"]
                ps[0]["initial_image"] = prompt_name
                ps[0]["label"] = RAF_DB_EMOTIONS[image_label - 1]

        image_fp = os.path.join(folder_path, f"{prompt_name}_{i}.png")

        image.save(image_fp)
        ps[i]["image"] = image_fp

        pb.write("Describing image...")

        ps[i]["description"] = describe_image_llava(processor, model, image)
        pb.write("Image described")

    with open(os.path.join(folder_path, f"{prompt_name}.json"), "w") as json_file:
        json.dump(ps, json_file, indent=4, sort_keys=True)

    # print(ps)

    # print(pd.json_normalize(ps))


def main(force, prompts_fp, output_dir, raf, skip_model_init, loops, iters, raf_split):
    pipe = None
    processor, model = None, None
    if not skip_model_init:
        pipe = init_stable_diffusion()
        processor, model = init_llava()

    print("Reading prompts YAML...")
    pf = Data(prompts_fp)
    print("...done!")

    if not raf:
        std_prompt_img(force, output_dir, pipe, processor, model, pf, loops, iters)
    else:
        raf_run(force, output_dir, pipe, processor, model, raf_split)


def std_prompt_img(force, output_dir, pipe, processor, model, pf, loops, iters):
    for p in (
        pb := tqdm(
            pf,
            desc="Iterating through prompts...",
            total=pf.size,
            leave=True,
            dynamic_ncols=True,
        )
    ):
        path = ""
        prompt = ""
        initial_image = None
        if pf.mode == DataMode.prompts:
            p = cast(PromptTuple, p)
            path = p.name.lower()
            prompt = p.prompt
        else:
            p = cast(DBImage, p)
            pf._data = cast(ImageFile, pf._data)
            path = p.image_path.split(".")[0].lower()
            initial_image = (os.path.join(pf._data.base_dir, p.image_path), p.label)
        if not force and os.path.exists(os.path.join(output_dir, path)):
            pb.write(f"Skipping {path}...")
            continue

        for it in tqdm(range(iters), desc="Iterating over prompt...", leave=False):
            generate_and_describe_loop(
                pipe,
                prompt,
                output_dir=output_dir,
                prompt_name=path if iters == 1 else f"{path}_{it}",
                loops=loops,
                processor=processor,
                model=model,
                initial_image=initial_image,
            )


def raf_run(force, output_dir, pipe, processor, model, raf_split="test"):
    (test_path, emotion_dirs, _), *rest = os.walk(f"RAF_DB/DATASET/{raf_split}")

    prompt = "Focussing on the expressions and emotions of the person shown, and ignoring other aspects such as the fact that the image is a closeup, describe the emotions and expressions of the person in the image. Keep the description to at most 50-60 words."

    ps = {str(i): {} for i in range(1, 8)}

    json_path = os.path.join(output_dir, "results.json")

    if not force and os.path.exists(json_path):
        print("Loaded existing result json")
        with open(json_path) as f:
            ps = json.load(f)

    for dirpath, _, filenames in (
        pb := tqdm(
            rest,
            desc=f"Iterating through {test_path}...",
            total=len(emotion_dirs),
            leave=True,
            dynamic_ncols=True,
        )
    ):
        _, label_num = os.path.split(dirpath)
        label = RAF_DB_EMOTIONS[int(label_num) - 1]

        output_sub_dir = os.path.join(output_dir, dirpath)
        os.makedirs(output_sub_dir, exist_ok=True)

        if (
            not force
            and ps.get(label_num) is not None
            and len(filenames)
            == (
                len(ps[label_num]["results"])
                if ps[label_num].get("results") is not None
                else -1
            )
        ):
            pb.write("Folder fully generated, skipping...")
            continue

        if ps.get(label_num) == {}:
            ps[label_num] = {"folder": label_num, "label": label, "results": {}}

        for file in (
            pb2 := tqdm(filenames, desc=f"Iterating through {dirpath}...", leave=False)
        ):
            img_path = os.path.join(dirpath, file)
            gen_img_path = os.path.join(output_dir, img_path)

            if not force and os.path.exists(gen_img_path):
                pb2.write("Image already exists, skipping...")
                continue

            image = Image.open(img_path)

            pb2.write("Describing RAF image...")
            fst_description = describe_image_llava(
                processor, model, image, prompt=prompt
            )
            pb2.write("RAF image described")

            pb2.write("Generating image...")
            new_image: Image.Image = pipe(
                fst_description,
                num_inference_steps=28,
                width=768,
                height=768,
            ).images[0]

            pb2.write("Image generated")

            new_image.save(gen_img_path)

            pb2.write("Describing generated image...")
            snd_description = describe_image_llava(
                processor, model, new_image, prompt=prompt
            )
            pb2.write("Generated image described")

            ps[label_num]["results"][file] = {
                "original image": img_path,
                "first description": fst_description,
                "image location": gen_img_path,
                "second description": snd_description,
            }

            with open(json_path, "w") as json_file:
                json.dump(ps, json_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force regeneration of all images"
    )
    parser.add_argument("--raf", action="store_true", help="Run RAF")
    parser.add_argument(
        "-p",
        "--prompts",
        type=str,
        default="prompts/initial_prompts.yaml",
        help="Path to the prompt file to process (optional).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output_dir",
        type=str,
        default="img_results/initial_images",
        help="Defines the output directory",
        dest="output_dir",
    )
    parser.add_argument(
        "--skip_model_init",
        action="store_true",
        help="Skip model initialisation, for test runs only.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=5,
        help="Sets number of generation/description loops.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Sets number of iterations per prompt.",
    )
    parser.add_argument("--raf_split", choices=["test", "train"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(
        force=args.force,
        prompts_fp=args.prompts,
        output_dir=args.output_dir,
        raf=args.raf,
        skip_model_init=args.skip_model_init,
        loops=args.loops,
        iters=args.iters,
        raf_split=args.raf_split,
    )
