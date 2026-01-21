import json
import re

from pathlib import Path

from PIL import Image

from utils.description import describe_image_llava
from utils.initialisation import init_llava

from tqdm import tqdm


prompt = """Answer the following questions about the main person shown in the image. Fill in the gaps based on the options provided. Do not describe the image or give extra details, just provide the best option from the list.

1. Skin tone: [Choose from: lighter, darker] -> 
2. Gender: [Choose from: male, female, unsure] -> 
3. Age: [Choose from: baby, child, young adult, adult, senior, unsure] -> 
"""


def main():
    processor, model = init_llava()

    print("model loaded!")

    regex = re.compile(r"(\[.*\]\s*->)")

    for focus in ("acts", "emos"):
        for split in ("train", "val"):
            folder_path = Path(f"phase_{focus}/{split}")
            json_path = folder_path / "results.json"
            output_path = json_path.with_stem("annotations")
            print(f"loading JSON for {str(folder_path)}...")
            with json_path.open() as f:
                j = json.load(f)
            print("JSON loaded...")

            if output_path.exists():
                with output_path.open() as f:
                    result_dict = json.load(f)
            else:
                result_dict = {}

            total = len(j)

            pbar = tqdm(enumerate(j.items(), start=1), total=total, leave=False)

            for it, (img_id, img_dict) in pbar:
                if img_id in result_dict:
                    continue

                img = Image.open(img_dict["image loc"])
                resp = describe_image_llava(processor, model, img, prompt)
                # cleanup response
                resp = regex.sub("", resp)
                result_dict[img_id] = resp

                if it % 25 == 0:
                    # pbar.(f"Progress: {it} / {total} ({(it / total) * 100:.1f}%)")
                    pbar.write(f"Response for {img_id}:\n{resp}")
                    pbar.write("Saving JSON checkpoint...")
                    with open(output_path, "w") as f:
                        json.dump(result_dict, f)
                    pbar.write("...saved!")

            print(f"Writing JSON at end of {str(folder_path)}...")
            with open(output_path, "w") as f:
                json.dump(result_dict, f)
            print("...saved!")


if __name__ == "__main__":
    main()
