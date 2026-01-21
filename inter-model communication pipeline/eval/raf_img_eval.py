import argparse
import json
import os
import re
from enum import Enum

from PIL import Image

from utils.description import describe_image_llava
from utils.initialisation import init_llava


class PromptEnum(Enum):
    custom = """
Answer the following questions about the person shown in the image. Fill in the gaps based on the options provided. Do not describe the image or give extra details, just provide the best option from the list.

1. Ethnicity: [Please choose from: white, Black, east Asian, south Asian, other/unclear] -> 
2. Gender: [Please choose from: male, female, other/unclear] -> 
3. Age: [Please choose from: child/teenager, young adult, middle-aged, old, other/unclear] -> 
"""
    raf = """
Answer the following questions about the person shown in the image. Fill in the gaps based on the options provided. Do not describe the image or give extra details, just provide the best option from the list.

1. Ethnicity: [Choose from: Caucasian, African-American, Asian] -> 
2. Gender: [Choose from: male, female, unsure] -> 
3. Age: [Choose from: 0-3, 4-19, 20-39, 40-69, 70+] -> 
"""

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PromptEnum[s]
        except KeyError:
            raise ValueError()


def main(json_path: str, output_path: str, prompt_choice: PromptEnum):
    prompt = prompt_choice.value

    processor, model = init_llava()
    print("Loading input JSON...")
    with open(json_path) as f:
        j = json.load(f)
    print("JSON loaded...")

    regex = re.compile(r"(\[.*\]\s*->)")

    for category in j:
        print(f"Iterating through {category} = {j[category]['label']}...")

        total = len(j[category]["results"])

        for it, (res, result_dict) in enumerate(
            j[category]["results"].items(), start=1
        ):
            img = Image.open(result_dict["image location"])
            resp = describe_image_llava(processor, model, img, prompt)
            # cleanup response
            resp = regex.sub("", resp)
            result_dict["llava categories"] = resp

            if it % 25 == 0:
                print(f"Progress: {it} / {total} ({(it / total) * 100:.1f}%)")
                print(f"Response for {res}:\n{resp}")
                print("Saving JSON checkpoint...")
                with open(output_path, "w") as f:
                    json.dump(j, f, indent=4, sort_keys=True)
                print("...saved!")

        print(f"Writing JSON at end of {j[category]['label']}...")
        with open(output_path, "w") as f:
            json.dump(j, f, indent=4, sort_keys=True)
        print("...saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", type=str, help="Location of input json")
    parser.add_argument("-o", "--output", type=str, help="Path to output")
    parser.add_argument(
        "-p",
        "--prompt",
        type=PromptEnum.from_string,
        choices=list(PromptEnum),
        help="Prompt format to choose",
    )

    args = parser.parse_args()

    head, _ = os.path.split(args.output)

    if head != "":
        os.makedirs(head, exist_ok=True)

    main(args.json, args.output, args.prompt)
