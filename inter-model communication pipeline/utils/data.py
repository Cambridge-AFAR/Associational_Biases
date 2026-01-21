from enum import Enum
from typing import List, NamedTuple, Optional, Union, cast

import yaml
from pydantic import BaseModel, model_validator
from typing_extensions import Self


class Prompt(BaseModel):
    name: str
    override: Optional[str] = None
    text: Optional[str] = None

    @model_validator(mode="after")
    def check_override_or_text(self) -> Self:
        assert (
            self.override is None or self.text is None
        ), "Cannot set both override and text fields"
        return self


class PromptFile(BaseModel):
    """Used to validate the format of a prompt YAML file"""

    prompt_template: str
    prompts: List[Prompt]


class PromptTuple(NamedTuple):
    name: str
    prompt: str


class DBImage(BaseModel):
    image_path: str
    label: int


class ImageFile(BaseModel):
    base_dir: str
    images: List[DBImage]


class DataMode(Enum):
    prompts = "prompts"
    images = "images"


class Data:
    def __init__(self, yaml_path) -> None:
        with open(yaml_path) as y:
            data = yaml.safe_load(y)
        if "prompt_template" in data:
            self._data = PromptFile(**data)
            self.template = self._data.prompt_template
            self.items = iter(self._data.prompts)
            self.size = len(self._data.prompts)
            self.mode = DataMode.prompts
        elif "base_dir" in data:
            self._data = ImageFile(**data)
            self.items = iter(self._data.images)
            self.size = len(self._data.images)
            self.base_dir = self._data.base_dir
            self.mode = DataMode.images
        else:
            raise ValueError("Invalid yaml file")

    def __iter__(self):
        return self

    def __next__(self) -> Union[PromptTuple, DBImage]:
        item = next(self.items)
        if self.mode == DataMode.prompts:
            item = cast(Prompt, item)
            if item.override is not None:
                return PromptTuple(item.name, item.override)
            format_string = item.text if item.text is not None else item.name
            return PromptTuple(item.name, self.template.format(format_string))
        elif self.mode == DataMode.images:
            item = cast(DBImage, item)
            return item
        else:
            raise StopIteration  # should never reach here

    def print(self):
        print(self._data)


if __name__ == "__main__":
    p = Data("prompts/raf_db.yaml")
    # p.print()
    for prompt in p:
        print(prompt)
