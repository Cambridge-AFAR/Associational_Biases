import os
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch
from diffusers import StableDiffusion3Pipeline  # type: ignore,  # type: ignore
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("could not get hf token")

HF_HOME = os.environ.get("HF_HOME")
if HF_HOME is None or "rds" not in HF_HOME:
    raise ValueError(f"got wrong HF_HOME: {HF_HOME}")

TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE")


def init_stable_diffusion() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    pipe = pipe.to("cuda")

    return pipe


def init_llava(*, to_gpu=True):
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        cache_dir=TRANSFORMERS_CACHE,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        cache_dir=TRANSFORMERS_CACHE,
    )

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id  # type: ignore

    processor.patch_size = model.config.vision_config.patch_size  # type: ignore

    processor.vision_feature_select_strategy = (  # type: ignore
        model.config.vision_feature_select_strategy
    )

    if to_gpu:
        model.to("cuda")  # type: ignore

    return (processor, model)
