import argparse
import torch
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re
import time
from llava.constants import ( # type: ignore
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle # type: ignore
from llava.model.builder import load_pretrained_model # type: ignore
from llava.utils import disable_torch_init # type: ignore
from llava.mm_utils import ( # type: ignore
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def init_model(args):
    """
    Initialize the model, tokenizer and image processor

    :param args: Arguments; structure with the following fields:
        - model_path: str
        - model_base: str
        - model_name: str
        - query: str
        - image_file: str
        - sep: str
        - temperature: float
        - top_p: float
        - num_beams: int
        - max_new_tokens: int
    :return: the model, tokenizer and image processor
    """
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    print(f"Model loaded: {model_name}; Conversation mode: {conv_mode}.")
    return model, tokenizer, conv_mode, image_processor

def query_builder(prompt, model, tokenizer, conv_mode):
    """
    Build the query

    :param prompt: str
    :param model: the model
    :param tokenizer: the tokenizer
    :return: the input ids for the model
    """
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in prompt:
        if model.config.mm_use_im_start_end:
            prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
        else:
            prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
    else:
        if model.config.mm_use_im_start_end:
            prompt = image_token_se + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids

def image_builder(args, model, image_processor):
    """
    Build the images

    :param args: Arguments; structure with the following fields:
        - model_path: str
        - model_base: str
        - model_name: str
        - query: str
        - image_file: str   <- this is a list of image files, required
        - sep: str
        - temperature: float
        - top_p: float
        - num_beams: int
        - max_new_tokens: int
    :param model: the model
    :param image_processor: the image processor
    :return: the images tensor and their sizes
    """
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    return images_tensor, image_sizes

def predict(model, args, input_ids, images_tensor, image_sizes, tokenizer):
    """
    Make the prediction

    :param model: the model
    :param args: Arguments; structure with the following fields:
        - temperature: float
        - top_p: float
        - num_beams: int
        - max_new_tokens: int
        - (optional) other fields
    :param input_ids: the input ids
    :param images_tensor: the images tensor
    :param image_sizes: the image sizes
    :param tokenizer: the tokenizer
    :return: the outputs, as a string
    """
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def eval_model(args):
    """
    Evaluate the model

    :param args: Arguments; structure with the following fields:
        - model_path: str
        - model_base: str
        - model_name: str
        - query: str
        - conv_mode: str
        - image_file: str
        - sep: str
        - temperature: float
        - top_p: float
        - num_beams: int
        - max_new_tokens: int
    """
    print("[EVAL] Loading model...")
    model, tokenizer, conv_mode, image_processor = init_model(args)
    print("[EVAL] Building query...")
    input_ids = query_builder(args.query, model, tokenizer, conv_mode)
    print("[EVAL] Building images...")
    images_tensor, image_sizes = image_builder(args, model, image_processor)
    print("[EVAL] Running inference...")
    outputs = predict(model, args, input_ids, images_tensor, image_sizes, tokenizer)
    print("----------------------\nModel output:")
    print(outputs)

def run_inference(model, args, input_ids, images_tensor, image_sizes, tokenizer):
    """
    Run inference

    :param model: the model
    :param args: Arguments; structure with the following fields:
        - temperature: float
        - top_p: float
        - num_beams: int
        - max_new_tokens: int
        - (optional) other fields
    :param input_ids: the input ids
    :param images_tensor: the images tensor
    :param image_sizes: the image sizes
    :param tokenizer: the tokenizer
    :return: the outputs, as a string
    """
    print("[EVAL] Running inference...")
    outputs = predict(model, args, input_ids, images_tensor, image_sizes, tokenizer)
    print("----------------------\nModel output:")
    print(outputs)

if __name__ == "__main__":
    """
    Example usage
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer, image_processor = init_model(args)
    input_ids = query_builder(args.query, model, tokenizer)
    images_tensor, image_sizes = image_builder(args, model, image_processor)
    outputs = predict(model, args, input_ids, images_tensor, image_sizes, tokenizer)
