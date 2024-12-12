from llava.mm_utils import get_model_name_from_path # type: ignore
from llava_utils import * # type: ignore
import warnings
warnings.filterwarnings("ignore")
import time

model_path = "liuhaotian/llava-v1.6-34b"
prompt = "Describe the image exhaustively."
image_file = "./1430lesassieges00robi/page7.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

print("[EVAL] Loading model...")
start = time.time()
model, tokenizer, conv_mode, image_processor = init_model(args)
print(f"Model loaded in {time.time() - start:.2f}s")

print("[EVAL] Building prompt...")
start = time.time()
input_ids = query_builder(args.query, model, tokenizer, conv_mode)
print(f"Prompt built in {time.time() - start:.2f}s")

print("[EVAL] Building images...")
start = time.time()
images_tensor, image_sizes = image_builder(args, model, image_processor)
print(f"Images built in {time.time() - start:.2f}s")

print("[EVAL] Running inference...")
start = time.time()
outputs = predict(model, args, input_ids, images_tensor, image_sizes, tokenizer)
print(f"Inference ran in {time.time() - start:.2f}s")

print("----------------------\nModel output:")
print(outputs)