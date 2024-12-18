import sys
from llava_utils import *
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import time

# SETUP ---------------------------------------------------------------------------------------------------------------------------------------------
# Images
if len(sys.argv) < 2:
    print("Usage: python full_pipeline.py <image_folder>")
    sys.exit(1)
images_folder = sys.argv[1]

images_files = []
for bookname in os.listdir(images_folder):
    if not os.path.isdir(os.path.join(images_folder, bookname)): continue
    # take only the png files. This is a simple way to filter out non-image files
    for page in os.listdir(os.path.join(images_folder, bookname)):
        if not page.endswith(".png"): continue
        page_number = int(page.split(".")[0][4:])
        images_files.append({
            "id": f"{bookname}_{page_number}",
            "bookname": bookname,
            "page": page_number,
            "image_path": os.path.join(images_folder, bookname, page)
        })

# Create a dataframe from the list of dictionaries and set the index
out_df = pd.DataFrame(images_files).set_index(["id"])

# Model
model_path = "liuhaotian/llava-v1.6-vicuna-13b"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

print("[EVAL] Loading model...")
model, tokenizer, conv_mode, image_processor = init_model(args)

# Prompts
zero_shot_prompt = "USER: Here is an illustration from a children's book. Your task is to analyze the image and describe the actions depicted. Focus on the following:\n\
1. Identify the setting of the scene (e.g., indoors, outdoors, specific objects in the environment).\n\
2. List the main entities (e.g., humans, animals, or objects).\n\
3. Describe the specific actions performed by each entity.\n\
4. Highlight any interactions or relationships between the entities.\n\
5. Summarize the emotional or narrative tone conveyed by the illustration.\n\
Your response should be clear and detailed, focusing only on observable actions and avoiding speculation. ASSISTANT:"

chain_of_thought_prompt = "USER: Analyze the provided illustration step by step. Follow the instructions below:\n\
1. First, describe the setting of the scene. Focus on details like whether it is indoors or outdoors and mention specific objects in the background.\n\
2. Next, identify the main entities in the illustration. These can include humans, animals, or objects.\n\
3. For each entity, describe the specific actions they are performing. Be as detailed as possible, focusing on observable movements or postures.\n\
4. Highlight any interactions or relationships between the entities. If there are no interactions, explicitly state that.\n\
5. Finally, summarize the emotional or narrative tone conveyed by the scene, considering the actions and interactions you described.\n\n\
Provide your response in the following structure:\n\
Setting: [Describe the setting].\n\
Entities: [List of entities].\n\
Actions: [Actions of each entity].\n\
Interactions: [Description of interactions].\n\
Emotional Tone: [Mood or narrative tone of the scene]. ASSISTANT:"

zero_shot_input_ids = query_builder(zero_shot_prompt, model, tokenizer, conv_mode)
chain_of_thought_input_ids = query_builder(chain_of_thought_prompt, model, tokenizer, conv_mode)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
for id, image_path in out_df["image_path"].items():
    start = time.time()
    args.image_file = image_path
    images_tensor, image_sizes = image_builder(args, model, image_processor)

    outputs = predict(model, args, zero_shot_input_ids, images_tensor, image_sizes, tokenizer)
    out_df.loc[id, "zero_shot"] = outputs

    outputs = predict(model, args, chain_of_thought_input_ids, images_tensor, image_sizes, tokenizer)
    out_df.loc[id, "chain_of_thought"] = outputs

    # Verbosity: print % images processed
    print(f"Processed {len(out_df)}/{len(images_files)} images")
    print(f"Time taken: {time.time() - start} seconds")

out_df.to_csv("out.csv")