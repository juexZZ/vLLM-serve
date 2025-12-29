# test multimodal grounding, changed from the qwen3vl cookbook
import requests
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
# from vllm import LLM, SamplingParams

from viz_utils import (
    decode_json_points, 
    plot_bounding_boxes, 
    plot_points, 
    plot_points_json, 
    parse_json
)

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

openai_api_key = "ai4ce"
openai_api_base = "http://gh116:8000/v1"

HTTP_URL = None
# HEADERS = {
#     'Content-Type': 'application/json',
# }


# # Initialize local OpenAI-compatible client
# client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
def get_image(img_url):
    if os.path.exists(img_url):
        image = Image.open(img_url)
    elif img_url.startswith("http://") or img_url.startswith("https://"):
        response = requests.get(img_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        raise ValueError("Invalid image URL")
    return image

def inference_with_openai_api(img_url, prompt, min_pixels=64 * 32 * 32, max_pixels=9800* 32 * 32):
    print("reading image")
    if os.path.exists(img_url):
        with open(img_url, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    elif img_url.startswith("http://") or img_url.startswith("https://"):
        response = requests.get(img_url)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode("utf-8")
    else:
        raise ValueError("Invalid image URL")
    print("creating client")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print("sending request")
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "greedy": False,
        },
    )
    return completion.choices[0].message.content


# ######################################## test examples ############################################ #
QWEN3_COOKBOOK_ROOT = "/scratch/jz4725/Qwen3-VL/cookbooks"



# ################################################################################################# #
#                                                                                                   #
# Example 1: Detecting different objects on a dining table                                          #
#                                                                                                   #
# ################################################################################################# #
# You can specify the categories of the instances you want to locate (negative categories are also supported and will be skipped during generation)
prompt = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
img_url = f"{QWEN3_COOKBOOK_ROOT}/assets/spatial_understanding/dining_table.png"
model_response = inference_with_openai_api(img_url, prompt)
print(model_response)

image = get_image(img_url)

image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, model_response, save_path="./test_visualizations_nonthinking_sugg_param_235B/dining_table_bbox.png")



# ################################################################################################# #
#                                                                                                   #
# Example 5: Pointing out the people inside a football field and output their role and shirt color. #
#                                                                                                   #
# ################################################################################################# #


# You can also set the output format to include additional key information like object attributes, descriptions, etc in point-based grounding.
prompt = '''Locate every person inside the football field with points, report their point coordinates, role(player, referee or unknown) and shirt color in JSON format like this: {"point_2d": [x, y], "label": "person", "role": "player/referee/unknown", "shirt_color": "the person's shirt color"}'''
img_url = f"{QWEN3_COOKBOOK_ROOT}/assets/spatial_understanding/football_field.jpg"
model_response = inference_with_openai_api(img_url, prompt)
print(model_response)

image = get_image(img_url)

image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_points_json(image, model_response, save_path="./test_visualizations_nonthinking_sugg_param_235B/football_field_points.png")