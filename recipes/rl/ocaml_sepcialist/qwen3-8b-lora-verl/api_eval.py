# You can use this model with any of the OpenAI clients in any language!
# Simply change the API Key to get started

import datasets
from openai import OpenAI
import os
from reward_function import extract_solution, check_ocaml_compilation
import random

test_dataset = datasets.load_dataset("nuprl/MultiPL-E", "mbpp-ml")
test_dataset = test_dataset["test"]
idx = random.randint(0, len(test_dataset) - 1)
example = test_dataset[idx]
print("==== Prompt ====")
print(example["prompt"])

with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

def evaluate_response(content):
    solution = extract_solution(content)
    if not solution:
        print("Failed to extract solution: ", content)
    else:
        print(f"Solution: {solution}")
        print(f"Compiles: {check_ocaml_compilation(solution)}")

API_KEY = os.getenv("BASETEN_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://model-vq07jj13.api.baseten.co/environments/production/sync/v1"
)


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": example["prompt"]}
]

# Base Model
response = client.chat.completions.create(
    model="baseten-model",
    messages=messages,
    temperature=0.0,
)

print("==== Base Model ====")
print("\n")
evaluate_response(response.choices[0].message.content)

# LoRA Model
response = client.chat.completions.create(
    model="global_step_22/actor/lora_adapter", # name of the checkpoint
    messages=messages,
    temperature=0.0,
)

print("\n")
print("==== Finetuned LoRA ====")

evaluate_response(response.choices[0].message.content)

