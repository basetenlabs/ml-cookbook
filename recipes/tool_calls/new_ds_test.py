import ast
import re
import json
import jsonschema
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, load_dataset, DatasetDict
from vllm import LLM, SamplingParams

ds = load_dataset("Salesforce/xlam-function-calling-60k")

# 4. Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # or Qwen2.5-3B, Qwen2.5-1.5B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(x, add_generation_prompt=True):
    try:
        pattern = r'<tools>(.*?)</tools>'
        tools = json.loads(x['tools'])
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": x['query']}, 
            ], 
            tools=tools, 
            add_generation_prompt=add_generation_prompt, 
            tokenize=False
        )
        return {
            "text": text,
            # "tools": tools,
            "gt_generation": x['answers'],
            }
    except Exception as e:
        print(e)
        # Return None or a sentinel value to filter out later
        return None

LIMIT = 500
formatted_dataset = ds.map(format_example).filter(lambda x: x is not None)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)
prompts = [sample['text'] for sample in formatted_dataset['train']][:LIMIT]
generations = llm.generate(prompts, sampling_params)

all_generations = []
for idx, output in enumerate(generations):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    all_generations.append({
        "prompt": prompt,
        "generated_text": generated_text, # needs to be parsed
        "gt_generation": formatted_dataset['train'][idx]['gt_generation'], # this is already a list of tools to be called
        "tools": formatted_dataset['train'][idx]['tools'],
        "idx": formatted_dataset['train'][idx]['id'],
    })

from typing import List, dict

def get_selected_tools_dict(generated_text):
    """Returns a dict of tool_name: list of tool_calls from the generated text in order"""
    pattern =r'<tool_call>(.*?)</tool_call>'
    tools = re.findall(pattern, generated_text, re.DOTALL)
    parsed_tools = []
    for tool in tools:
        try:
            # tool = ast.literal_eval(tool.encode().decode('unicode_escape').strip().replace('"', "'")) # escape chars and some other nonsense
            tool = json.loads(tool.encode().decode('unicode_escape').strip())
            parsed_tools.append(tool)
        except Exception as e:
            print(f"Exception in parsing generated tools for {tool}")
    return parsed_tools

def get_correctness_of_tool_calls(gen_text: str, gt_tools: str):
    gen_tools = get_selected_tools_dict(gen_text)
    gt_tools = json.loads(gt_tools)
    if gen_tools == gt_tools:
        return True
    return False

def get_parsed_tool_calls(gen_text: str, gt_tools: str):
    gen_tools = get_selected_tools_dict(gen_text)
    gt_tools = json.loads(gt_tools)
    return gen_tools, gt_tools

correct_count = 0
correct_len_incorrect = 0
too_many_count = 0
too_few_count = 0
for idx, sample in enumerate(all_generations):
    # print(idx)
    # if get_correctness_of_tool_calls(sample['generated_text'], sample['gt_generation']):
    #     correct_count += 1
    gen_tools, gt_tools = get_parsed_tool_calls(sample['generated_text'], sample['gt_generation'])
    if len(gen_tools) > len(gt_tools):
        too_many_count += 1
    elif len(gen_tools) < len(gt_tools):
        too_few_count += 1
    else:
        if gen_tools == gt_tools:
            correct_count += 1  
        else:
            correct_len_incorrect += 1

print(f"Total samples: {len(all_generations)}")
print(f"Correct tool calls: {correct_count}")
print(f"Correct length but incorrect tool calls: {correct_len_incorrect}")
print(f"Too many tool calls: {too_many_count}")
print(f"Too few tool calls: {too_few_count}")
