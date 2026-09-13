import ast
import re
import json
import jsonschema
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, load_dataset, DatasetDict
from vllm import LLM, SamplingParams

ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")

# 4. Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"  # or Qwen2.5-3B, Qwen2.5-1.5B
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(x, add_generation_prompt=True):
    try:
        pattern = r'<tools>(.*?)</tools>'
        tools = [s['function'] for s in ast.literal_eval(re.findall(pattern, x['conversations'][0]['value'], re.DOTALL)[-1].strip())]
        text = tokenizer.apply_chat_template(
            [
                # {"role": "system", "content": "You are an assistant capable of tool calls"}, 
                {"role": "user", "content": x['conversations'][-2]['value']}, 
                # {"role": "assistant", "content": x['conversations'][-1]['value']}
            ], 
            tools=tools, 
            add_generation_prompt=add_generation_prompt, 
            tokenize=False
        )
        return {
            "text": text,
            # "tools": tools,
            "gt_generation": x['conversations'][-1]['value'],
            }
    except Exception as e:
        print(e)
        # Return None or a sentinel value to filter out later
        return None

formatted_dataset = ds.map(format_example).filter(lambda x: x is not None)

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)
prompts = [sample['text'] for sample in formatted_dataset['train']]
generations = llm.generate(prompts, sampling_params)

all_generations = []
for idx, output in enumerate(generations):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    all_generations.append({
        "prompt": prompt,
        "generated_text": generated_text,
        "gt_generation": formatted_dataset['train'][idx]['gt_generation'],
        "conversations": formatted_dataset['train'][idx]['conversations'],
        "idx": formatted_dataset['train'][idx]['id'],
    })

## Sometimes GT generation is garbage, make sure they are approximately correct with. json schema
def get_all_available_tools_dict(prompt):
    pattern = r'<tools>(.*?)</tools>'
    try:
        data_string = re.findall(pattern, prompt, re.DOTALL)[-1].strip()
        json_objects = []
        for line in data_string.strip().split('\n'):
            if line:  # Skip empty lines
                json_objects.append(json.loads(line))
        tool_dict = {}
        for obj in json_objects:
            tool_dict[obj['name']] = obj
        return tool_dict
    except exception as e:
        print(f"Exception {e}")
        return {}

def get_selected_tools_dict(generated_text):
    pattern =r'<tool_call>(.*?)</tool_call>'
    tools = re.findall(pattern, generated_text, re.DOTALL)
    tool_dict = {}
    for tool in tools:
        try:
            tool = ast.literal_eval(tool.encode().decode('unicode_escape').strip().replace('"', "'")) # escape chars and some other nonsense
            if 'name' not in tool:
                print(" No name in tool???")
                continue
            tool_dict[tool['name']] = tool_dict.get(tool['name'], []) + [tool]
        except Exception as e:
            print(f"Exception in get_selected_tools_dict for {tool}")
    return tool_dict

def tools_validity(avail_tools, tools):
    if not tools:
        return False
    for tool_name, tools in tools.items():
        for tool in tools:
            if tool["name"] not in avail_tools:
                print(" Hallucinating b**ch??")
                return False
            tool_name = tool["name"]
            schema = avail_tools[tool_name].get("parameters", {}) # TODO - might depend on dataset
            gen = tool["arguments"]
            try:
                jsonschema.validate(instance=gen, schema=schema)
            except jsonschema.ValidationError as e:
                print(f"Valid data validation error: {e.message}")
                return False
    # all tools validated
    return True


success = 0
success_idxs = []
for idx, gen in enumerate(all_generations):
    avail_tools = get_all_available_tools_dict(gen['prompt'])
    # gen_tools = get_selected_tools_dict(gen['generated_text'])
    ref_tools = get_selected_tools_dict(gen['gt_generation'])
    validity = tools_validity(avail_tools, ref_tools)
    if validity:
        success += 1
        success_idxs.append(idx)
    
filtered_generations = [all_generations[idx] for idx in success_idxs]

new_ds = Dataset.from_list(filtered_generations)
dataset_dict = DatasetDict({"train": new_ds})
dataset_dict.push_to_hub("baseten-admin/CleanNousResearch_simple")