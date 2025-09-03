import asyncio
import aiohttp
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from openai import OpenAI, AsyncOpenAI
import time
from copy import deepcopy
import json

# extra utils 
def dict_of_lists_to_list_of_dicts(data: Dict[str, List]) -> List[Dict]:
    """
    Convert dict of lists to list of dicts
    
    Example:
    {'name': ['Alice', 'Bob'], 'age': [25, 30]} 
    -> [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
    """
    if not data:
        return []
    
    # Get the length from the first list
    length = len(next(iter(data.values())))
    
    # Convert to list of dicts
    return [
        {key: data[key][i] for key in data.keys()}
        for i in range(length)
    ]


OP_RESPONSE_PATH = "responses/llama3.3_70b_instruct_responses_first500.jsonl"
GPTOSS120B_RESPONSE_PATH = "responses/gptoss120b_responses_first500.jsonl"

async def send_batch_requests(client, batch: List[Dict], batch_size: int = 10):
    """Send a batch of requests concurrently"""
    async def make_request(sample):
        try:
            response = await client.chat.completions.create(
                # model="meta-llama/Llama-3.3-70B-Instruct",
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        # "content": sample["merged_messages"][0]["content"]
                        "content": sample[0]["content"]
                    },
                    {
                        "role": "user", 
                        # "content": sample["merged_messages"][1]["content"]
                        "content": sample[1]["content"]
                    }
                ],
            )
            print("Received response")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    # Create tasks for all samples in the batch
    tasks = [make_request(sample) for sample in batch["merged_messages"]]
    
    # Execute all requests concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def process_dataset_in_batches(dataset, client, batch_size: int = 10):
    """Process entire dataset in batches"""
    responses = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} (samples {i+1}-{min(i+batch_size, len(dataset))})")
        
        batch_results = await send_batch_requests(client, batch, batch_size)

        op_batch = {
            "interaction_id": batch["interaction_id"],
            "messages": batch["messages"],
            "output": batch["output"],
            "model": batch["model"],
        }
        # op_batch['llama3.3_70b_instruct_responses'] = batch_results
        op_batch['gptoss120b_instruct_responses'] = batch_results

        op_batch_list = dict_of_lists_to_list_of_dicts(op_batch)

        with open(GPTOSS120B_RESPONSE_PATH, 'a+', encoding='utf-8') as f:
            for item in op_batch_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Completed batch {i//batch_size + 1}")

# Usage
async def main():
    batch_size = 64
    
    # llama 3.3 70b instruct
    # client = AsyncOpenAI(
    #     api_key="5w0kcqE4.XoqP0e4crHwsP4PlPxqJk5XATBDkkdUY",
    #     base_url="https://model-4q9l0j9q.api.baseten.co/environments/production/sync/v1"
    # )

    # gptoss 120b
    client = AsyncOpenAI(
        api_key="5w0kcqE4.XoqP0e4crHwsP4PlPxqJk5XATBDkkdUY",
        base_url="https://model-6wg94lg3.api.baseten.co/environments/production/sync/v1"
    )

    TEST_DATASET = "baseten-admin/gamma-20aug2025-claudesonnet-bad"
    TEST_SPLIT = "test[:500]"
    dataset = load_dataset(TEST_DATASET, split=TEST_SPLIT)

    await process_dataset_in_batches(dataset, client, batch_size)

# Run the async function
asyncio.run(main())