from datasets import load_dataset, Dataset
from openai import OpenAI


TEST_DATASET = "baseten-admin/gamma-20aug2025-claudesonnet-bad"
TEST_SPLIT = "test"

test_dataset = load_dataset(TEST_DATASET, split=TEST_SPLIT)

def get_generations(
    model_url: str,
    api_key: str,
    dataset: Dataset,
    batch_size: int = 1
):
    client = OpenAI(
        api_key=api_key,
        base_url=model_url,
    )
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} (samples {i+1}-{min(i+batch_size, len(dataset))})")

    # for sample in dataset:
    #     response = client.chat.completions.create(
    #         model="meta-llama/Llama-3.3-70B-Instruct",
    #         messages=[
    #             {
    #                 "role":"system",
    #                 "content": sample["messages"][0]["content"]
    #             },
    #             {
    #                 "role":"user",
    #                 "content": sample["messages"][1]["content"]
    #             }
    #         ],
    #     )
    #     print(response.choices[0].message.content)

get_generations(
    model_url="https://model-4q9l0j9q.api.baseten.co/environments/production/sync/v1",
    api_key="5w0kcqE4.XoqP0e4crHwsP4PlPxqJk5XATBDkkdUY",
    dataset=test_dataset,
    batch_size=4,
)

# client = OpenAI(
#     api_key="5w0kcqE4.XoqP0e4crHwsP4PlPxqJk5XATBDkkdUY",
#     base_url="https://model-4q9l0j9q.api.baseten.co/environments/production/sync/v1"
# )

# sample = good_ds['train'][5]
# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.3-70B-Instruct",
#     messages=[
#         {
#             "role":"system",
#             "content": sample["messages"][0]["content"]
#         },
#         {
#             "role":"user",
#             "content": sample["messages"][1]["content"]
#         }
#     ],
# )

# def merge_messages(row):
#     row['merged_messages'] = [
#         {"role": "system", "content": row['messages'][0]['content']},
#         {"role": "user", "content": row['messages'][1]['content']},
#         {"role": "assistant", "content": row['output']}
#     ]
#     return row
# good_samples_fixed = good_samples.apply(merge_messages, axis=1)