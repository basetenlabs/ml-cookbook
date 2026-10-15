import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'nuprl/MultiPL-E'

    train_dataset = datasets.load_dataset(data_source, 'mbpp-ml')
    test_dataset = datasets.load_dataset(data_source, 'humaneval-ml')

    train_dataset = train_dataset['test']
    test_dataset = test_dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")

            system = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'

            tests = example.pop("tests")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "coding",
                "reward_model": {"style": "rule", "ground_truth": 0.0},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "tests": tests,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
