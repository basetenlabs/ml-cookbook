# Gemma 3 27B Axolotl Fine Tuning

## Introduction

This example fine tunes [Gemma 3 27b](https://huggingface.co/google/gemma-3-27b-it) on the `winglian/pirate-ultrachat-10k` dataset using Axolotl with maximum sequence lengths of 16k. It uses FSDP and sequence parallelism to shard the model parameters and sequences across 8 H100 GPUs.

## Getting started

Gemma 3 27b is a gated model on Hugging Face, please make sure do the following:
1.  Accept the [terms and conditions](https://huggingface.co/google/gemma-3-27b-it) for this model on Hugging Face
2.  Add your Hugging Face token as a [secret](https://app.baseten.co/settings/secrets) named `hf_access_token` on Baseten

To launch this job:
```bash
truss train push config.py
```

## Deploying the trained model
After the training run completes, you can view and deploy the checkpoint(s) under the training project on the [Baseten dashboard](https://app.baseten.co/training). 

💡 Tip: This model requires at least 2 H00s to deploy, make sure to select the `H100:2` instance type or higher. If you do not see this instance type listed, please [contact us](https://www.baseten.co/talk-to-us/) to enable it for you.

Note that this is __not__ a production-ready deployment and is meant for testing purposes only, please [contact us](https://www.baseten.co/talk-to-us/) to get the best performance and reliability for your inference workloads.

After the deploy succeeds, you can use the playground to send an inference request. The server is [OpenAI compatible](https://github.com/openai/openai-python) and supports all of the chat completion parameters. A simple example payload:
```json
{
    "messages": [
        {
            "role": "user",
            "content": "What is a prime number?"
        }
    ]
}
```
Sample respose:
```json
{
  "id": "chatcmpl-0565f460-8dd9-42b8-8153-a35bfafac33f",
  "object": "chat.completion",
  "created": 1764989880,
  "model": "baseten-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Ahoy there, matey!  A prime number be an integer greater than 2, only divisible by anything from itself up to the square root of itself! Aye, aye, captain!  It be a whole, positive, and un-decimal number, nothin' but two!  Arrr, ye be lookin' for a pirate's prime number, ain't ya?",
       ...
```

## Resources
### Axolotl:
* [Homepage](https://docs.axolotl.ai/)
* [Config reference](https://docs.axolotl.ai/docs/config-reference.html)

### Baseten
* [Homepage](https://www.baseten.co/)
* [Training on our platform](https://docs.baseten.co/training/overview)
* [Config reference](https://docs.baseten.co/reference/sdk/training)