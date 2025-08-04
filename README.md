<div align="center">

  <a href="https://www.baseten.co/"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="">
    <source media="(prefers-color-scheme: light)" srcset="">
    <img alt="baseten logo" src="" height="110" style="max-width: 100%;">
  </picture></a>
  
<!-- <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/start free finetune button.png" width="154"></a> -->
<a href="https://docs.baseten.co/training/overview"><img src="" width="165"></a>
<a href="https://docs.baseten.co"><img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/Documentation%20Button.png" width="137"></a>

### From training to serving - recipes for every ML appetite!

<!-- ![](https://i.ibb.co/sJ7RhGG/image-41.png) -->

</div>

# ML Training & Deployment Workflows
A collection of ready-to-use training recipes, fine-tuning examples, and deployment workflows for machine learning on Baseten. Whether you're training from scratch, fine-tuning existing models, or deploying for inference, these recipes provide practical, copy-paste solutions for your ML pipeline.

### What's Inside

- Training recipes - End-to-end examples for training models from scratch
- Fine-tuning workflows - Adapt pre-trained models to your specific use case
- Deployment guides - Get your trained models serving predictions in production
- Best practices - Optimized configurations and common patterns

From data preprocessing to production deployment, these recipes cover the complete ML lifecycle on Baseten's platform.

# Table of Contents

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Recipes](#recipes)
- [Contributing](#contributing)
- [License](#license)


## Getting Started

### Prerequisites

Before getting started, ensure you have the following:

- A Baseten account. [Sign up here](https://baseten.co/signup) if you don't have one.
    - Add any access tokens, API keys (Example: Huggingface access token, Weights&Biases access token), passwords to securely access credentials from your models in [secrets](https://app.baseten.co/settings/secrets).
- Python 3.8 to 3.11 installed. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) env recommended.
- Install [Truss](https://github.com/basetenlabs/truss), Baseten's open-source model packaging tool to configure and containerize model code.


### Clone this repository:

```bash
git clone https://github.com/basetenlabs/ml-cookbook.git
```

## Usage

#### Train an MNIST digit classifier with Pytorch. 

```bash
python train.py --config configs/llm_example.yaml
```

#### Fine-tune Llama 3 8b Instruct 

- Make sure you have access to the model on [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and your API token uploaded to your [secrets](https://app.baseten.co/settings/secrets).

```bash

```

## Recipes

- `data_preprocessing/`: Scripts for cleaning and preparing datasets.
- `configs/`: Example configuration files for different model architectures.
- `training/`: Training scripts and utilities.
- `evaluation/`: Evaluation and benchmarking tools.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

[MIT License](LICENSE)