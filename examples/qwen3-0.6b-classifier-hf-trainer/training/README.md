## Qwen3-0.6B Sequence Classification with HuggingFace Trainer

This example demonstrates how to fine-tune Qwen3-0.6B for binary text classification (IMDB sentiment analysis) using HuggingFace's `AutoModelForSequenceClassification` and `Trainer` API.

### Key Features

- Uses `AutoModelForSequenceClassification` instead of custom model architecture
- Leverages HuggingFace `Trainer` for simplified training loop
- Automatic handling of data collation, evaluation, and checkpointing
- Built-in metrics computation and evaluation

### Differences from the PyTorch version

- **Model**: Uses `AutoModelForSequenceClassification` which automatically adds a classification head
- **Training**: Uses HuggingFace `Trainer` instead of manual PyTorch training loop
- **Data Handling**: Uses `DataCollatorWithPadding` for efficient batching
- **Evaluation**: Built-in evaluation with automatic metric computation

## Run instructions

### Launch run 

```
truss train push config.py
```

Upon successful submission, the CLI will output helpful information about your job, including the job-id to track your run.

