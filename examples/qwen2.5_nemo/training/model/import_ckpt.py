import argparse
import nemo_run as run
import lightning.pytorch as pl
from nemo.collections import llm
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Hugging Face checkpoint to NeMo format")
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen2_7b",
        choices=["qwen2_7b"],  # Add more models as needed
        help="Model type to use"
    )
    
    # Source configuration
    parser.add_argument(
        "--source", 
        type=str, 
        default="hf://Qwen/Qwen2.5-7B-Instruct",
        help="Source path for the model (e.g., hf://model-name or local path)"
    )
    
    # Overwrite option
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        default=True,
        help="Whether to overwrite existing checkpoint"
    )
    
    # Executor type
    parser.add_argument(
        "--executor", 
        type=str, 
        default="local",
        choices=["local", "slurm"],
        help="Executor type to use"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for converted checkpoint"
    )
    
    return parser.parse_args()

def get_model_config(model_name):
    """Get model configuration based on model name"""
    model_configs = {
        "qwen2_7b": llm.qwen2_7b.model(),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model_configs[model_name]

def configure_checkpoint_conversion(args):
    """Configure checkpoint conversion with command line arguments"""
    model_config = get_model_config(args.model)
    
    conversion_config = {
        "model": model_config,
        "source": args.source,
        "overwrite": args.overwrite,
    }
    
    # Add output directory if specified
    if args.output_dir:
        conversion_config["output_dir"] = args.output_dir
    
    return run.Partial(llm.import_ckpt, **conversion_config)

def get_executor(executor_type):
    """Get executor based on type"""
    if executor_type == "local":
        return run.LocalExecutor()
    elif executor_type == "slurm":
        return run.SlurmExecutor()
    else:
        raise ValueError(f"Unsupported executor type: {executor_type}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Using model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Executor: {args.executor}")
    
    # Configure checkpoint conversion
    import_ckpt = configure_checkpoint_conversion(args)
    
    # Define executor
    executor = get_executor(args.executor)
    
    # Run experiment
    print("Starting checkpoint conversion...")
    run.run(import_ckpt, executor=executor)
    print("Checkpoint conversion completed!")

if __name__ == "__main__":
    main()