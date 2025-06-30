import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def save_model_in_pth_format(model_name, output_dir):
    """
    Download a model from Hugging Face and save it in PTH format
    for use with quantization benchmarks.
    
    Args:
        model_name: Name of the model on Hugging Face
        output_dir: Directory to save the model
    """
    print(f"Loading model {model_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Extract and save the model weights in PTH format
    model_state_dict = model.state_dict()
    
    # Save the model weights
    model_path = Path(output_dir) / "model.pth"
    torch.save(model_state_dict, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Print model size information
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return model_path

if __name__ == "__main__":
    # Use a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./tinyllama-1.1b-chat"
    
    model_path = save_model_in_pth_format(model_name, output_dir)
    
    print("\nYou can now use this model with the INT4 benchmark script.")
    print("Example command:")
    print(f"python int4benchmark.py --model_path {model_path}") 