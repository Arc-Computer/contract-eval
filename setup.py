#!/usr/bin/env python
"""Setup script for the contract evaluation framework."""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the environment for running the evaluation."""
    
    print("Contract Generation Evaluation Framework Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data",
        "results",
        "results/checkpoints",
        "logs",
        "model_cache"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created {len(directories)} directories")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("\nCreating .env file for API keys...")
        
        env_content = """# API Keys for Contract Evaluation
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
TOGETHER_API_KEY=your_together_key_here

# Optional: HuggingFace token for private models
HF_TOKEN=your_huggingface_token_here
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✓ Created .env file - Please add your API keys")
    else:
        print("✓ .env file exists")
    
    # Install requirements
    print("\nInstalling requirements...")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Update config.yaml with your preferences")
    print("3. Run: python evaluate.py --test  (for testing with 5 examples)")
    print("4. Run: python evaluate.py  (for full evaluation)")

if __name__ == "__main__":
    setup_environment()