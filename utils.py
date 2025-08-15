"""Utility functions for the evaluation framework."""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging(log_dir: str = "./logs"):
    """Set up logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/evaluation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if available
    config['api_keys']['openai'] = os.getenv('OPENAI_API_KEY', config['api_keys']['openai'])
    config['api_keys']['anthropic'] = os.getenv('ANTHROPIC_API_KEY', config['api_keys']['anthropic'])
    config['api_keys']['google'] = os.getenv('GOOGLE_API_KEY', config['api_keys']['google'])
    config['api_keys']['together'] = os.getenv('TOGETHER_API_KEY', config['api_keys']['together'])
    
    return config

def create_directories(config: Dict[str, Any]):
    """Create necessary directories for results and checkpoints."""
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)

def save_checkpoint(results: list, checkpoint_num: int, config: Dict[str, Any]):
    """Save intermediate checkpoint."""
    checkpoint_path = Path(config['paths']['checkpoints_dir']) / f"checkpoint_{checkpoint_num:03d}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2)
    return checkpoint_path

def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load checkpoint and return completed results and last index."""
    with open(checkpoint_path, 'r') as f:
        results = json.load(f)
    last_idx = len(results)
    return results, last_idx

def extract_steps_from_thinking(thinking_text: str) -> list:
    """Extract INSTRUCTION blocks from teacher's output using tags."""
    import re
    
    instructions = []
    
    # First, extract content from instruction tags
    tag_match = re.search(r'<instructions>(.*?)(?:</instructions>|$)', thinking_text, re.DOTALL)
    
    if tag_match:
        instructions_content = tag_match.group(1)
        
        # Split by INSTRUCTION markers
        parts = instructions_content.split('INSTRUCTION ')
        
        for part in parts[1:]:  # Skip first part before any INSTRUCTION
            # Each part starts with the instruction number
            instruction_text = part.strip()
            
            # Remove the number and colon at the start
            if ':' in instruction_text:
                instruction_text = instruction_text.split(':', 1)[1].strip()
            
            # Clean up any placeholders
            instruction_text = instruction_text.replace('[', '').replace(']', '')
            
            # Stop at next instruction or end
            if 'INSTRUCTION' in instruction_text:
                instruction_text = instruction_text.split('INSTRUCTION')[0].strip()
            
            if instruction_text:
                instructions.append(instruction_text)
    
    # Validate we got instructions
    if not instructions:
        # Fatal error - teacher didn't format correctly
        raise ValueError(f"Teacher failed to generate valid instructions. Output was: {thinking_text[:500]}")
    
    return instructions

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def api_call_with_retry(func, *args, **kwargs):
    """Wrapper for API calls with retry logic."""
    return func(*args, **kwargs)

def format_results_summary(results: Dict[str, Any]) -> str:
    """Format evaluation results for display."""
    total = sum(results['win_counts'].values())
    summary = f"\n{'='*50}\n"
    summary += f"EVALUATION RESULTS SUMMARY\n"
    summary += f"{'='*50}\n\n"
    summary += f"Total Examples Evaluated: {total}\n\n"
    summary += f"Win Counts:\n"
    for model, count in results['win_counts'].items():
        win_rate = (count / total * 100) if total > 0 else 0
        summary += f"  {model}: {count} ({win_rate:.1f}%)\n"
    summary += f"\n{'='*50}\n"
    return summary