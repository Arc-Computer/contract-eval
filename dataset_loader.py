"""Dataset loader for the contract generation dataset."""

import logging
from typing import Dict, Any, List
from datasets import load_dataset
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ContractDatasetLoader:
    """Load and manage the contract generation dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['dataset']
        self.dataset = None
        self.examples = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the dataset from HuggingFace."""
        logger.info(f"Loading dataset from {self.config['repo_id']}...")
        
        try:
            # Load the dataset
            dataset = load_dataset(
                self.config['repo_id'],
                split=self.config['split']
            )
            
            # Load ONLY the last 99 examples (test set not used in SFT)
            # SFT was done on first 500, so we use 500-599 for evaluation
            start_idx = 500  # Skip first 500 used for training
            total_examples = len(dataset)
            
            logger.info(f"Dataset has {total_examples} total examples")
            logger.info(f"Using examples {start_idx} to {total_examples} (last 99 for evaluation)")
            
            # Convert to list of dictionaries
            self.examples = []
            
            # Load from index 500 onwards (last 99 examples)
            for i in range(start_idx, total_examples):
                example = {
                    'id': i - start_idx,  # Reset ID to start from 0
                    'original_idx': i,    # Keep original dataset index
                    'prompt': dataset[i].get('prompt', ''),
                    'thinking_trace': dataset[i].get('thinking_trace', '')  # Correct field name
                }
                
                # Validate that we have the required fields
                if not example['prompt']:
                    logger.warning(f"Example {i} missing prompt, skipping")
                    continue
                    
                if not example['thinking_trace']:
                    logger.warning(f"Example {i} missing thinking_trace, skipping")
                    continue
                    
                self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} valid examples from dataset")
            
            # Save a sample for inspection
            if self.examples:
                sample_path = Path("./data/dataset_sample.json")
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sample_path, 'w') as f:
                    json.dump(self.examples[:5], f, indent=2)
                logger.info(f"Saved dataset sample to {sample_path}")
            
            return self.examples
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            
            # Try to load from local cache if available
            cache_path = Path("./data/cached_dataset.json")
            if cache_path.exists():
                logger.info("Loading from cached dataset...")
                with open(cache_path, 'r') as f:
                    self.examples = json.load(f)
                return self.examples
            
            raise
    
    def save_cache(self):
        """Save dataset to local cache."""
        cache_path = Path("./data/cached_dataset.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(self.examples, f, indent=2)
        logger.info(f"Saved dataset cache to {cache_path}")
    
    def get_example(self, index: int) -> Dict[str, Any]:
        """Get a specific example by index."""
        if not self.examples:
            self.load_dataset()
        
        if 0 <= index < len(self.examples):
            return self.examples[index]
        else:
            raise IndexError(f"Example index {index} out of range (0-{len(self.examples)-1})")
    
    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.examples)
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)