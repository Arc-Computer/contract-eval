"""Main evaluation script for contract generation."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

from utils import (
    setup_logging, 
    load_config, 
    create_directories,
    save_checkpoint,
    load_checkpoint,
    format_results_summary
)
from models import TeacherModel, StudentModel, BaselineModels
from judge import GeminiJudge
from dataset_loader import ContractDatasetLoader

logger = logging.getLogger(__name__)

class ContractEvaluator:
    """Main evaluator for contract generation comparison."""
    
    def __init__(self, config_path: str = "config.yaml", resume_from: str = None):
        self.config = load_config(config_path)
        create_directories(self.config)
        
        # Initialize models
        logger.info("Initializing models...")
        self.teacher = TeacherModel(self.config)
        self.student = StudentModel(self.config)
        self.baselines = BaselineModels(self.config)
        self.judge = GeminiJudge(self.config)
        
        # Load dataset
        self.dataset_loader = ContractDatasetLoader(self.config)
        self.dataset = self.dataset_loader.load_dataset()
        
        # Initialize results
        self.results = []
        self.win_counts = {
            "teacher_student": 0,
            "gpt5": 0,
            "claude_opus": 0
        }
        
        # Resume from checkpoint if specified
        self.start_idx = 0
        if resume_from:
            self.results, self.start_idx = load_checkpoint(resume_from)
            # Recalculate win counts from loaded results
            for result in self.results:
                winner = result['evaluation'].get('winner_model', 'unknown')
                if winner in self.win_counts:
                    self.win_counts[winner] += 1
            logger.info(f"Resumed from checkpoint at example {self.start_idx}")
    
    def save_contracts_json(self, example_id: int, original_idx: int, result: Dict[str, Any]):
        """Save individual contract outputs for human validation."""
        contracts_dir = Path(self.config['paths']['results_dir']) / 'contracts'
        contracts_dir.mkdir(parents=True, exist_ok=True)
        
        contract_file = contracts_dir / f"example_{example_id:03d}_orig_{original_idx:03d}.json"
        
        # Structure for human validation
        validation_data = {
            'example_id': example_id,
            'original_dataset_index': original_idx,  # Track original position in dataset
            'prompt': result.get('prompt', ''),
            'golden_standard': result.get('thinking_trace', ''),
            'teacher_instructions': result.get('teacher_steps', []),
            'contracts': {
                'teacher_student': result.get('student_contract', ''),
                'gpt5': result.get('gpt5_contract', ''),
                'claude': result.get('claude_contract', '')
            },
            'evaluation': result.get('evaluation', {}),
            'winner': result.get('evaluation', {}).get('winner_model', 'unknown'),
            'processing_time': result.get('processing_time', 0)
        }
        
        with open(contract_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        logger.debug(f"Saved contracts to {contract_file}")
    
    def evaluate_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single example through the full pipeline."""
        
        logger.info(f"Processing example {example['id']} (original dataset index: {example.get('original_idx', 'N/A')})")
        start_time = time.time()
        
        result = {
            'example_id': example['id'],
            'original_idx': example.get('original_idx', example['id']),
            'prompt': example['prompt'],
            'thinking_trace': example['thinking_trace']  # Correct field name
        }
        
        try:
            # Phase 1: Teacher generates step-by-step instructions
            logger.info("Phase 1: Teacher generating instructions...")
            logger.info(f"EVAL: Sending prompt to teacher: {example['prompt'][:100]}...")
            teacher_steps = self.teacher.generate_instructions(example['prompt'])
            logger.info(f"EVAL: Received {len(teacher_steps)} steps from teacher")
            result['teacher_steps'] = teacher_steps
            result['num_steps'] = len(teacher_steps)
            
            # Phase 2: Student generates contract following steps
            logger.info(f"Phase 2: Student generating contract with {len(teacher_steps)} steps...")
            student_contract = self.student.generate_contract_stepwise(
                example['prompt'], 
                teacher_steps
            )
            result['student_contract'] = student_contract
            
            # Phase 3: Baseline generations (can be done in parallel)
            logger.info("Phase 3: Generating baseline contracts...")
            
            # GPT-5 generation
            logger.info("  - Generating with GPT-5...")
            gpt5_contract = self.baselines.generate_gpt5(example['prompt'])
            result['gpt5_contract'] = gpt5_contract
            
            # Claude generation
            logger.info("  - Generating with Claude...")
            claude_contract = self.baselines.generate_claude(example['prompt'])
            result['claude_contract'] = claude_contract
            
            # Phase 4: Judge evaluation
            logger.info("Phase 4: Judge evaluating all contracts...")
            evaluation = self.judge.evaluate_contracts(
                prompt=example['prompt'],
                thinking_trace=example['thinking_trace'],  # Correct field name
                teacher_student_contract=student_contract,
                gpt5_contract=gpt5_contract,
                claude_contract=claude_contract
            )
            result['evaluation'] = evaluation
            
            # Update win count
            winner = evaluation.get('winner_model', 'unknown')
            if winner in self.win_counts:
                self.win_counts[winner] += 1
                logger.info(f"Winner: {winner}")
            
            # Record timing
            result['processing_time'] = time.time() - start_time
            logger.info(f"Example {example['id']} completed in {result['processing_time']:.2f} seconds")
            
            # Save contracts for human validation
            self.save_contracts_json(example['id'], example.get('original_idx', example['id']), result)
            
        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            result['error'] = str(e)
            result['evaluation'] = {
                'winner_model': 'error',
                'reasoning': f"Error during evaluation: {str(e)}"
            }
            # Save even failed attempts for debugging
            self.save_contracts_json(example['id'], example.get('original_idx', example['id']), result)
        
        return result
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        
        logger.info(f"Starting evaluation of {len(self.dataset)} examples")
        logger.info(f"Starting from index {self.start_idx}")
        
        # Progress bar
        pbar = tqdm(
            range(self.start_idx, len(self.dataset)),
            desc="Evaluating",
            total=len(self.dataset) - self.start_idx
        )
        
        for idx in pbar:
            example = self.dataset[idx]
            
            # Evaluate single example
            result = self.evaluate_single_example(example)
            self.results.append(result)
            
            # Update progress bar with current win rates
            total_evaluated = len(self.results)
            win_rates_str = " | ".join([
                f"{model}: {(count/total_evaluated*100):.1f}%"
                for model, count in self.win_counts.items()
            ])
            pbar.set_postfix_str(win_rates_str)
            
            # Save checkpoint at intervals
            if (idx + 1) % self.config['evaluation']['checkpoint_interval'] == 0:
                checkpoint_path = save_checkpoint(
                    self.results,
                    idx + 1,
                    self.config
                )
                logger.info(f"Saved checkpoint at example {idx + 1}: {checkpoint_path}")
        
        # Save final results
        self.save_final_results()
    
    def save_final_results(self):
        """Save the final evaluation results."""
        
        # Calculate final statistics
        total_examples = len(self.results)
        win_rates = {
            model: (count / total_examples * 100) if total_examples > 0 else 0
            for model, count in self.win_counts.items()
        }
        
        # Prepare final report
        final_report = {
            'metadata': {
                'total_examples': total_examples,
                'dataset': self.config['dataset']['repo_id'],
                'teacher_model': self.config['models']['teacher']['repo_id'],
                'student_model': self.config['models']['student']['model_name'],
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'win_counts': self.win_counts,
            'win_rates': win_rates,
            'detailed_results': self.results
        }
        
        # Save to file
        report_path = Path(self.config['paths']['final_report'])
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"Saved final report to {report_path}")
        
        # Print summary
        print(format_results_summary(final_report))
        
        # Save a separate summary file
        summary_path = report_path.parent / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(format_results_summary(final_report))
            f.write("\n\nDetailed Analysis:\n")
            f.write("="*50 + "\n\n")
            
            # Add per-example analysis
            for result in self.results:
                f.write(f"Example {result['example_id']}:\n")
                f.write(f"  Prompt: {result['prompt'][:100]}...\n")
                f.write(f"  Winner: {result['evaluation'].get('winner_model', 'unknown')}\n")
                f.write(f"  Reasoning: {result['evaluation'].get('reasoning', 'N/A')}\n")
                f.write(f"  Processing Time: {result.get('processing_time', 0):.2f}s\n")
                f.write("-"*30 + "\n")
        
        logger.info(f"Saved summary to {summary_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate contract generation models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test on first 5 examples only"
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["api", "local", "hybrid"],
        default=None,
        help="Override inference mode from config (api/local/hybrid)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Contract Generation Evaluation")
    
    # Load config
    config = load_config(args.config)
    
    # Override inference mode if specified
    if args.inference_mode:
        config['inference_mode'] = args.inference_mode
        logger.info(f"Using inference mode: {args.inference_mode}")
    
    # Modify config for test mode
    if args.test:
        config['dataset']['num_examples'] = 5
        config['evaluation']['checkpoint_interval'] = 2
        
        # Save modified config
        test_config_path = "config_test.yaml"
        import yaml
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f)
        args.config = test_config_path
        logger.info("Running in TEST mode with 5 examples")
    
    # Run evaluation
    evaluator = ContractEvaluator(args.config, args.resume)
    evaluator.run_evaluation()
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()