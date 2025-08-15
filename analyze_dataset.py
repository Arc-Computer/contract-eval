"""Script to analyze the dataset and understand its structure."""

from datasets import load_dataset
import json
from pathlib import Path

# Load the dataset
print("Loading dataset...")
dataset = load_dataset(
    "Jarrodbarnes/rcl-specialized-teacher-enterprise",
    split="train"
)

print(f"\nDataset size: {len(dataset)} examples")
print(f"Features: {dataset.features}")
print("\n" + "="*80)

# Analyze first few examples
num_examples_to_analyze = 5

for i in range(min(num_examples_to_analyze, len(dataset))):
    print(f"\n\nEXAMPLE {i+1}:")
    print("="*80)
    
    example = dataset[i]
    
    # Print prompt
    if 'prompt' in example:
        print("\nPROMPT:")
        print("-"*40)
        print(example['prompt'][:500] + "..." if len(example['prompt']) > 500 else example['prompt'])
    
    # Print thinking_trace (correct field name)
    if 'thinking_trace' in example:
        print("\nTHINKING_TRACE:")
        print("-"*40)
        content = example['thinking_trace']
        
        # Check for thinking tags
        if "<thinking>" in content:
            thinking_start = content.find("<thinking>")
            thinking_end = content.find("</thinking>")
            if thinking_end > thinking_start:
                thinking = content[thinking_start+10:thinking_end]
                print("\nTHINKING SECTION (first 1000 chars):")
                print(thinking[:1000] + "..." if len(thinking) > 1000 else thinking)
                
                contract_part = content[thinking_end+11:].strip()
                print("\nCONTRACT/CONTENT SECTION (first 1000 chars):")
                print(contract_part[:1000] + "..." if len(contract_part) > 1000 else contract_part)
        else:
            print(content[:2000] + "..." if len(content) > 2000 else content)
    
    print("\n" + "="*80)

# Save sample for analysis
print("\n\nSaving sample to file for detailed analysis...")
sample = []
for i in range(min(10, len(dataset))):
    example = dataset[i]
    thinking_trace = example.get('thinking_trace', '')
    sample.append({
        'id': i,
        'prompt': example.get('prompt', ''),
        'thinking_trace': thinking_trace,
        'has_thinking_tags': '<thinking>' in thinking_trace,
        'has_steps': 'Step ' in thinking_trace or 'step ' in thinking_trace,
        'trace_length': len(thinking_trace)
    })

with open('dataset_sample_analysis.json', 'w') as f:
    json.dump(sample, f, indent=2)

print("Sample saved to dataset_sample_analysis.json")

# Analyze patterns
print("\n\nPATTERN ANALYSIS:")
print("="*80)

thinking_count = 0
step_count = 0
contract_count = 0
guide_count = 0
prompt_types = {}

for i in range(min(100, len(dataset))):
    example = dataset[i]
    thinking_trace = example.get('thinking_trace', '')
    prompt = example.get('prompt', '').lower()
    
    if '<thinking>' in thinking_trace:
        thinking_count += 1
    
    if 'Step ' in thinking_trace or 'step ' in thinking_trace:
        step_count += 1
    
    if 'CONTRACT' in thinking_trace.upper() or 'AGREEMENT' in thinking_trace.upper():
        contract_count += 1
    
    # Analyze prompt types
    if 'contract' in prompt or 'agreement' in prompt:
        prompt_types['contract'] = prompt_types.get('contract', 0) + 1
    elif 'guide' in prompt or 'training' in prompt:
        prompt_types['guide/training'] = prompt_types.get('guide/training', 0) + 1
    elif 'nda' in prompt or 'non-disclosure' in prompt:
        prompt_types['nda'] = prompt_types.get('nda', 0) + 1
    elif 'service' in prompt:
        prompt_types['service'] = prompt_types.get('service', 0) + 1
    else:
        prompt_types['other'] = prompt_types.get('other', 0) + 1

print(f"Examples with <thinking> tags: {thinking_count}/100")
print(f"Examples with 'Step' mentions: {step_count}/100")
print(f"Examples with CONTRACT/AGREEMENT: {contract_count}/100")
print(f"\nPrompt types distribution:")
for prompt_type, count in prompt_types.items():
    print(f"  {prompt_type}: {count}")

print("\n\nAnalysis complete!")