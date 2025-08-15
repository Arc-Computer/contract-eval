# Contract Generation Evaluation Framework

This framework evaluates the effectiveness of a teacher-student approach for contract generation, comparing it against state-of-the-art models.

## Overview

The evaluation pipeline tests whether a fine-tuned teacher model (et-8b) can guide a smaller student model (Qwen3-8B) to generate contracts that match or exceed the quality of direct generation from GPT-5 and Claude Opus 4.1.

### Key Features

- **Step-by-Step Teaching**: Teacher model generates incremental instructions, student follows each step
- **Multi-Model Comparison**: Teacher-Student vs GPT-5 vs Claude Opus 4.1
- **Automated Judging**: Gemini 2.5 Pro evaluates all outputs
- **Checkpoint System**: Resume from interruptions
- **Comprehensive Logging**: Detailed tracking of all operations

## Architecture

```
Pipeline Flow:
1. Teacher Model (et-8b) → Generates step-by-step instructions
2. Student Model (Qwen3-8B) → Follows steps to build contract
3. Baseline Models → Direct generation without guidance
4. Judge (Gemini 2.5) → Evaluates and picks winner
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for teacher model)
- API keys for OpenAI, Anthropic, Google, and Together

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd contract-eval
```

2. Run setup:
```bash
python setup.py
```

3. Configure API keys in `.env`:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Update `config.yaml` with your preferences

## Usage

### Test Run (5 examples)
```bash
python evaluate.py --test
```

### Full Evaluation (100 examples)
```bash
python evaluate.py
```

### Resume from Checkpoint
```bash
python evaluate.py --resume results/checkpoints/checkpoint_050.json
```

## Configuration

Edit `config.yaml` to customize:

- Model parameters (temperature, max tokens)
- Dataset settings (number of examples)
- API endpoints and models
- Checkpoint intervals
- Output paths

## Project Structure

```
contract-eval/
├── config.yaml           # Main configuration
├── evaluate.py          # Main evaluation script
├── models.py            # Model loading and generation
├── judge.py             # Gemini judge implementation
├── dataset_loader.py    # Dataset management
├── utils.py             # Utility functions
├── setup.py             # Setup script
├── requirements.txt     # Python dependencies
├── data/                # Dataset cache
├── results/             # Evaluation results
│   ├── checkpoints/     # Intermediate checkpoints
│   └── final_report.json
└── logs/                # Execution logs
```

## Models

### Teacher Model
- **Model**: aman-jaglan/et-8b (HuggingFace)
- **Role**: Generate step-by-step instructions without contract text
- **Architecture**: 8B parameter model with distributed checkpoints

### Student Model
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Role**: Generate contract following teacher's steps
- **Provider**: Together AI

### Baseline Models
- **GPT-5**: Direct contract generation (using GPT-4o until GPT-5 available)
- **Claude Opus 4.1**: Direct contract generation (using Claude 3.5 Sonnet until Opus 4.1 available)

### Judge
- **Model**: Gemini 2.5 Pro (gemini-2.0-flash-exp)
- **Role**: Evaluate all contracts against golden standard

## Dataset

- **Source**: Jarrodbarnes/rcl-specialized-teacher-enterprise
- **Format**: Each example contains:
  - `prompt`: Contract generation request
  - `thinking_traces`: Expert thinking process + ideal contract

## Output

### Results Structure
```json
{
  "metadata": {
    "total_examples": 100,
    "dataset": "...",
    "evaluation_date": "..."
  },
  "win_counts": {
    "teacher_student": 45,
    "gpt5": 30,
    "claude_opus": 25
  },
  "win_rates": {
    "teacher_student": 45.0,
    "gpt5": 30.0,
    "claude_opus": 25.0
  },
  "detailed_results": [...]
}
```

### Files Generated
- `results/final_report.json`: Complete evaluation data
- `results/evaluation_summary.txt`: Human-readable summary
- `results/checkpoints/`: Intermediate saves
- `logs/evaluation.log`: Detailed execution log

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure HuggingFace token is set if model is private
2. **API Rate Limits**: Adjust sleep times in config or use checkpoints
3. **Memory Issues**: Reduce batch size or use CPU for teacher model
4. **JSON Parse Errors**: Check Gemini response format in logs

### Debugging

Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Expectations

- **Processing Time**: ~2-3 minutes per example
- **Total Runtime**: 3-5 hours for 100 examples
- **Memory Usage**: ~16GB GPU RAM for teacher model
- **API Costs**: Varies by provider and token usage

## Contributing

To modify the evaluation:

1. **Add New Models**: Update `models.py` with new generation functions
2. **Change Judge Criteria**: Modify prompt in `judge.py`
3. **Adjust Step Processing**: Update `generate_contract_stepwise()` in `models.py`

## License

[Your License Here]

## Citation

If you use this framework, please cite:
```
[Your Citation Here]
```