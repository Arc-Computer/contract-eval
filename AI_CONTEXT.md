# AI Assistant Context - Contract Evaluation System

## System Overview
This is a contract generation evaluation framework that compares a teacher-student approach against state-of-the-art models (GPT-5 and Claude Opus). The system evaluates whether a fine-tuned teacher model can guide a smaller student model to generate contracts that match or exceed the quality of direct generation from SOTA models.

## Architecture Components

### 1. **Inference Server** (GPU Server - 2xH100)
- **Location**: `inference/inference_server.py`
- **Purpose**: Runs teacher (et-8b) and student (Qwen3-8B) models on GPU
- **Endpoints**:
  - `POST /teacher/generate` - Generates step-by-step instructions
  - `POST /student/generate` - Generates contracts following teacher steps
  - `GET /health` - Server health check
  - `GET /models/info` - Loaded models information
  - `POST /models/reload` - Reload models if crashed

**Common Issues & Fixes**:
- **GPU OOM**: Check `inference/server_config.yaml`, reduce `max_memory` or enable `load_in_8bit`
- **Model not loading**: Verify HuggingFace credentials, check `model_cache/` directory
- **API timeout**: Increase `timeout` in `inference_api` section of `config.yaml`

### 2. **Evaluation Pipeline**
- **Main Script**: `evaluate.py`
- **Flow**:
  1. Teacher generates instructions (NO contract, only planning)
  2. Student follows steps to build contract progressively
  3. Baselines (GPT-5, Claude) generate directly
  4. Gemini judge compares all outputs against golden standard

### 3. **Dataset**
- **Source**: `Jarrodbarnes/rcl-specialized-teacher-enterprise`
- **Fields**: 
  - `prompt`: Contract generation request
  - `thinking_trace`: Expert thinking + ideal contract (golden standard)
- **Size**: 599 examples total, evaluation uses first 100

## Critical Files & Their Purposes

```
contract-eval/
├── inference/
│   ├── inference_server.py      # FastAPI server, runs on GPU
│   ├── inference_client.py      # Client wrapper for API calls
│   ├── model_manager.py         # Handles GPU memory, model loading
│   └── server_config.yaml       # Server settings (ports, API keys, memory)
├── models.py                     # API wrappers for teacher/student/baselines
├── evaluate.py                   # Main evaluation orchestrator
├── judge.py                      # Gemini 2.5 Pro evaluation logic
├── dataset_loader.py             # Loads and validates dataset
├── config.yaml                   # Main config (API keys, endpoints)
├── results/                      # Output directory
│   ├── checkpoints/              # Intermediate saves every 10 examples
│   ├── final_report.json        # Complete evaluation results
│   └── contracts/                # Individual contract outputs (JSON)
└── logs/                         # Execution logs
```

## API Keys Required
- **OpenAI**: For GPT-5 baseline (currently using GPT-4o)
- **Anthropic**: For Claude baseline (currently using Claude 3.5 Sonnet)
- **Google**: For Gemini judge
- **Together**: For Qwen3-8B if not using local inference

## How to Start/Debug

### Starting the System:
```bash
# 1. On GPU server:
cd inference
python inference_server.py --host 0.0.0.0 --port 8000

# 2. On evaluation machine:
python evaluate.py --test  # Test with 5 examples
python evaluate.py         # Full 100 examples
```

### Common Debugging Commands:
```bash
# Check server health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Test teacher endpoint
curl -X POST http://localhost:8000/teacher/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a software license agreement"}'

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f logs/evaluation.log
tail -f inference/inference_server.log
```

## Troubleshooting Guide

### Issue: "Model not loaded" error
```python
# In inference_server.py, check:
# 1. HF_TOKEN environment variable is set
# 2. model_cache/ has sufficient space
# 3. Network can reach HuggingFace

# Fix: Manually download models
from huggingface_hub import snapshot_download
snapshot_download("aman-jaglan/et-8b", cache_dir="./model_cache")
```

### Issue: "API key invalid" 
```bash
# Check .env file has all keys:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
TOGETHER_API_KEY=...

# Or set in config.yaml directly
```

### Issue: "CUDA out of memory"
```python
# In server_config.yaml, adjust:
models:
  teacher:
    load_in_8bit: true  # Enable 8-bit quantization
    max_memory: {0: "35GB", 1: "35GB"}  # Reduce per-GPU allocation
```

### Issue: "Connection refused to inference server"
```bash
# Check firewall allows port 8000
# Verify server is running: ps aux | grep inference_server
# Check correct endpoint in config.yaml
inference_api:
  endpoint: "http://YOUR_GPU_SERVER_IP:8000"  # Update this
```

### Issue: "Judge evaluation failing"
```python
# Common causes:
# 1. Gemini API quota exceeded - wait or use different API key
# 2. Response not JSON - check judge.py line 85-95 for parsing
# 3. Contract too long - Gemini has token limits
```

## Model Details

### Teacher Model (et-8b)
- **Location**: HuggingFace `aman-jaglan/et-8b`
- **Checkpoints**: 2 distributed files that get merged
- **Purpose**: Generate strategic planning steps WITHOUT writing contracts
- **Context**: 32k tokens

### Student Model (Qwen3-8B)
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Purpose**: Follow teacher's steps to build contracts progressively
- **Context**: 32k tokens

## Evaluation Metrics
- **Primary**: Win rate percentage (teacher-student vs baselines)
- **Judge Criteria**:
  - Legal completeness
  - Alignment with requirements
  - Professional structure
  - Risk coverage
  - Clarity of language

## Output Files
- `results/final_report.json`: Complete evaluation with win rates
- `results/evaluation_summary.txt`: Human-readable summary
- `results/contracts/example_XXX.json`: Individual contract outputs
- `results/checkpoints/`: Recovery points if interrupted

## Recovery from Interruption
```bash
# Find latest checkpoint
ls -la results/checkpoints/

# Resume from checkpoint
python evaluate.py --resume results/checkpoints/checkpoint_050.json
```

## Performance Expectations
- **Per example**: 2-3 minutes
- **100 examples**: 3-5 hours total
- **GPU Memory**: ~32GB per model (64GB total)
- **API costs**: Varies by provider usage

## Emergency Fixes

### Reset everything:
```bash
rm -rf model_cache/ results/ logs/
mkdir -p results/checkpoints results/contracts logs
python setup.py
```

### Force reload models:
```python
import requests
requests.post("http://localhost:8000/models/reload", 
              headers={"X-API-Key": "your-key"})
```

### Kill stuck process:
```bash
ps aux | grep inference_server
kill -9 [PID]
```

## Contact for Issues
If the system is fundamentally broken:
1. Check this document first
2. Review logs in `logs/` directory
3. Verify all API keys are valid
4. Ensure GPU server has internet access for model downloads