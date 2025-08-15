"""FastAPI inference server for teacher and student models using vLLM."""

import os
import sys
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import extract_steps_from_thinking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager
model_manager = None
config = None

# Request/Response models
class TeacherRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class TeacherResponse(BaseModel):
    steps: List[str]
    num_steps: int
    processing_time: float
    
class StudentRequest(BaseModel):
    prompt: str
    teacher_steps: List[str]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9

class StudentResponse(BaseModel):
    contract: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    gpu_memory: Dict[int, Dict[str, float]]
    uptime_seconds: float

# Global variables for vLLM models
teacher_llm: Optional[LLM] = None
student_llm: Optional[LLM] = None
config = None

# Server startup time
SERVER_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage vLLM model loading on startup and cleanup on shutdown."""
    global teacher_llm, student_llm, config
    
    # Load configuration
    config_path = os.getenv('SERVER_CONFIG', 'server_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize vLLM models
    logger.info("Initializing vLLM models with 90% GPU utilization...")
    
    try:
        # Load BOTH models with shared GPU memory allocation
        # Each model gets 45% of total GPU memory (90% total utilization)
        logger.info("Loading teacher model with vLLM (45% GPU memory per model)...")
        teacher_llm = LLM(
            model="Arc-Intelligence/arc-teacher-8b",
            tensor_parallel_size=2,  # Use both GPUs for speed
            gpu_memory_utilization=0.45,  # 45% for teacher model
            max_model_len=32768,
            dtype="half",  # float16
            enforce_eager=False,  # Enable CUDA graphs for speed
            trust_remote_code=True,
            download_dir="./model_cache"
        )
        logger.info("Teacher model loaded successfully with vLLM")
        
        # Load student model with vLLM 
        logger.info("Loading student model with vLLM (45% GPU memory per model)...")
        student_llm = LLM(
            model="Qwen/Qwen3-8B",
            tensor_parallel_size=2,  # Use both GPUs for speed
            gpu_memory_utilization=0.45,  # 45% for student model
            max_model_len=32768,
            dtype="half",
            enforce_eager=False,
            trust_remote_code=True,
            download_dir="./model_cache"
        )
        
        logger.info("Student model loaded successfully with vLLM")
        logger.info("All models loaded with vLLM - ready for fast inference!")
        
    except Exception as e:
        logger.error(f"Failed to load vLLM models: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down vLLM server...")
    # vLLM handles cleanup automatically

# Create FastAPI app
app = FastAPI(
    title="Contract Generation Inference Server",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware - simplified for local use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """Verify API key for authentication."""
    if not config or not config.get('server', {}).get('enable_auth', False):
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    if x_api_key != config['server']['api_key']:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True

@app.post("/test")
def test_post():
    """Simple test endpoint."""
    logger.info("Test POST endpoint hit!")
    return {"message": "POST request received", "timestamp": time.time()}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and status."""
    # Get GPU memory info
    gpu_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory[i] = {
                "allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                "total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3
            }
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "teacher": teacher_llm is not None,
            "student": student_llm is not None
        },
        gpu_memory=gpu_memory,
        uptime_seconds=time.time() - SERVER_START_TIME
    )

@app.get("/models/info")
async def get_models_info():
    """Get detailed information about loaded models."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return model_manager.get_model_info()

@app.post("/teacher/generate", response_model=TeacherResponse)
def generate_teacher_instructions(
    request: TeacherRequest
):
    """Generate step-by-step instructions using vLLM teacher model."""
    start_time = time.time()
    
    logger.info(f"=== vLLM TEACHER GENERATION REQUEST ===")
    logger.info(f"Prompt: {request.prompt[:100]}...")
    logger.info(f"Max tokens: {request.max_tokens}")
    logger.info(f"Temperature: {request.temperature}")
    logger.info(f"Top-p: {request.top_p}")
    
    if not teacher_llm:
        logger.error("Teacher vLLM model not initialized!")
        raise HTTPException(status_code=503, detail="Teacher model not initialized")
    
    try:
        
        # Create the optimized teacher prompt
        teacher_prompt = f"""You are an expert legal and business consultant specializing in contract generation and business documentation. Your role is to provide detailed, step-by-step strategic guidance for creating professional documents.

Given the following request, generate ONLY strategic thinking and planning steps. DO NOT write the actual contract or document itself.

Your instructions must:
1. Break down the task into clear, numbered steps (Step 1, Step 2, etc.)
2. For each step, explain:
   - What needs to be considered
   - Key legal/business implications
   - Risk factors to address
   - Essential elements to include
3. Provide strategic reasoning for each decision
4. Identify potential challenges and mitigation strategies
5. Suggest structure and organization approaches

Request: {request.prompt}

Generate comprehensive step-by-step planning instructions:
Step 1:"""
        
        # vLLM sampling parameters - optimized for speed
        gen_config = config['generation']['teacher']
        sampling_params = SamplingParams(
            temperature=request.temperature or gen_config['temperature'],
            top_p=request.top_p or gen_config['top_p'],
            max_tokens=request.max_tokens or gen_config['max_new_tokens'],
            presence_penalty=0.1,  # Prevent repetition
            frequency_penalty=0.1,
            stop=["<|endoftext|>", "<|assistant|>"]
        )
        
        logger.info(f"Starting vLLM generation with max_tokens={sampling_params.max_tokens}")
        logger.info(f"Sampling params: temp={sampling_params.temperature}, top_p={sampling_params.top_p}")
        
        # Generate with vLLM - MUCH FASTER!
        outputs = teacher_llm.generate([teacher_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        logger.info(f"vLLM generation complete! Generated {len(generated_text)} characters")
        
        # Extract instructions
        instructions_text = generated_text.split("<|assistant|>")[-1].strip()
        steps = extract_steps_from_thinking(instructions_text)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Teacher generated {len(steps)} steps in {processing_time:.2f}s")
        
        return TeacherResponse(
            steps=steps,
            num_steps=len(steps),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Teacher generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/student/generate", response_model=StudentResponse)
def generate_student_contract(
    request: StudentRequest
):
    """Generate contract using vLLM student model with step-by-step generation."""
    start_time = time.time()
    
    logger.info(f"=== vLLM STUDENT GENERATION REQUEST ===")
    logger.info(f"Prompt: {request.prompt[:100]}...")
    logger.info(f"Number of teacher steps: {len(request.teacher_steps)}")
    logger.info(f"Max tokens: {request.max_tokens}")
    logger.info(f"Temperature: {request.temperature}")
    
    if not student_llm:
        logger.error("Student vLLM model not initialized!")
        raise HTTPException(status_code=503, detail="Student model not initialized")
    
    try:
        # Generate contract step by step - SAME LOGIC, just using vLLM for speed
        contract_parts = []
        
        # Initial context for the student
        initial_context = f"""You are a professional contract writer and business document specialist. You will receive step-by-step instructions from an expert consultant, and your task is to implement each step to build a complete, professional document.

Original Request: {request.prompt}

You will now receive instructions step by step. For each step, generate the corresponding section of the document."""
        
        accumulated_content = ""
        gen_config = config['generation']['student']
        
        # vLLM sampling parameters - optimized for speed
        sampling_params = SamplingParams(
            temperature=request.temperature or gen_config['temperature'],
            top_p=request.top_p or gen_config['top_p'],
            max_tokens=request.max_tokens or gen_config['max_new_tokens'],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=["<|endoftext|>", "<|assistant|>"]
        )
        
        # Process each step sequentially - SAME AS BEFORE, each builds on previous
        for i, step in enumerate(request.teacher_steps, 1):
            logger.info(f"Processing step {i}/{len(request.teacher_steps)}: {step[:100]}...")
            
            # Build progressive prompt with ACTUAL accumulated content from previous steps
            step_prompt = f"""{initial_context}

Current Progress:
{accumulated_content if accumulated_content else "[Beginning of document]"}

Expert Instruction - Step {i} of {len(request.teacher_steps)}:
{step}

Based on this instruction and the context above, generate the appropriate content for this step:
- If Step 1: Begin with document header and initial structure
- If middle step: Continue building on previous sections
- If final step: Complete and conclude the document properly

Output for Step {i}:"""
            
            logger.info(f"Step {i} - Prompt length: {len(step_prompt)} chars")
            logger.info(f"Step {i} - Starting vLLM generation with max_tokens={sampling_params.max_tokens}")
            
            # Generate with vLLM - MUCH FASTER than transformers!
            outputs = student_llm.generate([step_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"Step {i} - vLLM generated {len(generated_text)} characters")
            
            # Extract the contract part (clean output)
            contract_part = generated_text.strip()
            
            # Remove any artifacts if present
            if f"Output for Step {i}:" in contract_part:
                contract_part = contract_part.split(f"Output for Step {i}:")[-1].strip()
            
            contract_parts.append(contract_part)
            
            # Update accumulated content with ACTUAL output for next step
            # This is CRITICAL - next step must see what was actually generated
            accumulated_content += f"\n[Step {i} Output]:\n{contract_part}\n"
            
            logger.info(f"Completed step {i}/{len(request.teacher_steps)}")
        
        # Combine all parts
        final_contract = "\n\n".join(contract_parts)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Student generated contract in {processing_time:.2f}s")
        logger.info(f"Total contract length: {len(final_contract)} characters")
        
        return StudentResponse(
            contract=final_contract,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Student generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/reload")
async def reload_models(
    model_type: Optional[str] = None
):
    """Reload models (useful for switching or recovering from errors)."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        if model_type == "teacher":
            model_manager.unload_model("teacher")
            model_manager.load_teacher_model()
            return {"message": "Teacher model reloaded"}
        elif model_type == "student":
            model_manager.unload_model("student")
            model_manager.load_student_model()
            return {"message": "Student model reloaded"}
        else:
            model_manager.unload_model("teacher")
            model_manager.unload_model("student")
            model_manager.load_all_models()
            return {"message": "All models reloaded"}
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference server for contract generation")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="server_config.yaml", help="Config file path")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set config path in environment
    os.environ['SERVER_CONFIG'] = args.config
    
    logger.info(f"Starting inference server on {args.host}:{args.port}")
    
    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()