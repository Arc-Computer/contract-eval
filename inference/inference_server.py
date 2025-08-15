"""FastAPI inference server for teacher and student models."""

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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model_manager import ModelManager
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

# Server startup time
SERVER_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading on startup and cleanup on shutdown."""
    global model_manager, config
    
    # Load configuration
    config_path = os.getenv('SERVER_CONFIG', 'server_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model manager
    logger.info("Initializing model manager...")
    model_manager = ModelManager(config_path)
    
    # Load models
    logger.info("Loading models on startup...")
    try:
        model_manager.load_all_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Continue anyway - models can be loaded on demand
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down server...")
    if model_manager:
        model_manager.clear_gpu_cache()

# Create FastAPI app
app = FastAPI(
    title="Contract Generation Inference Server",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['server']['cors_origins'] if config else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """Verify API key for authentication."""
    if not config['server'].get('enable_auth', True):
        return True
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    if x_api_key != config['server']['api_key']:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and status."""
    model_info = model_manager.get_model_info() if model_manager else {}
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "teacher": model_info.get('teacher_loaded', False),
            "student": model_info.get('student_loaded', False)
        },
        gpu_memory=model_info.get('gpu_memory', {}),
        uptime_seconds=time.time() - SERVER_START_TIME
    )

@app.get("/models/info")
async def get_models_info(authorized: bool = Depends(verify_api_key)):
    """Get detailed information about loaded models."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return model_manager.get_model_info()

@app.post("/teacher/generate", response_model=TeacherResponse)
async def generate_teacher_instructions(
    request: TeacherRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Generate step-by-step instructions using the teacher model."""
    start_time = time.time()
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        # Load teacher model if not already loaded
        teacher_model, teacher_tokenizer = model_manager.load_teacher_model()
        
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
        
        # Tokenize with full context window
        inputs = teacher_tokenizer(
            teacher_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=32768  # Full 32k context window
        )
        
        # Move to same device as model
        if hasattr(teacher_model, 'device'):
            inputs = {k: v.to(teacher_model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        gen_config = config['generation']['teacher']
        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or gen_config['max_new_tokens'],
                temperature=request.temperature or gen_config['temperature'],
                top_p=request.top_p or gen_config['top_p'],
                do_sample=gen_config['do_sample'],
                pad_token_id=teacher_tokenizer.pad_token_id,
                eos_token_id=teacher_tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
async def generate_student_contract(
    request: StudentRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Generate contract using the student model with teacher's guidance."""
    start_time = time.time()
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    try:
        # Load student model if not already loaded
        student_model, student_tokenizer = model_manager.load_student_model()
        
        # Generate contract step by step with optimized prompts
        contract_parts = []
        
        # Initial context for the student
        initial_context = f"""You are a professional contract writer and business document specialist. You will receive step-by-step instructions from an expert consultant, and your task is to implement each step to build a complete, professional document.

Original Request: {request.prompt}

You will now receive instructions step by step. For each step, generate the corresponding section of the document."""
        
        accumulated_content = ""
        gen_config = config['generation']['student']
        
        for i, step in enumerate(request.teacher_steps, 1):
            # Build progressive prompt with accumulated content
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
            
            # Tokenize with full context (32k window support)
            inputs = student_tokenizer(
                step_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=32768  # Full 32k context window
            )
            
            # Move to device
            if hasattr(student_model, 'device'):
                inputs = {k: v.to(student_model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate this part
            with torch.no_grad():
                outputs = student_model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or gen_config['max_new_tokens'],
                    temperature=request.temperature or gen_config['temperature'],
                    top_p=request.top_p or gen_config['top_p'],
                    do_sample=gen_config['do_sample'],
                    pad_token_id=student_tokenizer.pad_token_id,
                    eos_token_id=student_tokenizer.eos_token_id
                )
            
            # Decode
            generated = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove the prompt)
            if step_prompt in generated:
                contract_part = generated.split(step_prompt)[-1].strip()
            else:
                # Fallback: take everything after the last occurrence of "Step"
                contract_part = generated.split(f"Contract Part for Step {i}:")[-1].strip()
            
            contract_parts.append(contract_part)
            
            # Update accumulated content with FULL output (no truncation with 32k context)
            accumulated_content += f"\n[Step {i} Output]:\n{contract_part}\n"
            
            logger.info(f"Completed step {i}/{len(request.teacher_steps)}")
        
        # Combine all parts
        final_contract = "\n\n".join(contract_parts)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Student generated contract in {processing_time:.2f}s")
        
        return StudentResponse(
            contract=final_contract,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Student generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/reload")
async def reload_models(
    model_type: Optional[str] = None,
    authorized: bool = Depends(verify_api_key)
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