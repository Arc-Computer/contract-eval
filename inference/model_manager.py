"""Model manager for GPU inference server."""

import torch
import gc
import logging
import psutil
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig
)
from huggingface_hub import hf_hub_download
import yaml

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading and GPU memory for inference server."""
    
    def __init__(self, config_path: str = "server_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None
        
        # Track GPU memory
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"Found {self.gpu_count} GPUs")
        
    def get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """Get current GPU memory usage."""
        info = {}
        for i in range(self.gpu_count):
            torch.cuda.synchronize(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info[i] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated
            }
        return info
    
    def clear_gpu_cache(self):
        """Clear GPU cache."""
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_teacher_model(self) -> Tuple[Any, Any]:
        """Load the teacher model - using Arc-Intelligence model directly."""
        if self.teacher_model is not None:
            logger.info("Teacher model already loaded")
            return self.teacher_model, self.teacher_tokenizer
            
        logger.info("Loading teacher model...")
        start_time = time.time()
        
        try:
            config_dict = self.config['models']['teacher']
            
            # Use Arc-Intelligence teacher model directly
            logger.info("Loading Arc-Intelligence teacher model directly from HuggingFace...")
            model_id = "Arc-Intelligence/arc-teacher-8b"
            
            # Load tokenizer
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            
            # Load model with appropriate settings
            device_map = config_dict.get('device_map', 'auto')
            max_memory = config_dict.get('max_memory', None)
            
            logger.info(f"Loading model with device_map={device_map}")
            
            # Load the model directly from HuggingFace
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            
            self.teacher_model.eval()
            self.clear_gpu_cache()
            
            load_time = time.time() - start_time
            logger.info(f"Teacher model loaded in {load_time:.2f} seconds")
            
            # Log memory usage
            memory_info = self.get_gpu_memory_info()
            for gpu_id, info in memory_info.items():
                logger.info(f"GPU {gpu_id}: {info['allocated_gb']:.2f}/{info['total_gb']:.2f} GB used")
            
            return self.teacher_model, self.teacher_tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
    
    def load_student_model(self) -> Tuple[Any, Any]:
        """Load the student model (Qwen3-8B)."""
        if self.student_model is not None:
            logger.info("Student model already loaded")
            return self.student_model, self.student_tokenizer
            
        logger.info("Loading student model...")
        start_time = time.time()
        
        try:
            config_dict = self.config['models']['student']
            
            # Load tokenizer
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                config_dict['repo_id'],
                trust_remote_code=config_dict.get('trust_remote_code', True)
            )
            
            if self.student_tokenizer.pad_token is None:
                self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
            
            # Prepare model kwargs
            model_kwargs = {
                'trust_remote_code': config_dict.get('trust_remote_code', True),
                'torch_dtype': torch.float16,
                'device_map': config_dict.get('device_map', 'auto')
            }
            
            # Add 8-bit quantization if specified
            if config_dict.get('load_in_8bit', False):
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            
            # Load model
            self.student_model = AutoModelForCausalLM.from_pretrained(
                config_dict['repo_id'],
                **model_kwargs
            )
            
            self.student_model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Student model loaded in {load_time:.2f} seconds")
            
            # Log memory usage
            memory_info = self.get_gpu_memory_info()
            for gpu_id, info in memory_info.items():
                logger.info(f"GPU {gpu_id}: {info['allocated_gb']:.2f}/{info['total_gb']:.2f} GB used")
            
            return self.student_model, self.student_tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load student model: {e}")
            raise
    
    def load_all_models(self):
        """Load both teacher and student models."""
        logger.info("Loading all models...")
        
        # Load teacher first
        self.load_teacher_model()
        
        # Clear cache before loading student
        self.clear_gpu_cache()
        
        # Load student
        self.load_student_model()
        
        logger.info("All models loaded successfully")
        
        # Final memory report
        memory_info = self.get_gpu_memory_info()
        logger.info("Final GPU memory usage:")
        for gpu_id, info in memory_info.items():
            logger.info(f"  GPU {gpu_id}: {info['allocated_gb']:.2f}/{info['total_gb']:.2f} GB ({(info['allocated_gb']/info['total_gb']*100):.1f}%)")
    
    def unload_model(self, model_type: str):
        """Unload a specific model to free memory."""
        if model_type == "teacher":
            if self.teacher_model is not None:
                del self.teacher_model
                self.teacher_model = None
                del self.teacher_tokenizer
                self.teacher_tokenizer = None
                logger.info("Teacher model unloaded")
        elif model_type == "student":
            if self.student_model is not None:
                del self.student_model
                self.student_model = None
                del self.student_tokenizer
                self.student_tokenizer = None
                logger.info("Student model unloaded")
        
        self.clear_gpu_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'teacher_loaded': self.teacher_model is not None,
            'student_loaded': self.student_model is not None,
            'gpu_count': self.gpu_count,
            'gpu_memory': self.get_gpu_memory_info()
        }
        
        if self.teacher_model is not None:
            info['teacher_params'] = sum(p.numel() for p in self.teacher_model.parameters()) / 1e9
        
        if self.student_model is not None:
            info['student_params'] = sum(p.numel() for p in self.student_model.parameters()) / 1e9
        
        return info