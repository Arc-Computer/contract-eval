"""Model wrappers for API-based inference."""

import sys
import logging
from typing import Dict, Any, List
from pathlib import Path
from openai import OpenAI
import openai
import anthropic
import google.generativeai as genai

# Add inference directory to path
sys.path.append(str(Path(__file__).parent / 'inference'))

from inference.inference_client import InferenceAPIWrapper
from utils import api_call_with_retry

logger = logging.getLogger(__name__)

class TeacherModel:
    """Teacher model using inference API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Using API inference for teacher model")
        self.api_wrapper = InferenceAPIWrapper(config)
    
    def generate_instructions(self, prompt: str) -> List[str]:
        """Generate step-by-step instructions via API."""
        return self.api_wrapper.generate_teacher_instructions(prompt)

class StudentModel:
    """Student model using inference API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Using API inference for student model")
        self.api_wrapper = InferenceAPIWrapper(config)
    
    def generate_contract_stepwise(self, prompt: str, teacher_steps: List[str]) -> str:
        """Generate contract via API following teacher's steps."""
        return self.api_wrapper.generate_student_contract(prompt, teacher_steps)

class BaselineModels:
    """Baseline models for direct contract generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize API clients
        self.openai_client = openai.OpenAI(api_key=config['api_keys']['openai'])
        self.anthropic_client = anthropic.Anthropic(api_key=config['api_keys']['anthropic'])
        genai.configure(api_key=config['api_keys']['google'])
        
    def generate_gpt5(self, prompt: str) -> str:
        """Generate contract using GPT-5."""
        model_name = self.config['models']['baselines']['gpt5']['model']
        
        try:
            response = api_call_with_retry(
                self.openai_client.chat.completions.create,
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert legal contract writer. Generate professional, comprehensive contracts."},
                    {"role": "user", "content": f"Generate a complete legal contract for: {prompt}"}
                ],
                max_tokens=self.config['models']['baselines']['gpt5']['max_tokens'],
                temperature=self.config['models']['baselines']['gpt5']['temperature']
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error with GPT-5 generation: {e}")
            return f"[Error: {str(e)}]"
    
    def generate_claude(self, prompt: str) -> str:
        """Generate contract using Claude Opus 4.1."""
        model_name = self.config['models']['baselines']['claude']['model']
        
        try:
            response = api_call_with_retry(
                self.anthropic_client.messages.create,
                model=model_name,
                messages=[
                    {"role": "user", "content": f"Generate a complete legal contract for: {prompt}"}
                ],
                max_tokens=self.config['models']['baselines']['claude']['max_tokens'],
                temperature=self.config['models']['baselines']['claude']['temperature'],
                system="You are an expert legal contract writer. Generate professional, comprehensive contracts."
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error with Claude generation: {e}")
            return f"[Error: {str(e)}]"