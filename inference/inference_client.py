"""Client for interacting with the inference server."""

import requests
import logging
import time
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class InferenceClient:
    """Client for the GPU inference server."""
    
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json'}
        
        # Only add API key header if provided
        if api_key:
            self.headers['X-API-Key'] = api_key
        
    def check_health(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_teacher_instructions(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate teacher instructions."""
        
        logger.info(f"CLIENT: Preparing teacher generation request")
        logger.info(f"CLIENT: URL: {self.base_url}/teacher/generate")
        logger.info(f"CLIENT: Max tokens: {max_tokens}, Temp: {temperature}, Top-p: {top_p}")
        logger.info(f"CLIENT: Timeout: {self.timeout} seconds")
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            logger.info(f"CLIENT: Sending POST request to teacher endpoint...")
            response = requests.post(
                f"{self.base_url}/teacher/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            logger.info(f"CLIENT: Response received! Status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"CLIENT: Got {result.get('num_steps', 0)} steps from teacher")
            return result['steps']
            
        except requests.exceptions.Timeout as e:
            logger.error(f"CLIENT: Request timed out after {self.timeout} seconds!")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"CLIENT: Teacher generation request failed: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_student_contract(
        self,
        prompt: str,
        teacher_steps: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> str:
        """Generate student contract with teacher guidance."""
        
        payload = {
            "prompt": prompt,
            "teacher_steps": teacher_steps,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/student/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result['contract']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Student generation request failed: {e}")
            raise
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        try:
            response = requests.get(
                f"{self.base_url}/models/info",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get models info: {e}")
            return {"error": str(e)}
    
    def reload_models(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Reload models on the server."""
        try:
            params = {}
            if model_type:
                params['model_type'] = model_type
                
            response = requests.post(
                f"{self.base_url}/models/reload",
                params=params,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to reload models: {e}")
            return {"error": str(e)}

class InferenceAPIWrapper:
    """Wrapper that provides the same interface as local models but uses the API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        inference_config = config.get('inference_api', {})
        
        self.client = InferenceClient(
            base_url=inference_config.get('endpoint', 'http://localhost:8000'),
            api_key=inference_config.get('api_key', ''),
            timeout=inference_config.get('timeout', 120)
        )
        
        # Check server health on initialization
        health = self.client.check_health()
        if health['status'] != 'healthy':
            logger.warning(f"Inference server not healthy: {health}")
    
    def generate_teacher_instructions(self, prompt: str) -> List[str]:
        """Generate teacher instructions via API."""
        teacher_config = self.config['models']['teacher']
        
        return self.client.generate_teacher_instructions(
            prompt=prompt,
            max_tokens=teacher_config.get('max_new_tokens', 2048),
            temperature=teacher_config.get('temperature', 0.7),
            top_p=teacher_config.get('top_p', 0.9)
        )
    
    def generate_student_contract(self, prompt: str, teacher_steps: List[str]) -> str:
        """Generate student contract via API."""
        student_config = self.config['models']['student']
        
        return self.client.generate_student_contract(
            prompt=prompt,
            teacher_steps=teacher_steps,
            max_tokens=student_config.get('max_new_tokens', 1000),
            temperature=student_config.get('temperature', 0.3),
            top_p=student_config.get('top_p', 0.9)
        )