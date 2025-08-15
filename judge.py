"""Judge module using Gemini 2.5 Pro for evaluation."""

import google.generativeai as genai
import json
import logging
from typing import Dict, Any
from utils import api_call_with_retry

logger = logging.getLogger(__name__)

class GeminiJudge:
    """Gemini 2.5 Pro judge for evaluating contract generations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['models']['judge']
        genai.configure(api_key=config['api_keys']['google'])
        self.model = genai.GenerativeModel(self.config['model'])
        
    def evaluate_contracts(
        self,
        prompt: str,
        thinking_trace: str,  # Correct field name
        teacher_student_contract: str,
        gpt5_contract: str,
        claude_contract: str
    ) -> Dict[str, Any]:
        """Evaluate all contract generations and determine winner."""
        
        judge_prompt = f"""You are an expert legal contract evaluator. Your task is to compare three contract generations against a golden standard and determine which one is best.

ORIGINAL REQUEST:
{prompt}

GOLDEN STANDARD (Contains both expert thinking process and ideal contract):
{thinking_trace}

SUBMISSIONS TO EVALUATE:

=== SUBMISSION A: Teacher-Student Approach ===
{teacher_student_contract}

=== SUBMISSION B: GPT-5 Direct Generation ===
{gpt5_contract}

=== SUBMISSION C: Claude Opus Direct Generation ===
{claude_contract}

EVALUATION INSTRUCTIONS:
1. The golden standard contains both the thinking process and the ideal contract structure
2. Evaluate each submission based on how well it captures the intent and quality shown in the golden standard
3. Consider these criteria:
   - Legal completeness and accuracy
   - Alignment with the requirements in the original request
   - Professional structure and formatting
   - Clarity and precision of language
   - Coverage of important clauses and provisions
   - Risk mitigation and protection for relevant parties

IMPORTANT: You must choose exactly ONE winner. Even if submissions are similar in quality, pick the one that best matches the golden standard.

Provide your evaluation in the following JSON format:
{{
    "analysis": {{
        "submission_a_strengths": "Brief description of Teacher-Student contract strengths",
        "submission_a_weaknesses": "Brief description of Teacher-Student contract weaknesses",
        "submission_b_strengths": "Brief description of GPT-5 contract strengths",
        "submission_b_weaknesses": "Brief description of GPT-5 contract weaknesses",
        "submission_c_strengths": "Brief description of Claude contract strengths",
        "submission_c_weaknesses": "Brief description of Claude contract weaknesses"
    }},
    "scores": {{
        "submission_a": 85,
        "submission_b": 82,
        "submission_c": 88
    }},
    "winner": "A",
    "winner_model": "teacher_student",
    "reasoning": "Brief explanation of why this submission won"
}}

The winner field must be exactly one of: "A", "B", or "C"
The winner_model field must be exactly one of: "teacher_student", "gpt5", "claude_opus"
"""
        
        try:
            # Generate evaluation
            response = api_call_with_retry(
                self.model.generate_content,
                judge_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.config['temperature'],
                    response_mime_type="application/json"
                )
            )
            
            # Parse the JSON response
            result_text = response.text.strip()
            
            # Clean up the response if needed
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            evaluation = json.loads(result_text)
            
            # Map winner letter to model name (backup if winner_model not provided)
            if 'winner_model' not in evaluation:
                winner_map = {
                    'A': 'teacher_student',
                    'B': 'gpt5',
                    'C': 'claude_opus'
                }
                evaluation['winner_model'] = winner_map.get(evaluation['winner'], 'unknown')
            
            logger.info(f"Judge selected winner: {evaluation['winner_model']}")
            
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            logger.error(f"Response was: {response.text if 'response' in locals() else 'No response'}")
            
            # Fallback evaluation
            return {
                "winner": "A",
                "winner_model": "teacher_student",
                "reasoning": "Failed to parse judge response, defaulting to teacher_student",
                "error": str(e)
            }
            
        except Exception as e:
            logger.error(f"Error in judge evaluation: {e}")
            
            # Fallback evaluation
            return {
                "winner": "A",
                "winner_model": "teacher_student",
                "reasoning": f"Judge error: {str(e)}, defaulting to teacher_student",
                "error": str(e)
            }