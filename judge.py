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
        
        judge_prompt = f"""You are a senior business professional with 20 years of experience reviewing contracts and business documents.

THE REQUEST TO FULFILL:
{prompt}

EXPERT'S IDEAL APPROACH (Golden Standard):
{thinking_trace}

Now evaluate these three attempts to fulfill the same request:

========================================
SUBMISSION A (Teacher-Student Approach):
========================================
{teacher_student_contract}

========================================
SUBMISSION B (GPT-5):
========================================
{gpt5_contract}

========================================
SUBMISSION C (Claude Opus):
========================================
{claude_contract}

========================================
EVALUATION FRAMEWORK:
========================================

AS A BUSINESS PROFESSIONAL, EVALUATE BASED ON:

1. **COMPLETENESS** (50% weight)
   - Does it address EVERY requirement from the original request?
   - Are all dates, technical specs, pricing models, names mentioned in the request included?
   - Would you need to ask for any clarifications before using this document?

2. **BUSINESS VIABILITY** (30% weight)
   - Is this the right type of document for the request?
   - Would this hold up in a real business scenario?
   - Does it protect the relevant parties?
   - Are the terms clear and enforceable?

3. **PROFESSIONAL SUBSTANCE** (20% weight)
   - Does it demonstrate understanding of the business context?
   - Are industry-specific requirements properly addressed?
   - Is it comprehensive enough for the scenario?

EXPLICITLY IGNORE (NO PENALTY):
- Formatting differences (bullets, numbering, spacing)
- Length (a comprehensive 20-page contract can be better than a 2-page summary)
- Minor stylistic choices
- Whether sections are numbered or bulleted

FOCUS ONLY ON:
- Did they deliver what was asked?
- Is anything critical missing?
- Would you sign/approve this in real life?

Note: In real business, thoroughness beats brevity. A complete document is always preferred over a pretty but incomplete one.

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