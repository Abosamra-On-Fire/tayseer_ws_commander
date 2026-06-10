#!/usr/bin/env python3
from openai import OpenAI
import json
import os
from ament_index_python.packages import get_package_share_directory
from typing import List, Dict, Any


class GroqLLMClient:
    PROMPT_PATH = os.path.join(
        get_package_share_directory('tayseer_commander'),
        'config',
        'system_prompt.txt'
    )
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY env var.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

        with open(self.PROMPT_PATH, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def generate_response(self, messages: list, world_state: dict) -> dict:
        """
        Args:
            messages: list of {"role": "user"|"assistant", "content": str}
            World_state: dict of the objects in the world ({"blue_cube": {"position": [1.2, 3.4, 0.0], ...}, ...})
        Returns: 
            {"mode": "plan"|"clarify", ...}
        """
        world_state_json = json.dumps(world_state, indent=2)
        system_prompt = self.system_prompt.replace("{world_state_json}", world_state_json)
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        
        with open("temp_propt.txt", "w", encoding="utf-8") as f:
            json.dump(api_messages, f, indent=2, ensure_ascii=False)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            text = response.choices[0].message.content.strip()
            
            with open("temp_response.txt", "w", encoding="utf-8") as f:
                f.write(text)
            
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            return json.loads(text.strip())
            
        except json.JSONDecodeError as e:
            return {
                "mode": "clarify",
                "reasoning": f"JSON parse error: {str(e)}",
                "question": "I got confused. Can you repeat that?",
                "error": True
            }
        except Exception as e:
            return {
                "mode": "clarify",
                "reasoning": f"API error: {str(e)}",
                "question": "I'm having trouble connecting. Can you try again?",
                "error": True
            }

    def generate_plan(self, user_prompt: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            user_prompt: str of the order that the uder wants tayseer to preform
            World_state: dict of the objects in the world ({"blue_cube": {"position": [1.2, 3.4, 0.0], ...}, ...})
        Returns: 
            {"mode": "plan"|"clarify", ...}
        """
        # Build the full prompt
        object_list = ", ".join(world_state.keys()) if world_state else "none detected"
        
        prompt = f"""{self.system_prompt.replace("{object_list}", object_list)}
            CURRENT WORLD STATE
            ```json
            {json.dumps(world_state, indent=2)}
            USER REQUEST
            "{user_prompt}"
            Generate the plan:
            """
            
        api_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        with open("temp_propt.txt", "w", encoding="utf-8") as f:
            json.dump(api_messages, f, indent=2, ensure_ascii=False)
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.2,
                response_format={"type": "json_object"}  # Supported on Groq Llama/Mixtral models
            )
            text = response.choices[0].message.content.strip()
            
            with open("temp_response.txt", "w", encoding="utf-8") as f:
                f.write(text)
                
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            plan = json.loads(text.strip())
            return plan
            
        except json.JSONDecodeError as e:
            return {
                "mode": "clarify",
                "reasoning": f"JSON parse error: {str(e)}",
                "question": "I got confused. Can you repeat that?",
                "error": True
            }
        except Exception as e:
            return {
                "mode": "clarify",
                "reasoning": f"API error: {str(e)}",
                "question": "I'm having trouble connecting. Can you try again?",
                "error": True
            }

    def replan(self, user_prompt: str, world_state: Dict, 
            failed_action: Dict, failure_reason: str) -> Dict[str, Any]:
        """Replan after a failed action."""
        prompt_addition = f"""
            PREVIOUS PLAN FAILED
            Failed action: {json.dumps(failed_action)}
            Reason: {failure_reason}
            Please generate a new plan considering this failure.
            """
        return self.generate_plan(user_prompt + prompt_addition, world_state)