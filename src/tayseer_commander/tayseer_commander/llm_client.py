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

    # Native tool schema — the model is FORCED to call this function
    TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "submit_robot_plan",
            "description": (
                "Submit a structured robot plan or clarification. "
                "ALWAYS use this tool to respond. Never put JSON in the message content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["plan", "clarify", "denied"],
                        "description": "Response mode"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of your decision"
                    },
                    "plan": {
                        "type": "array",
                        "description": "Required when mode is 'plan'. Empty for other modes.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["navigate_to", "pick", "place", "slide"],
                                    "description": "Action name"
                                },
                                "params": {
                                    "type": "object",
                                    "description": (
                                        "navigate_to/pick: {object_name: str}. "
                                        "place: {object_name: str, target_location: str}. "
                                        "slide: {object_name: str, direction: str, distance_meters: float}."
                                    )
                                }
                            },
                            "required": ["action", "params"]
                        }
                    },
                    "question": {
                        "type": "string",
                        "description": "Question for user when mode is 'clarify'"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for denial when mode is 'denied'"
                    },
                    "options": {
                        "type": "array",
                        "description": "Suggested options when mode is 'clarify'",
                        "items": {"type": "string"}
                    }
                },
                "required": ["mode", "reasoning"]
            }
        }
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY env var.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        # NOTE: If function calling is not supported on this model, switch to:
        # "llama-3.3-70b-versatile" or "mixtral-8x7b-32768"
        self.model = "openai/gpt-oss-120b"

        with open(self.PROMPT_PATH, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def generate_response(self, messages: list, world_state: dict) -> dict:
        """
        Calls the LLM with forced tool use. Returns a validated dict.
        """
        world_state_json = json.dumps(world_state, indent=2)
        system_prompt = self.system_prompt.replace("{world_state_json}", world_state_json)
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.1,  # Lower = more deterministic
                tools=[self.TOOL_SCHEMA],
                tool_choice={"type": "function", "function": {"name": "submit_robot_plan"}}
            )

            message = response.choices[0].message

            # Handle model refusal (OpenAI feature, defensive for Groq)
            refusal = getattr(message, 'refusal', None)
            if refusal:
                return {
                    "mode": "clarify",
                    "reasoning": f"Model refused: {refusal}",
                    "question": "I cannot process that request. Can you rephrase?",
                    "error": True
                }

            # Extract tool call arguments
            tool_calls = getattr(message, 'tool_calls', None)
            if not tool_calls or len(tool_calls) == 0:
                # Fallback: try to parse content if model ignored tool instruction
                content = getattr(message, 'content', '') or ''
                try:
                    result = json.loads(content.strip())
                except json.JSONDecodeError:
                    return {
                        "mode": "clarify",
                        "reasoning": "Model did not use the required tool and content was not valid JSON",
                        "question": "I got confused. Can you repeat that?",
                        "error": True
                    }
            else:
                arguments = tool_calls[0].function.arguments
                if isinstance(arguments, str):
                    result = json.loads(arguments)
                else:
                    result = arguments

            # --- Post-processing: normalize common LLM parameter hallucinations ---
            if result.get("mode") == "plan" and isinstance(result.get("plan"), list):
                for step in result["plan"]:
                    if not isinstance(step, dict):
                        continue

                    # Ensure params is a dict, not flat keys
                    if not isinstance(step.get("params"), dict):
                        step["params"] = {k: v for k, v in step.items() if k != "action"}

                    p = step.get("params", {})
                    # Fix common key mismatches
                    if "target" in p and "object_name" not in p:
                        p["object_name"] = p.pop("target")
                    if "object" in p and "object_name" not in p:
                        p["object_name"] = p.pop("object")
                    if "location" in p and "target_location" not in p:
                        p["target_location"] = p.pop("location")
                    step["params"] = p

            return result

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
        """Legacy entrypoint — delegates to generate_response for consistency."""
        messages = [{"role": "user", "content": user_prompt}]
        return self.generate_response(messages, world_state)

    def replan(self, user_prompt: str, world_state: Dict,
               failed_action: Dict, failure_reason: str) -> Dict[str, Any]:
        """Replan after a failed action."""
        prompt_addition = f"""
PREVIOUS PLAN FAILED
Failed action: {json.dumps(failed_action)}
Reason: {failure_reason}
Please generate a new plan considering this failure.
"""
        messages = [{"role": "user", "content": user_prompt + prompt_addition}]
        return self.generate_response(messages, world_state)