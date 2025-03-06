import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union
import numpy as np

# transform action to id 
def action_to_id(action: str, action_space: List[str]) -> int:
    """
    Convert action to id
    """
    action_to_id = {action: i for i, action in enumerate(action_space)}
    return action_to_id[action]

# transform id to action
def id_to_action(id: int, action_space: List[str]) -> str:
    """
    Convert id to action
    """
    id_to_action = {i: action for i, action in enumerate(action_space)}
    return id_to_action[id]

# build messages
def build_messages(image: dict, prompt: str) -> list:
    """
    Build the messages for the Qwen2.5-vl-3b model
    """
    messages = []
    content = []

    for camera_name, camera_image in image.items():
        content.append({
            "type": "image",
            "image": camera_image
        })
    content.append({"type": "text", "text": prompt})
    messages.append({
        "role": "user",
        "content": content
    })
    return messages

# build state description
def build_state_description(state: Dict) -> str:
    """
    Build the state description
    """
    state_description = f"Current speed: {state['speed']} m/s\n"
    state_description += f"Current steering: {state['steering']}\n"
    state_description += f"Current lane: {state['lane']}"
    return state_description

# build history description
def build_history_description(history: List[str]) -> str:
    """
    Build the history description
    """
    history_description = "Previous actions:\n"
    for action in history:
        history_description += f"{action}\n"
    return history_description

# build macro action prompt
def build_macro_action_prompt(state_description: str, history_description: str, action_space: List[str], navigation_info: str) -> str:
    """
    Build the prompt
    """
    action_space_str = "\n".join([f"{i}. {action}" for i, action in enumerate(action_space)])
    prompt = f"""Based on the following vehicle state, camera images, previous actions, 
        and navigation info, please analyze the situation and choose the most appropriate action from the action space.
        Consider safety, efficiency, and traffic rules.
        
        Current vehicle state:
        {state_description}

        Previous actions:
        {history_description}
        
        Action space:
        {action_space_str}

        Navigation info:
        {navigation_info}

        Please provide your analysis in the following format:
        Reasoning: [reasoning]
        Most appropriate action: [action]
        """
    return prompt

# build mid action prompt
def build_mid_action_prompt(state_description: str, history_description: str, macro_action: str) -> str:
    """
    Build the prompt
    """
    prompt = f"""Based on the following vehicle state, camera images, previous actions, 
        and the macro action, please analyze the situation and refine the macro action.
        Consider safety, efficiency, and traffic rules.
        
        Current vehicle state:
        {state_description}

        Previous actions:
        {history_description}
        
        Macro action:
        {macro_action}

        Please provide your analysis in the following format:
        Reasoning: [reasoning]
        Refined macro action: [refined macro action]
        """
    return prompt

# parse action from response
def parse_action(response: str) -> str:
    """
    Parse the action from the response
    """
    action = response.split("Most appropriate action: ")[1].strip()
    return action

# parse reasoning from response
def parse_reasoning(response: str) -> str:
    """
    Parse the reasoning from the response
    """
    reasoning = response.split("Reasoning: ")[1].strip()
    return reasoning