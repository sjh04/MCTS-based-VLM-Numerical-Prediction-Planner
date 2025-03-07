import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Union, Tuple
import numpy as np
from PIL import Image
from .utils import *
from qwen_vl_utils import process_vision_info

# high level policy generator
class HighLevelPolicyGenerator:
    def __init__(self, state: Dict, image: dict, history: List[str], navigation_info: str, model_name: str = "Qwen/Qwen2.5-VL-3B"
                 , device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 model: Qwen2_5_VLForConditionalGeneration = None,
                 processor: AutoProcessor = None):
        """
        Initialize the policy generator
        Args:
            state: Dict
            image: dict
            history: List[str]
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = device
        self.model = model
        self.processor = processor

        self.image = image
        self.history = history
        self.state = state
        self.action_space = ["overtaking", "keeping lane", "turning left", "turning right", 
                           "left change", "right change", "brake"]
        self.messages = None
        self.navigation_info = navigation_info

    def generate_policy_id(self):
        """
        Generate the policy
        """
        # build messages
        state_description = build_state_description(self.state)
        history_description = build_history_description(self.history)
        prompt = build_macro_action_prompt(state_description, history_description, self.action_space, self.navigation_info)
        self.messages = build_messages(self.image, prompt)

        # generate action
        output_text = generate_output_text(self.device, self.model, self.processor, self.messages)

        # parse action from response
        action = parse_action(output_text)
        reasoning = parse_reasoning(output_text)
        print(f"Reasoning: {reasoning}")
        print(f"Generated action: {action}")
        
        # transform action to id
        action_id = action_to_id(action, self.action_space)
        return action_id


def refinement_policy_id(macro_action: str, state: Dict, image: dict, history: List[str], model_name: str = "Qwen/Qwen2.5-VL-3B"
                         , device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         model: Qwen2_5_VLForConditionalGeneration = None,
                         processor: AutoProcessor = None):
    """
    Refine the policy id
    """
    # build messages
    state_description = build_state_description(state)
    history_description = build_history_description(history)
    prompt = build_mid_action_prompt(state_description, history_description, macro_action)
    messages = build_messages(image, prompt)
    
    output_text = generate_output_text(device, model, processor, messages)
    
    # parse action from response
    speed = parse_speed(output_text)    
    steering = parse_steering(output_text)
    print(f"Speed: {speed}")
    print(f"Steering: {steering}")
    mid_action = {"speed": speed, "steering": steering}
    return mid_action

# low level policy generator
class LowLevelPolicyGenerator:
    def __init__(self, state: Dict, mid_action: dict, image: dict, history: List[str], navigation_info: str, model_name: str = "Qwen/Qwen2.5-VL-3B"
                 , device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 model: Qwen2_5_VLForConditionalGeneration = None,
                 processor: AutoProcessor = None):
        """
        Initialize the policy generator
        Args:
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = device
        self.model = model
        self.processor = processor

        self.image = image
        self.history = history
        self.state = state
        self.mid_action = mid_action
        self.navigation_info = navigation_info
        self.action_space = {"acceleration": 0, "steering": 0}

    def generate_policy(self):
        """
        Generate the policy
        """
        # build messages
        state_description = build_state_description(self.state)
        history_description = build_history_description(self.history)
        prompt = build_atomic_action_prompt(state_description, history_description, self.mid_action)
        messages = build_messages(self.image, prompt)

        output_text = generate_output_text(self.device, self.model, self.processor, messages)

        # parse action from response
        acceleration = parse_acceleration(output_text)
        steering = parse_steering(output_text)
        print(f"Acceleration: {acceleration}")
        print(f"Steering: {steering}")

        self.action_space["acceleration"] = acceleration
        self.action_space["steering"] = steering
        return self.action_space
