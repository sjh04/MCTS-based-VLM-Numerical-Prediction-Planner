import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Union, Tuple
import numpy as np
from PIL import Image
import sys
sys.path.append("/home/ubuntu/sjh04/MCTS-based-VLM-Numerical-Prediction-Planner/src")

from VLM.utils import *
from qwen_vl_utils import process_vision_info
from VLM.qwen import Qwen
import json

# high level policy generator
class HighLevelPolicyGenerator:
    def __init__(self, model):
        """
        Initialize the policy generator
        Args:
            state: Dict
            image: dict
            history: List[str]
            model_name: Qwen2.5-VL-3B model name
        """
        self.model = model

        self.action_space = ["overtaking", "keeping lane", "turning left", "turning right", 
                           "left change", "right change", "brake"]
        self.messages = None

    def generate_policy_id(self, state: Dict, image: dict, history: List[str], navigation_info: str):
        """
        Generate the policy
        """
        # build messages
        state_description = build_state_description(state)
        history_description = build_history_description(history)
        prompt = build_macro_action_prompt(state_description, history_description, self.action_space, navigation_info)
        self.messages = build_messages(image, prompt)

        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, self.messages)

        # parse action from response
        action = parse_action(output_text)
        reasoning = parse_reasoning(output_text)
        print(f"Reasoning: {reasoning}")
        print(f"Generated action: {action}")
        
        # transform action to id
        action_id = action_to_id(action, self.action_space)
        return action_id

    def calculate_probabilities(self, state: dict, camera_image: dict, historys: List[str], navigation_info: str, trust_level: float = 1.0, aggression: float = 0.1):
        """Calculate probabilities using both VLM and safety/feasibility rules"""
        # Get base probability from VLM
        base_probabilities = np.ones(len(self.action_space)) * (trust_level / 2)
        
        # Execute standard policy generation
        try:
            action = self.generate_policy_id(state, camera_image, historys, navigation_info)
            base_probabilities[action] = 1.0
        except:
            # Fallback to uniform
            base_probabilities = np.ones(len(self.action_space)) / len(self.action_space)
        
        # Apply domain rules to modify probabilities
        safety_mask = np.ones(len(self.action_space))
        
        if state.get('speed', 0) > 100:  # High speed
            # Reduce probability of sharp turns
            turn_indices = [i for i, a in enumerate(self.action_space) if 'turn' in a]
            for idx in turn_indices:
                safety_mask[idx] = aggression
        
        # Apply mask and normalize  
        modified_probs = base_probabilities * safety_mask
        if np.sum(modified_probs) > 0:
            modified_probs = modified_probs / np.sum(modified_probs)
        else:
            modified_probs = np.ones(len(self.action_space)) / len(self.action_space)
        
        return modified_probs


def refinement_policy_id(macro_action: str, state: Dict, image: dict, history: List[str], 
                         model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",):
    """
    Refine the policy id
    """
    # build messages
    qwen = Qwen(model_name)
    device, model, processor = qwen.get_components()
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
    def __init__(self, model):
        """
        Initialize the policy generator
        Args:
            model_name: Qwen2.5-VL-3B model name
        """
        self.model = model

        self.action_space = {"acceleration": 0, "steering": 0}

    def generate_policy(self, state: Dict, image: dict, history: List[str], mid_action: Dict):
        """
        Generate the policy
        """
        # build messages
        state_description = build_state_description(state)
        history_description = build_history_description(history)
        prompt = build_atomic_action_prompt(state_description, history_description, self.mid_action)
        messages = build_messages(image, prompt)

        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, messages)

        # parse action from response
        acceleration = parse_acceleration(output_text)
        steering = parse_steering(output_text)
        print(f"Acceleration: {acceleration}")
        print(f"Steering: {steering}")

        self.action_space["acceleration"] = acceleration
        self.action_space["steering"] = steering
        return self.action_space

class VehicleBelief:
    def __init__(self, model):
        """
        Initialize the policy generator
        Args:
            state: Dict
            image: dict
            history: List[str]
            model_name: Qwen2.5-VL-3B model name
        """
        self.model = model

        self.action_space = ["overtaking", "keeping lane", "turning left", "turning right", 
                           "left change", "right change", "brake"]
        self.messages = None

    def generate_belief(self, state: Dict, image: dict):
        """
        Generate the belief
        """
        # build messages
        state_description = build_state_description(state)
        prompt = build_belief_prompt(state_description, self.action_space)
        self.messages = build_messages(image, prompt)

        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, self.messages)
        json_output = parse_json(output_text)

        json_path = 'belief/belief.json'

        # write output to json
        with open(json_path, 'w') as f:
            json.dump(json_output, f)
        
        return json_path
