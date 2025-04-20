import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Union, Tuple
import numpy as np
from PIL import Image
import sys
import os
sys.path.append("/home/ubuntu/sjh04/MCTS-based-VLM-Numerical-Prediction-Planner/src")

from VLM.utils import *
from qwen_vl_utils import process_vision_info
from VLM.qwen import Qwen
import json

# high level policy generator
class MacroActionPolicy:
    def __init__(self, model):
        self.model = model
        self.action_space = ["overtaking", "keeping_lane", "turning_left", "turning_right", 
                           "left_change", "right_change", "brake"]
        self.messages = None
        
    def act(self, state_description, history_description, navigation_info, image=None, observation=None):
        """
        Generate an action based on the current state
        
        Args:
            state_description: Description of the current state
            history_description: Description of action history
            navigation_info: Navigation information
            valid_actions: List of valid actions
            image: Camera images (for CARLA)
            observation: Text observation (for Highway Environment)
        """
        # Determine environment type
        env_type = detect_environment_type(observation, image)
        
        # Build prompt using the appropriate approach for the environment
        prompt = build_macro_action_prompt(
            state_description, 
            history_description, 
            self.action_space, 
            navigation_info,
            env_type
        )
        print("==========")
        print(f"high_level Prompt: {prompt}")
        print("==========")
        # Build messages - will use text observation for Highway Environment
        self.messages = build_messages(image=image, prompt=prompt, observation=observation)
        
        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, self.messages)

        # Parse action from response
        try:
            action = output_text.strip()
            # Find the best matching action in action_space
            if action not in self.action_space:
                # Find closest match if not exact
                best_match = None
                best_score = 0
                for valid_action in self.action_space:
                    if valid_action in action:
                        score = len(valid_action)
                        if score > best_score:
                            best_score = score
                            best_match = valid_action
                
                if best_match:
                    action = best_match
                else:
                    # Default to first action if no match found
                    action = list(self.action_space)[0]
            
            print(f"Generated action: {action}")
            return action
        except Exception as e:
            print(f"Error parsing action: {e}")
            # Default to first action if parsing fails
            return list(self.action_space)[0]

    def calculate_probabilities(self, state_description, history_description, valid_actions, navigation_info, image=None, observation=None, trust_level=1.0, aggression=0.1):
        """Calculate probabilities using both VLM and safety/feasibility rules"""
        # Initialize with uniform distribution
        self.action_space = valid_actions
        if isinstance(self.action_space, list):
            action_list = self.action_space
        else:
            action_list = list(self.action_space.keys())
            
        base_probabilities = np.ones(len(action_list)) * (trust_level / 2)
        
        # Execute standard policy generation
        try:
            action = self.act(state_description, history_description, navigation_info, image=image, observation=observation)
            action_idx = action_list.index(action) if action in action_list else 0
            base_probabilities[action_idx] = 1.0
        except Exception as e:
            print(f"Error in VLM policy: {e}")
            # Fallback to uniform
            base_probabilities = np.ones(len(action_list)) / len(action_list)
        
        # Apply domain rules to modify probabilities based on state information
        safety_mask = np.ones(len(action_list))
        
        # Extract speed information if available
        speed = 0
        if isinstance(state_description, dict) and 'speed' in state_description:
            speed = state_description['speed']
        elif isinstance(state_description, str) and 'speed' in state_description:
            # Try to extract speed from description string
            try:
                speed_text = state_description.split('speed:')[1].split()[0]
                speed = float(speed_text)
            except:
                speed = 0
        
        # Apply safety rules based on speed
        if speed > 100:  # High speed
            # Reduce probability of sharp turns
            turn_indices = [i for i, a in enumerate(action_list) if 'turn' in a]
            for idx in turn_indices:
                safety_mask[idx] = aggression
        
        # Apply mask and normalize  
        modified_probs = base_probabilities * safety_mask
        if np.sum(modified_probs) > 0:
            modified_probs = modified_probs / np.sum(modified_probs)
        else:
            modified_probs = np.ones(len(action_list)) / len(action_list)
        
        return modified_probs


def get_mid_level_action(model, state_description, history_description, macro_action, image=None, observation=None):
    # Detect environment type
    env_type = detect_environment_type(observation, image)
    # print(f"state_description: {state_description}")
    # print(f"observation: {observation}")
    # Build prompt with environment-specific formatting
    prompt = build_mid_action_prompt(
        state_description, 
        history_description, 
        macro_action,
        env_type
    )
    print("==========")
    print(f"Mid-level Prompt: {prompt}")
    print("==========")
    # Build messages with both image and observation
    messages = build_messages(image=image, prompt=prompt, observation=observation)
    
    device, model, processor = model.get_components()
    output_text = generate_output_text(device, model, processor, messages)
    print(f"Mid-level Output: {output_text}")
    # parse action from response
    acc = parse_acceleration(output_text)    
    steering = parse_steering(output_text)
    print(f"Acceleration: {acc}")
    print(f"Steering: {steering}")
    mid_action = {"acceleration": acc, "steering": steering}
    return mid_action

# low level policy generator
class AtomicActionPolicy:
    def __init__(self, model):
        self.model = model
        self.high_level_action = None
        self.mid_action = None
        self.messages = None
        
    def act(self, state_description, history_description, mid_action, high_level_action, image=None, observation=None):
        # Detect environment type
        env_type = detect_environment_type(observation, image)
        
        self.mid_action = mid_action
        self.high_level_action = high_level_action
        # Build appropriate prompt
        prompt = build_atomic_action_prompt(
            state_description, 
            history_description, 
            self.high_level_action,
            self.mid_action,
            env_type
        )
        print("==========")
        print(f"low-level prompt: {prompt}")
        print("==========")
        # Build messages with both options
        messages = build_messages(image=image, prompt=prompt, observation=observation)
        
        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, messages)
        print(f"Atomic Output: {output_text}")
        # parse action from response
        acceleration = parse_acceleration(output_text)
        steering = parse_steering(output_text)
        print(f"Acceleration: {acceleration}")
        print(f"Steering: {steering}")

        return {"acceleration": acceleration, "steering": steering}
        

class BeliefUpdater:
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space
        self.messages = None
    
    def update(self, episode, step, state_description, image=None, observation=None):
        # Detect environment type
        env_type = detect_environment_type(observation, image)
        
        # Build appropriate prompt
        prompt = build_belief_prompt(
            state_description, 
            self.action_space,
            env_type
        )
        
        # Build messages with both options
        self.messages = build_messages(image=image, prompt=prompt, observation=observation)
        
        device, model, processor = self.model.get_components()
        # generate action
        output_text = generate_output_text(device, model, processor, self.messages)
        # print(f"Belief Output: {output_text}")
        # print(f"Type of belief output: {type(output_text)}")
        # parse action from response
        json_output = parse_json(output_text)
        # print(f"Parsed JSON: {json_output}")
        # sys.exit(0)
        # print(f"Parsed JSON: {json_output}")
        # print(f"type(json_output): {type(json_output)}")
        # Save JSON output to file
        directory = os.path.join("belief/belief_output", f"episode_{episode}")
        if not os.path.exists(directory):
            os.makedirs(directory)

        json_path = os.path.join(directory, f"step_{step}_belief_output.json")

        # write output to json
        with open(json_path, 'w') as f:
            f.write(json_output)
        
        return json_path
