import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Dict, List, Union
import numpy as np
from qwen_vl_utils import process_vision_info
import sys
sys.path.append("/home/ubuntu/sjh04/MCTS-based-VLM-Numerical-Prediction-Planner/src")
from utils import *

# transform action to id 
def action_to_id(action: str, action_space: List[str]) -> int:
    """Convert action string to index in action space"""
    if not action_space:
        return 0
        
    try:
        return action_space.index(action)
    except ValueError:
        # Return default action (0) if not found
        return 0

# transform id to action
def id_to_action(id: int, action_space: List[str]) -> str:
    """Convert index to action string"""
    if 0 <= id < len(action_space):
        return action_space[id]
    # Return default action if index is out of bounds
    return action_space[0] if action_space else ""

# build messages
def build_messages(image: dict = None, prompt: str = "", observation: str = None, systerm_promt="You are the ego driver, driving on a highway. You must drive smoothly and safely!") -> list:
    """
    Build messages for VLM model
    
    Args:
        image: Image dictionary (front camera)
        prompt: Text prompt
        observation: Text observation
        systerm_promt: System prompt
        
    Returns:
        List of messages in VLM format
    """
    messages = [{"role": "system", "content": systerm_promt}]
    
    # Use observation OR image + prompt
    if observation is not None and isinstance(observation, str):
        # For Highway Environment, use text observation
        messages.append({
            "role": "user", 
            "content": f"Observation: {observation}\n\n{prompt}"
        })
    elif image is not None:
        # For CARLA Environment, use image + prompt
        if isinstance(image, dict) and "front" in image:
            front_img = Image.fromarray(image["front"])
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            # Fallback to text-only if image format is unexpected
            messages.append({"role": "user", "content": prompt})
    else:
        # Text-only prompt
        messages.append({"role": "user", "content": prompt})
        
    return messages

# build state description
def build_state_description(state: Dict) -> str:
    """
    Build text description of state
    
    Args:
        state: State dictionary
        
    Returns:
        Text description
    """
    description = "Current vehicle state:\n"
    
    # Handle different state formats
    if isinstance(state, dict):
        # Extract key information like position, speed, heading
        if "ego_pos" in state and state["ego_pos"].size > 0:
            pos = state["ego_pos"][0, -1]
            description += f"Position: ({pos[0]:.1f}, {pos[1]:.1f})\n"
            
        if "ego_speed" in state and state["ego_speed"].size > 0:
            speed = state["ego_speed"][0, -1][0]
            description += f"Speed: {speed * 3.6:.1f} km/h\n"  # Convert m/s to km/h
            
        if "ego_yaw" in state and state["ego_yaw"].size > 0:
            yaw = state["ego_yaw"][0, -1][0]
            description += f"Heading: {yaw:.2f} rad\n"
            
        # Add information about other vehicles if available
        if "agents" in state and state["agents"].size > 0:
            description += f"\nSurrounding vehicles: {state['agents'].shape[1]}\n"
    elif isinstance(state, str):
        # If state is already a string, use it directly
        description = state
        
    return description

# build history description
def build_history_description(history: List[str]) -> str:
    """
    Build text description of action history
    
    Args:
        history: List of previous actions
        
    Returns:
        Text description
    """
    if not history:
        return "No previous actions."
        
    description = "Previous actions:\n"
    # Only include the last few actions to avoid too much text
    recent_history = history[-5:] if len(history) > 5 else history
    
    for i, action in enumerate(recent_history):
        description += f"{len(history) - len(recent_history) + i + 1}. {action}\n"
        
    return description

# build macro action prompt with environment-specific handling
def build_macro_action_prompt(state_description: str, history_description: str, action_space: List[str], navigation_info: str, env_type: str = "highway") -> str:
    """
    Build prompt for macro action prediction
    
    Args:
        state_description: Description of current state
        history_description: Description of action history
        action_space: Available actions
        navigation_info: Navigation information
        env_type: Environment type ("highway" or "carla")
        
    Returns:
        Prompt for VLM
    """
    # Base prompt
    prompt = f"{state_description}\n\n{history_description}\n\n"
    
    # Add navigation info
    prompt += f"Navigation information: {navigation_info}\n\n"
    
    # Add available actions
    prompt += "Available high-level actions:\n"
    for action in action_space:
        prompt += f"- {action}\n"
        
    # Add specific prompt based on environment type
    if env_type.lower() == "highway":
        prompt += "\nYou are driving on a highway. Based on the observation, choose the best high-level action from the available actions.\n"
    else:  # carla
        prompt += "\nYou are driving in an urban environment. Based on the camera view, choose the best high-level action from the available actions.\n"
        
    prompt += "\nResponse format: Only return the chosen action without any explanation."
    
    return prompt

# build mid action prompt
def build_mid_action_prompt(state_description: str, history_list: List[str], macro_action: str, env_type: str = "highway") -> str:
    """
    Build prompt for mid-level action prediction
    
    Args:
        state_description: Description of current state
        history_list: List of previous high-level actions
        macro_action: Selected macro action
        env_type: Environment type ("highway" or "carla")
        
    Returns:
        Prompt for VLM
    """
    steering_angle = np.pi/4

    formatted_history = build_history_description(history_list)
    macro_action_text = translate_high_action2text(macro_action)

    prompt = f"{state_description}\n\n{formatted_history}\n\n"
    prompt += f"Recommended high-level action: {macro_action_text}\n\n"
    
    if env_type.lower() == "highway":
        prompt += "Based on the Recommended high-level action and the current state, provide appropriate acceleration and steering angle values.\n"
        prompt += "- Acceleration should be a value between -3 and 3 m/s^2.\n"
        prompt += f"- Steering angle: value between -{round(steering_angle, 2)} (full left) and {round(steering_angle, 2)} (full right)\n"
    else:  # carla
        prompt += "Based on the selected high-level action and the camera view, provide appropriate speed values.\n"
        prompt += "Speed should be a value between 0 and 60 km/h.\n"
        
    prompt += "\nResponse format: Acceleration: value, Steering: value"
    
    return prompt

# Build atomic action prompt
def build_atomic_action_prompt(state_description: str, history_list: List[str], high_action: str, mid_action_dict: dict, env_type: str = "highway") -> str:
    """
    Build prompt for atomic action prediction
    
    Args:
        state_description: Description of current state
        history_list: List of previous high-level actions
        high_action: Selected macro action
        mid_action_dict: Dictionary of mid-level action parameters
        env_type: Environment type ("highway" or "carla")
        
    Returns:
        Prompt for VLM
    """
    formatted_history = build_history_description(history_list)
    high_action_text = translate_high_action2text(high_action)
    steering_angle = np.pi/4
    prompt = f"{state_description}\n\n{formatted_history}\n\n"
    prompt += f"Recommended high-level action: {high_action_text}\n\n"
    
    prompt += f"Recommended mid-level action:\n"
    if isinstance(mid_action_dict, dict):
        if "acceleration" in mid_action_dict:
            prompt += f"Acceleration: {mid_action_dict['acceleration']} m/s^2\n"
        if "steering" in mid_action_dict:
            prompt += f"Steering angle: {mid_action_dict['steering']}\n"
    
    if env_type.lower() == "highway":
        prompt += "Based on the high-level and mid-level action, provide specific vehicle control inputs:\n"
        prompt += "- Acceleration: value between -3 (decelerate)m/s^2 and 3 (accelerate) m/s^2\n"
        prompt += f"- Steering angle: value between -{round(steering_angle, 2)} (full left) and {round(steering_angle, 2)} (full right)\n"
    else:  # carla
        prompt += "Based on the high-level action and the camera view, provide specific vehicle control inputs:\n"
        prompt += "- Acceleration: value between -1.0 (full brake) and 1.0 (full throttle)\n"
        prompt += "- Steering: value between -1.0 (full left) and 1.0 (full right)\n"
        
    prompt += "\nResponse format: Acceleration: value, Steering: value"
    
    return prompt

# Build belief prompt
def build_belief_prompt(state_description: Dict, action_space: List[str], env_type: str = "highway") -> str:
    """
    Build prompt for belief prediction
    
    Args:
        state_description: Description of current state
        action_space: Available actions
        env_type: Environment type ("highway" or "carla")
        
    Returns:
        Prompt for VLM
    """
    # Convert state_description to string if it's a dictionary
    if isinstance(state_description, dict):
        # Format a simple text representation of vehicle states
        prompt = "Vehicle states:\n"
        for vehicle_id, state in state_description.items():
            prompt += f"Vehicle {vehicle_id}:\n"
            for key, value in state.items():
                prompt += f"  {key}: {value}\n"
    else:
        prompt = f"{state_description}\n\n"
    
    # Add available actions
    prompt += "\nAvailable actions for other vehicles:\n"
    for action in action_space:
        prompt += f"- {action}\n"
    
    if env_type.lower() == "highway":
        prompt += "\nBased on the highway situation described above, predict the most likely actions of other vehicles.\n"
    else:  # carla
        prompt += "\nBased on the urban traffic situation, predict the most likely actions of other vehicles.\n"
        
    prompt += "\nProvide predictions in JSON format with vehicle IDs and most likely actions.Strictly follow the JSON format.\n"
    prompt += "\nExample: [{\"vehicle_id\": 1, \"action\": \"keeping_lane\"},]"
    
    return prompt

# Helper function to detect environment type
def detect_environment_type(observation, image):
    """
    Detect the environment type based on available inputs
    
    Args:
        observation: Text observation (for Highway Environment)
        image: Image dictionary or list (for CARLA)
        
    Returns:
        Environment type: "highway" or "carla"
    """
    # If we have an image with CARLA-specific structure (as dictionary)
    if isinstance(image, dict) and any(key in image for key in ["front", "lidar", "semantic", "depth"]):
        return "carla"
    
    # If we have an image as list
    elif isinstance(image, list):
        # Try to determine if it's a CARLA image list
        if len(image) > 0 and isinstance(image[0], (np.ndarray, Image.Image)):
            return "carla"
    
    # If we have a text observation, it's Highway Environment
    if isinstance(observation, str) and len(observation) > 0:
        return "highway"
    
    # Default to Highway Environment
    return "highway"

# Generate output text from VLM model
def generate_output_text(device, model, processor, messages):
    """
    Generate text output from VLM model
    
    Args:
        device: Device to run model on
        model: VLM model
        processor: VLM processor
        messages: Input messages
        
    Returns:
        Generated text
    """
    try:        
        # Process vision information in messages
        # image_inputs, video_inputs = process_vision_info(messages)
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # if model.model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        #     inputs = processor(
        #         text=[text],
        #         images=image_inputs,
        #         videos=video_inputs,
        #         padding=True,
        #         return_tensors="pt",
        #     )
        # else:
        #     inputs = processor([text], return_tensors="pt")

        inputs = processor([text], return_tensors="pt")
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    except Exception as e:
        print(f"Error in text generation: {e}")
        return ""

# Parse values from the model's output text
def parse_speed(text):
    """Parse speed value from text"""
    try:
        import re
        speed_pattern = r"Speed:\s*([0-9.]+)"
        match = re.search(speed_pattern, text)
        if match:
            return float(match.group(1))
        return 20.0  # Default speed
    except:
        return 20.0

def parse_steering(text):
    """Parse steering value from text"""
    try:
        import re
        steering_pattern = r"Steering:\s*([+-]?[0-9.]+)"
        match = re.search(steering_pattern, text)
        if match:
            return float(match.group(1))
        return 0.0  # Default steering
    except:
        return 0.0

def parse_acceleration(text):
    """Parse acceleration value from text"""
    try:
        import re
        acc_pattern = r"Acceleration:\s*([+-]?[0-9.]+)"
        match = re.search(acc_pattern, text)
        if match:
            return float(match.group(1))
        return 0.0  # Default acceleration
    except:
        return 0.0

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def translate_high_action2text(high_action):
    """
    Translate high-level action to text description
    Args:
        high_action: High-level action string
    Returns:
        Text description of the action
    """
    if high_action == "keeping_lane":
        return "Keeping lane"
    elif high_action == "left_change":
        return "Changing lane to the left"
    elif high_action == "right_change":
        return "Changing lane to the right"
    elif high_action == "overtaking":
        return "Overtaking"
    elif high_action == "turning_left":
        return "Turning left"
    elif high_action == "turning_right":
        return "Turning right"
    else:
        return high_action  # Default case