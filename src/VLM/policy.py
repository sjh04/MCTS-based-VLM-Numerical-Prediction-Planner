import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union, Tuple
import numpy as np
from PIL import Image
from .process_image import process_camera_images

class HighLevelPolicyGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B"):
        """
        Initialize the policy generator
        Args:
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.action_space = ["overtaking", "keeping lane", "turning left", "turning right", 
                           "left change", "right change", "brake"]
        self.action_to_id = {action: i for i, action in enumerate(self.action_space)}
        self.id_to_action = {i: action for i, action in enumerate(self.action_space)}
    
    def generate_single_action_id(self, camera_images: Dict[str, Image.Image], 
                             current_state: Dict,
                             history: List[str]) -> int:
        """
        Generate the most likely action based on the current state and history
        Args:
            camera_images: dictionary of camera images
            current_state: current vehicle state
            history: list of previous actions
        Returns:
            action id
        """
        # build the state description
        state_description = self._build_state_description(current_state, history)
        
        # process the image input
        model_input = process_camera_images(camera_images, state_description)
        
        # build the prompt
        prompt = self._build_single_action_prompt(model_input['text'])
        
        # generate the model output
        with torch.no_grad():
            # Prepare text input
            text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Prepare image input
            image_inputs = []
            for image in model_input['images']:
                image_input = self.model.processor(images=image, return_tensors="pt").to(self.device)
                image_inputs.append(image_input)
            
            # Combine text and image inputs
            inputs = {
                **text_inputs,
                'images': image_inputs
            }
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # parse the model output to get the most likely action
        action = self._parse_single_action_output(response)
        print(f"Action: {action}")

        return self.action_to_id[action]

    def _build_state_description(self, current_state: Dict, history: List[str]) -> str:
        """
        Build the state description
        """
        desc = f"Current vehicle state:\n"
        desc += f"Speed: {current_state.get('speed', 0)} km/h\n"
        desc += f"Lane: {current_state.get('lane', 'unknown')}\n"
        desc += f"Action history: {', '.join(history)}\n"
        return desc
    
    
    def _build_single_action_prompt(self, state_text: str) -> str:
        """
        Build the model prompt for single action generation
        """
        prompt = f"""Based on the following vehicle state and camera images, 
        please analyze the situation and provide the most appropriate action.
        Consider safety, efficiency, and traffic rules.
        
        {state_text}
        
        Please provide your analysis in the following format:
        Most appropriate action: [action]
        """
        return prompt
    
    
    def _parse_single_action_output(self, response: str) -> str:
        """
        Parse the model output to get the most likely action
        """
        action = None

        # parse the action
        action_line = response.split("Most appropriate action:")[1]
        action = action_line.strip().strip("[]")
        
        return action


class LowLevelPolicyGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B"):
        """
        Initialize the policy generator
        Args:
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()
