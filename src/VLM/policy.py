import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Union, Tuple
import numpy as np
from PIL import Image
from .utils import *
from qwen_vl_utils import process_vision_info


class HighLevelPolicyGenerator:
    def __init__(self, state: Dict, image: dict, history: List[str], navigation_info: str, model_name: str = "Qwen/Qwen2.5-VL-3B"):
        """
        Initialize the policy generator
        Args:
            state: Dict
            image: dict
            history: List[str]
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

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
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # parse action from response
        action = parse_action(output_text)
        reasoning = parse_reasoning(output_text)
        print(f"Reasoning: {reasoning}")
        print(f"Generated action: {action}")
        
        # transform action to id
        action_id = action_to_id(action, self.action_space)
        return action_id


def refinement_policy_id(macro_action: str, state: Dict, image: dict, history: List[str]):
    """
    Refine the policy id
    """
    # build messages
    state_description = build_state_description(state)
    history_description = build_history_description(history)
    prompt = build_mid_action_prompt(state_description, history_description, macro_action)
    messages = build_messages(image, prompt)
    
    pass

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
