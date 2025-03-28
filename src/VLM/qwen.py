from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Dict, List, Union
import torch 

class Qwen:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize the Qwen model
        Args:
            model_name: Qwen2.5-VL-3B model name
        """
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_components(self):
        """
        Get the components
        """
        return self.device, self.model, self.processor