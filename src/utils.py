import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# loading model
def loading_model(model_name: str) -> Qwen2_5_VLForConditionalGeneration:
    """
    Loading the model
    """
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# loading processor
def loading_processor(model_name: str) -> AutoProcessor:
    """
    Loading the processor
    """
    return AutoProcessor.from_pretrained(model_name)

# loading device
def loading_device() -> torch.device:
    """
    Loading the device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


