import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union
import numpy as np

def process_camera_images(camera_images: Dict[str, Image.Image], description: str) -> Dict:
        
    """
    Process camera images and convert to the Qwen2.5-vl-3b model input format
    Args:
        camera_images: A dictionary containing all camera images
        description: The image description text
    Returns:
        The processed model input format
    """
    # Build the input text
    input_text = f"<image>Here are the vehicle's surrounding images:\n{description}</image>\n"
        
    # Process all images
    processed_images = []
    for name, image in camera_images.items():
        # Ensure the image is RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize the image to the standard size required by the model
        image = image.resize((448, 448))
        processed_images.append(image)
        
    # Build the model input
    model_input = {
        'text': input_text,
        'images': processed_images
    }
        
    return model_input    
    
    


