import os
import torch
import numpy as np
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import matplotlib.pyplot as plt
from PIL import Image

def loading_device(device_str="cuda:0"):
    """
    Load the appropriate device for computation
    
    Args:
        device_str: The device string, e.g., 'cuda:0', 'cpu'
        
    Returns:
        torch.device: The PyTorch device
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    return torch.device(device_str)

def loading_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0", use_flash_attn=True):
    """
    Load a pre-trained VLM model
    
    Args:
        model_name: The model name or path
        device: Device to load the model on
        use_flash_attn: Whether to use Flash Attention 2 for acceleration
        
    Returns:
        The loaded model
    """
    print(f"Loading model {model_name} on {device}")
    
    device_obj = loading_device(device)
    dtype = torch.float16 if device_obj.type == "cuda" else torch.float32
    
    if use_flash_attn and device_obj.type == "cuda":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            device_map=device
        )
    
    return model

def loading_processor(model_name="Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=None, max_pixels=None):
    """
    Load the processor for a VLM model
    
    Args:
        model_name: The model name or path
        min_pixels: Minimum number of pixels per image token
        max_pixels: Maximum number of pixels per image token
        
    Returns:
        The loaded processor
    """
    print(f"Loading processor for {model_name}")
    
    if min_pixels is not None and max_pixels is not None:
        processor = AutoProcessor.from_pretrained(
            model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name)
    
    return processor

def save_test_results(result_dir, step, num_step, metrics_list, images=None):
    """
    Save test results including metrics and images
    
    Args:
        result_dir: Directory to save results
        step: Step number
        metrics: Dictionary of metrics to save
        images: Dictionary of images to save
    """
    # Save metrics as JSON
    print(f"metrics length: {len(metrics_list)}")
    for metrics_step in range(len(metrics_list)):

        metrics = metrics_list[metrics_step]

        metrics_path = os.path.join(result_dir, "metrics", f"step_{metrics_step:04d}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
        if metrics_step % 5 == 0:  # Create visualizations every 10 steps
            visualize_metrics(result_dir, metrics_step)
    # Save images if provided
    # if images:
    #     for img_name, img_data in images.items():
    #         if img_data is not None:
    #             # Convert to PIL Image if numpy array
    #             if isinstance(img_data, np.ndarray):
    #                 img = Image.fromarray(img_data)
    #                 img_path = os.path.join(result_dir, "images", f"step_{step:04d}_{img_name}.png")
    #                 img.save(img_path)
    
    # Create visualizations for key metrics
    

def visualize_metrics(result_dir, max_step):
    """
    Create visualizations of metrics
    
    Args:
        result_dir: Directory containing metrics
        max_step: Maximum step to include
    """
    # Load metrics for all steps
    metrics_data = []
    for step in range(max_step + 1):
        metrics_path = os.path.join(result_dir, "metrics", f"step_{step:04d}.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                data['step'] = step
                metrics_data.append(data)
    
    if not metrics_data:
        return
    
    # Extract data for plotting
    steps = [data['step'] for data in metrics_data]
    rewards = [data.get('reward', 0) for data in metrics_data]
    total_rewards = [data.get('total_reward', 0) for data in metrics_data]
    planning_times = [data.get('planning_time', 0) for data in metrics_data]
    
    # New metrics
    trajectory_smoothness = [data.get('trajectory_smoothness', 0) for data in metrics_data]
    vehicle_speeds = [data.get('vehicle_speed', 0) for data in metrics_data]
    
    # Extract reward components if available
    efficiency_rewards = [data.get('efficiency_reward', None) for data in metrics_data]
    safety_rewards = [data.get('safety_reward', None) for data in metrics_data]
    progress_rewards = [data.get('progress_reward', None) for data in metrics_data]
    comfort_rewards = [data.get('comfort_reward', None) for data in metrics_data]
    
    # Check if we have reward components to plot
    has_reward_components = not all(r is None for r in efficiency_rewards)
    
    # Create reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, 'b-', label='Step Reward')
    plt.plot(steps, total_rewards, 'r-', label='Cumulative Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "visualizations", f"rewards_{max_step:04d}.png"))
    plt.close()
    
    # Create planning time plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, planning_times, 'g-')
    plt.xlabel('Step')
    plt.ylabel('Planning Time (s)')
    plt.title('Planning Time Over Steps')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "visualizations", f"planning_time_{max_step:04d}.png"))
    plt.close()
    
    # Create vehicle speed plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, vehicle_speeds, 'g-')
    plt.xlabel('Step')
    plt.ylabel('Speed (km/h)')
    plt.title('Vehicle Speed Over Time')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "visualizations", f"speed_{max_step:04d}.png"))
    plt.close()
    
    # Create reward components plot if available
    if has_reward_components:
        plt.figure(figsize=(10, 5))
        if not all(r is None for r in efficiency_rewards):
            plt.plot(steps, efficiency_rewards, 'g-', label='Efficiency')
        if not all(r is None for r in safety_rewards):
            plt.plot(steps, safety_rewards, 'r-', label='Safety')
        if not all(r is None for r in progress_rewards):
            plt.plot(steps, progress_rewards, 'b-', label='Progress')
        if not all(r is None for r in comfort_rewards):
            plt.plot(steps, comfort_rewards, 'y-', label='Comfort')
        plt.xlabel('Step')
        plt.ylabel('Reward Component')
        plt.title('Reward Components Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, "visualizations", f"reward_components_{max_step:04d}.png"))
        plt.close()

def create_directories(output_dir, episode=None):
    """
    Create directories for output results
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to created results directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp-based directory for this run
    timestamp = datetime.now().strftime("%H%M%S")
    results_dir = os.path.join(output_dir, f"test_{timestamp}_episode_{episode}" if episode is not None else f"results_{timestamp}")
    os.makedirs(results_dir)
    
    # Create subdirectories for different types of data
    os.makedirs(os.path.join(results_dir, "metrics"))
    os.makedirs(os.path.join(results_dir, "images"))
    os.makedirs(os.path.join(results_dir, "visualizations"))
    
    return results_dir

def extract_state_description(observation, env_type="highway"):
    """
    Extract human-readable state description from observation
    
    Args:
        observation: Environment observation (can be text or structured data)
        env_type: Type of environment ("highway" or "carla")
        
    Returns:
        state_description: Human-readable state description
    """
    if isinstance(observation, str):
        # Already a text description
        return observation
        
    elif isinstance(observation, dict):
        # Extract structured information from dictionary
        description = "Current vehicle state:\n"
        
        # Extract ego vehicle state
        if "ego" in observation:
            ego = observation["ego"]
            if "position" in ego:
                description += f"Position: ({ego['position'][0]:.1f}, {ego['position'][1]:.1f})\n"
            if "speed" in ego:
                description += f"Speed: {ego['speed']:.1f} km/h\n"
            if "heading" in ego:
                description += f"Heading: {ego['heading']:.1f} degrees\n"
        
        # Extract information about other vehicles
        if "vehicles" in observation and observation["vehicles"]:
            description += f"\nSurrounding vehicles ({len(observation['vehicles'])}):\n"
            for i, vehicle in enumerate(observation["vehicles"]):
                description += f"Vehicle {i+1}: "
                if "position" in vehicle:
                    rel_x = vehicle["position"][0] - observation["ego"]["position"][0]
                    rel_y = vehicle["position"][1] - observation["ego"]["position"][1]
                    description += f"at ({rel_x:.1f}, {rel_y:.1f}) meters relative, "
                if "speed" in vehicle:
                    description += f"moving at {vehicle['speed']:.1f} km/h"
                description += "\n"
        
        # Extract environment information
        if "road" in observation:
            description += "\nRoad information:\n"
            if "lanes" in observation["road"]:
                description += f"Number of lanes: {len(observation['road']['lanes'])}\n"
            if "curvature" in observation["road"]:
                description += f"Road curvature: {observation['road']['curvature']:.2f}\n"
        
        return description
        
    elif isinstance(observation, np.ndarray):
        # Numeric observation array
        if env_type == "highway":
            # Typical Highway-Env observation format
            description = "Current vehicle state:\n"
            
            # For Kinematics observation type
            if len(observation.shape) == 2 and observation.shape[1] >= 5:
                # First row typically contains ego vehicle
                ego = observation[0]
                if ego[0] > 0:  # Presence feature
                    description += f"Speed: {ego[2]*130:.1f} km/h\n"  # Normalized speed
                    description += f"Lane position: {ego[1]:.1f}\n"  # Lateral position
                
                # Other vehicles
                others = [v for i, v in enumerate(observation) if i > 0 and v[0] > 0]
                if others:
                    description += f"\nSurrounding vehicles ({len(others)}):\n"
                    for i, vehicle in enumerate(others):
                        rel_x = vehicle[1] - ego[1]  # Relative lateral position
                        rel_y = vehicle[2] - ego[2]  # Relative longitudinal position
                        description += f"Vehicle {i+1}: at ({rel_x:.1f}, {rel_y:.1f}) relative\n"
            
            return description
        
        elif env_type == "carla":
            # For CARLA, this should be rare, but handle as fallback
            return f"Numeric observation with shape {observation.shape}"
    
    else:
        # Fallback for unknown observation format
        return str(observation)

def extract_history_description(history, max_actions=5):
    """
    Create a description of the action history
    
    Args:
        history: List of previous actions
        max_actions: Maximum number of actions to include
        
    Returns:
        history_description: Description of action history
    """
    if not history:
        return "No previous actions"
        
    # Limit the number of actions to show
    recent_actions = history[-max_actions:] if len(history) > max_actions else history
    
    description = f"Last {len(recent_actions)} actions:\n"
    for i, action in enumerate(recent_actions):
        description += f"{len(history) - len(recent_actions) + i + 1}. {action}\n"
        
    return description


def calculate_curvature(x, y):
    """
    Calculate the mean curvature of a 2D curve defined by x and y coordinates.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        float: Mean curvature of the curve
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-8
    denominator = (dx**2 + dy**2)**1.5
    
    # Avoid division by zero
    mask = denominator > epsilon
    curvature = np.zeros_like(dx)
    curvature[mask] = np.abs(ddx[mask] * dy[mask] - dx[mask] * ddy[mask]) / denominator[mask]
    
    # Handle any remaining NaN or Inf values
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    
    curvature_mean = np.mean(curvature)
    return curvature_mean