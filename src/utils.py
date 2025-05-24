import os
import torch
import numpy as np
from datetime import datetime
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor # Not used here
import json
import matplotlib.pyplot as plt
from PIL import Image

# loading_device, loading_model, loading_processor, extract_state_description
# are more VLM specific and might belong to a VLM utils file or be used by Qwen class.
# For now, keeping save_test_results, visualize_metrics, create_directories here.

def save_test_results(result_dir, total_hl_steps_in_episode, metrics_list, images=None):
    """
    Save test results including metrics and images for one episode.
    
    Args:
        result_dir: Directory to save results for this episode.
        total_hl_steps_in_episode: Total number of high-level steps taken in the episode.
        metrics_list: List of metric dictionaries, one per high-level step.
        images: Dictionary of images to save (not currently used in this flow).
    """
    # Save metrics as JSON, one file per high-level step
    # print(f"Saving metrics for {len(metrics_list)} high-level steps in {result_dir}")
    for idx, metrics_at_hl_step in enumerate(metrics_list):
        metrics_path = os.path.join(result_dir, "metrics", f"hl_step_{idx:04d}.json")
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics_at_hl_step, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics for hl_step {idx} to {metrics_path}: {e}")
            # Fallback: try to print problematic metrics
            # print("Problematic metrics data:", metrics_at_hl_step)


    # Create visualizations for key metrics for the entire episode
    if metrics_list: # Ensure there's data to visualize
        visualize_metrics(result_dir, len(metrics_list) - 1) # Pass the last index of hl_step

def visualize_metrics(result_dir, max_hl_step_index):
    """
    Create visualizations of metrics for a single episode.
    
    Args:
        result_dir: Directory containing metrics for the episode.
        max_hl_step_index: The maximum index of high-level steps recorded (i.e., len(metrics_list) - 1).
    """
    metrics_data = []
    for step_idx in range(max_hl_step_index + 1):
        metrics_path = os.path.join(result_dir, "metrics", f"hl_step_{step_idx:04d}.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    # 'hl_step' should already be in data, but ensure it for x-axis
                    if 'hl_step' not in data: data['hl_step'] = step_idx 
                    metrics_data.append(data)
            except Exception as e:
                print(f"Error loading metrics from {metrics_path}: {e}")
        # else:
            # print(f"Metrics file not found: {metrics_path}")
    
    if not metrics_data:
        print(f"No metrics data found in {result_dir} to visualize.")
        return
    
    # Ensure visualization directory exists
    viz_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Extract data for plotting
    hl_steps_x_axis = [data['hl_step'] for data in metrics_data]
    
    # Plot accumulated reward per HL step
    hl_rewards_accum = [data.get('hl_reward_accum', 0) for data in metrics_data]
    plt.figure(figsize=(10, 5))
    plt.plot(hl_steps_x_axis, hl_rewards_accum, 'b-', label='Accumulated Reward per HL-Step')
    plt.xlabel('High-Level Step')
    plt.ylabel('Reward')
    plt.title('HL-Step Accumulated Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, f"hl_step_rewards.png"))
    plt.close()

    # Plot total cumulative reward for the episode
    total_rewards_episode = [data.get('total_hl_reward_episode', 0) for data in metrics_data]
    plt.figure(figsize=(10, 5))
    plt.plot(hl_steps_x_axis, total_rewards_episode, 'r-', label='Total Episode Reward Over HL-Steps')
    plt.xlabel('High-Level Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Total Episode Reward Progression')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(viz_dir, f"total_episode_reward_progression.png"))
    plt.close()
    
    # Plot planning time
    planning_times = [data.get('planning_time_total', 0) for data in metrics_data]
    if any(pt > 0 for pt in planning_times): # Only plot if data exists
        plt.figure(figsize=(10, 5))
        plt.plot(hl_steps_x_axis, planning_times, 'g-')
        plt.xlabel('High-Level Step')
        plt.ylabel('Total Planning Time (s)')
        plt.title('Total Planning Time per HL-Step')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, f"planning_time.png"))
        plt.close()
    
    # Plot vehicle speed
    vehicle_speeds = [data.get('ego_speed', 0) * 3.6 for data in metrics_data] # Convert m/s to km/h
    if any(vs > 0 for vs in vehicle_speeds) :
        plt.figure(figsize=(10, 5))
        plt.plot(hl_steps_x_axis, vehicle_speeds, 'm-')
        plt.xlabel('High-Level Step')
        plt.ylabel('Ego Speed (km/h)')
        plt.title('Ego Vehicle Speed Over HL-Steps')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, f"ego_speed.png"))
        plt.close()

    # Add more plots as needed (e.g., DQN steps per HL step, on_road status if boolean over time)

def create_directories(output_dir_base, episode_identifier_str):
    """
    Create directories for output results for a specific episode/run.
    
    Args:
        output_dir_base: Base output directory (e.g., './output').
        episode_identifier_str: String to identify this specific run/episode (e.g., "train_ep_0", "test_dqn_ep_5").
        
    Returns:
        Path to the created results directory for this specific run/episode.
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir_base, exist_ok=True)
    
    # Create a unique directory for this specific run/episode
    # Using a combination of a timestamp for uniqueness if multiple runs happen close together,
    # and the episode identifier for clarity.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize episode_identifier_str for directory naming if needed
    safe_ep_id_str = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in episode_identifier_str)
    
    # results_dir_for_episode = os.path.join(output_dir_base, f"{safe_ep_id_str}_{timestamp}")
    results_dir_for_episode = os.path.join(output_dir_base, safe_ep_id_str) # Simpler name if ep_id is unique enough
    os.makedirs(results_dir_for_episode, exist_ok=True) # exist_ok=True if retrying same episode
    
    # Create subdirectories for different types of data within this episode's directory
    os.makedirs(os.path.join(results_dir_for_episode, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir_for_episode, "images"), exist_ok=True) # If images are saved
    os.makedirs(os.path.join(results_dir_for_episode, "visualizations"), exist_ok=True)
    
    return results_dir_for_episode