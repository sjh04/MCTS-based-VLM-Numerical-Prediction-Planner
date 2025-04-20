import sys
import os

from sympy import Intersection

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import logging
import random
from datetime import datetime

# Import custom modules with absolute imports
from src.HighwayEnv.goal_checker import GoalChecker
from src.HighwayEnv.utils import *
from src.HighwayEnv.mcts_env import MCTSEnv
from src.MCTS.high_level_mcts import MCTSAgent as HighLevelMCTS
from src.MCTS.utils import get_vehicle_parameters, highway_lanes_to_path
from src.VLM.qwen import Qwen
from src.utils import *
from src.HighwayEnv.envScenario import EnvScenario

def idx2action(idx, action_space, action_coef):
    """
    Convert action index to action name
    Args:
        idx: Action index
        action_space: List of action names
        action_coef: Coefficients for actions
    Returns:
        Action name
    """
    action = action_space[0] + action_coef * idx
    return action
    
def normalize_action(action, action_range):
    """
    Normalize action from -1 to 1
    Args:
        action: Action to normalize
        action_range: Range of the action
    Returns:
        Normalized action
    """
    action = action / (action_range[1] - action_range[0])
    return action

def inverse_lmap(y, output_range, input_range = [-1, 1]):
    """
    将输出值 y 从 output_range 线性映射回 input_range。
    
    Args:
        y: 需要映射的值（在 output_range 内）。
        output_range: 原始输出范围（如 [-5, 5]）。
    
    Returns:
        x: 映射回 input_range 后的值。
    """
    A_in, B_in = input_range
    A_out, B_out = output_range

    # 避免除以零（假设 output_range 是有效范围）
    if B_out == A_out:
        return A_in  # 或抛出异常

    # 逆向线性映射
    x = ((y - A_out) / (B_out - A_out)) * (B_in - A_in) + A_in
    return x

def parse_args():
    parser = argparse.ArgumentParser(description="Run MCTS-based VLM Numerical Prediction Planner")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for results')
    return parser.parse_args()

def calculate_action_smoothness(current_action, previous_action):
    """Calculate the smoothness between consecutive actions"""
    if previous_action is None:
        return 1.0  # First action is considered perfectly smooth
    
    try:
        # For Highway Environment with [acceleration, steering] format
        if isinstance(current_action, list) and len(current_action) == 2:
            acc_change = abs(current_action[0] - previous_action[0])
            steering_change = abs(current_action[1] - previous_action[1])
            
            # Normalize the changes by their typical ranges
            max_acc_change = 6.0  # Assuming acceleration range is [-3, 3]
            max_steering_change = np.pi/2  # Assuming steering range is [-pi/4, pi/4]
            
            normalized_acc_change = min(acc_change / max_acc_change, 1.0)
            normalized_steering_change = min(steering_change / max_steering_change, 1.0)
            
            # Higher value means smoother action
            return 1.0 - 0.5 * (normalized_acc_change + normalized_steering_change)
        return 0.5  # Default value for unknown action format
    except:
        return 0.5  # Default fallback

def run_highway_env_test(model_name, num_episodes, max_steps, output_dir):
    """
    Run hierarchical MCTS with Highway Environment
    
    Args:
        model_name: Name of the model to use
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        output_dir: Output directory for results
    """
    result_dir = None
    curvature_means = []
    
    # Initialize VLM model
    print(f"Loading LLM model: {model_name}")
    model = Qwen(model_name=model_name)
    
    # Initialize Highway Environment with MCTS wrapper
    env_config = {
        "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": 5,
                "see_behind": True,
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-3.0, 3.0],
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "lanes_count": 5,
        "vehicles_count": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        # "ego_vehicle_type": "highway_env.vehicle.kinematics.Vehicle",
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -20,
        # "reward_speed_range": [20, 30],
        "simulation_frequency": 10,
        'render_mode': 'rgb_array',
        # "initial_speed": 10,
        "show_trajectories": True,
        "render_agent": True,
        # 'record_frames': 10,
        # "vehicles_count": 5,
        "offroad_terminal": True,
        # "duration": 13,  # [s]
        # "destination": "o1",
        # "initial_vehicle_count": 10,
        # "spawn_probability": 0.6,
        # "screen_width": 600,
        # "screen_height": 600,
        # "centering_position": [0.5, 0.6],
        # "scaling": 5.5 * 1.3,
        # "collision_reward": -20,
        # "normalize_reward": False
    }
    
    env = MCTSEnv(env_id='highway-v0', model=model, config=env_config)
    frames = env_config['simulation_frequency']
    
    # Initialize high-level MCTS
    high_level_args = argparse.Namespace(
        seed=42,
        round=0,
        exploration_constant=2.0,
        max_depth=10,
        discount_factor=0.95,
        simulation_num=5
    )
    # high_level_mcts = HighLevelMCTS(model, env=env, args=high_level_args)
    
    # # Connect the environment to the MCTS agents
    # env.connect_to_MCTS(high_level_mcts)
    # print("acc_coef:", env.acc_coef)
    # print("steer_coef:", env.steer_coef)
    # sys.exit(0)
    # Run episodes
    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        # Reset environment
        observation, valid_actions = env.reset(episode=episode, step=0)
        high_level_mcts = HighLevelMCTS(model, env=env, args=high_level_args)
    
        # Connect the environment to the MCTS agents
        env.connect_to_MCTS(high_level_mcts)

        total_reward = 0
        done = False
        previous_action = None
        previous_position = None
        reward_components = {}

        # Initialize result directory for this episode
        result_dir = create_directories(output_dir, episode)
        step = 0
        previous_action = [0, 0]
        metrics_list = []
        # Run episode
        while step < env_config['duration']:
                
            # Plan next action using hierarchical MCTS
            action, metrics = env.plan_next_action(planning_time=0.1, step=step, previous_action=previous_action)
            print("==== Metrics ====")
            print(f"metrics: {metrics}")
            print("==================")
            acceleration = action['acceleration']
            steering = action['steering']
            print(f"action: {acceleration}, {steering}")
            # Execute action
            # print(f"action num_steps: {action['num_steps']}")
            
            
            ego_positions = []

            # # Store the current waypoints for trajectory smoothness calculation
            # if hasattr(env, 'ego_vehicle') and hasattr(env.ego_vehicle, 'position'):
            #     current_position = np.array(env.ego_vehicle.position)
            # else:
            #     current_position = None
                
            # Track each action execution
            for num_step in range(0, action['num_steps'], frames):
                if done:
                    break
                # print(f"Action Step {num_step+1}/{action['num_steps']}")
                
                # Get most probable acc and steering idx
                acc_prob = action['acceleration'][num_step, :]
                steering_prob = action['steering'][num_step, :]
                acc_idx = np.argmax(acc_prob)
                steering_idx = np.argmax(steering_prob)
                
                acc = idx2action(acc_idx, env.acc_target_range, env.acc_coef)
                steering = idx2action(steering_idx, env.steering_target_range, env.steer_coef)
                # Normalize actions
                acc_normalization = inverse_lmap(acc, env.acc_target_range)
                steering_normalization = inverse_lmap(steering, env.steering_target_range)
                action_list = np.array([acc_normalization, steering_normalization], dtype=float)
                # action_list = [acc_normalization, steering_normalization]
                # print("==== Action ====")
                # print(f"Action: {action_list}")
                # print("==================")
                # Calculate action smoothness before executing
                # action_smoothness = calculate_action_smoothness(action_list, previous_action)
                previous_action = action_list
                
                # Perform action in environment
                observation, reward, done, info, valid_actions, ego_position = env.step(action_list, episode=episode, step=step+1)
                total_reward += reward
                ego_positions.append(ego_position)

                # Extract reward components if available
                if hasattr(env, '_calculate_reward'):
                    try:
                        # Try to extract individual reward components if the method is accessible
                        reward_components = {
                            'efficiency_reward': env.reward_weights.get('efficiency', 1.0) * info.get('speed_reward', 0.0),
                            'safety_reward': env.reward_weights.get('safety', 10.0) * info.get('safety_penalty', 0.0),
                            'progress_reward': env.reward_weights.get('progress', 1.0) * info.get('progress_reward', 0.0),
                            'comfort_reward': env.reward_weights.get('comfort', 0.5) * info.get('comfort_reward', 0.0)
                        }
                    except:
                        reward_components = {}
                        
                # Calculate trajectory smoothness if we have position data
                trajectory_smoothness = 1.0  # Default value
                
                # Enhanced step metrics with reward components and smoothness metrics
                step_metrics = {
                    'step': step * action['num_steps'] / 10 + num_step / 10,
                    'reward': float(reward),
                    'total_reward': float(total_reward),
                    'high_level_action': metrics['high_level_action'],
                    'planning_time': metrics['total_time'],
                    'high_level_completed': bool(info['high_level_completed']),
                    # New metrics
                    'trajectory_smoothness': float(trajectory_smoothness),
                    'on_road': bool(getattr(env.ego_vehicle, 'on_road', True)),
                    'vehicle_speed': float(getattr(env.ego_vehicle, 'speed', 0.0) * 3.6),  # Convert m/s to km/h
                    'lane_position': getattr(env.ego_vehicle, 'lane_index', [-1, -1, -1])[2] if hasattr(env.ego_vehicle, 'lane_index') else -1
                }
                
                # Add reward components if available
                step_metrics.update(reward_components)
                
                # Get state for visualization
                images = {}
                # if hasattr(env, 'env') and hasattr(env.env, 'render'):
                #     try:
                #         images['env_render'] = env.env.render(mode='rgb_array')
                #     except:
                #         pass
                        
                # Create results directory
                
                metrics_list.append(step_metrics)
                step += 1


        

            xy_array = np.array(ego_positions)
            xy_array = xy_array.reshape(-1, 2)
            # curvature_mean = calculate_curvature(xy_array[:, 0], xy_array[:, 1])
            # print(f"Curvature mean: {curvature_mean:.4f}")
            # curvature_means.append(curvature_mean)
            
            # Log metrics
            print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"High-level action: {metrics['high_level_action']}")
            print(f"High-level completed: {info['high_level_completed']}")
            
            # End the loop if we're done
            if done:
                env.env.close()
                break

        save_test_results(result_dir, step, metrics_list, images)
        print("======================")
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
        print("======================")
        env.env.close()

    curvature_means = np.array(curvature_means)
    mean_curvature = np.mean(curvature_means)

    print(f"Testing completed. Results saved to {result_dir}")
    print("=====================")
    print(f"Mean curvature: {mean_curvature:.4f}")
    print("=====================")
    env.env.close()

if __name__ == "__main__":
    args = parse_args()
    run_highway_env_test(args.model, args.num_episodes, args.max_steps, args.output_dir)
    visualize_metrics(args.output_dir, args.max_steps)
    print("All episodes completed.")
    
