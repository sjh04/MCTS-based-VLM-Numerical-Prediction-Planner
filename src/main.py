import sys
import os

from sympy import Intersection
import tqdm

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
import torch # Added
from src.MCTS.dqn import DQN, ReplayBuffer # Added, assuming DQN is in src.MCTS

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
    parser.add_argument('--max_steps_per_episode', type=int, default=200, help='Maximum high-level steps per episode') # Renamed from max_steps
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for results')
    
    # Arguments for training/testing DQN
    parser.add_argument('--train_dqn', action='store_true', help='Train the DQN low-level planner')
    parser.add_argument('--test_dqn', action='store_true', help='Test the trained DQN low-level planner')
    parser.add_argument('--dqn_model_load_path', type=str, default=None, help='Path to load a pre-trained DQN model for testing or continued training')

    # DQN Hyperparameters (add more as needed from your DQN/MCTSEnv implementation)
    parser.add_argument('--dqn_state_dim', type=int, default=14, help='State dimension for DQN') # Example, adjust
    parser.add_argument('--dqn_hidden_dim', type=int, default=128, help='Hidden dimension for DQN Q-network')
    parser.add_argument('--num_throttle_bins', type=int, default=5, help='Number of discrete bins for throttle')
    parser.add_argument('--num_steering_bins', type=int, default=5, help='Number of discrete bins for steering')
    parser.add_argument('--dqn_learning_rate', type=float, default=1e-4, help='Learning rate for DQN optimizer')
    parser.add_argument('--dqn_gamma', type=float, default=0.99, help='Discount factor for DQN')
    parser.add_argument('--dqn_epsilon_start', type=float, default=1.0, help='DQN Epsilon start value')
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.01, help='DQN Epsilon end value')
    parser.add_argument('--dqn_epsilon_decay', type=float, default=0.995, help='DQN Epsilon decay rate')
    parser.add_argument('--dqn_target_update_frequency', type=int, default=100, help='Frequency (in DQN steps) to update target network') # This is DQN internal count
    parser.add_argument('--dqn_replay_buffer_capacity', type=int, default=50000, help='Capacity of DQN replay buffer')
    parser.add_argument('--dqn_batch_size', type=int, default=64, help='Batch size for DQN training')
    parser.add_argument('--dqn_sub_steps', type=int, default=10, help='Number of DQN steps per high-level environment step')
    parser.add_argument('--dqn_model_save_path', type=str, default='./model/dqn_planner.pth', help='Path to save the trained DQN model')
    parser.add_argument('--dqn_save_model_frequency', type=int, default=50, help='Save DQN model every N episodes')
    parser.add_argument('--dqn_use_llm_guidance', type=bool, default=False, help="Whether DQN uses LLM guidance")
    parser.add_argument('--dqn_p_follow_llm', type=float, default=0.3, help="Probability for DQN to follow LLM guidance")


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

def run_highway_env_test(model_name, num_episodes, max_steps_per_episode, output_dir):
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

def train_dqn_planner(args, model_vlm):
    """
    Trains the DQN low-level planner integrated within MCTSEnv.
    """
    print("Starting DQN Low-Level Planner Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare env_config by merging args
    env_config_dict = {
        "observation": {"type": "Kinematics", "vehicles_count": 5, "features": ["presence", "x", "y", "vx", "vy"], "absolute": True, "normalize": False, "see_behind": True},
        "action": {"type": "ContinuousAction", "acceleration_range": [-3.0, 3.0], "steering_range": [-np.pi/4, np.pi/4]},
        "lanes_count": 5, "vehicles_count": 5, "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": args.max_steps_per_episode * args.dqn_sub_steps, # Approximate total low-level steps
        "initial_spacing": 2, "collision_reward": -20, "simulation_frequency": 10, 'render_mode': 'rgb_array',
        "show_trajectories": True, "render_agent": True, "offroad_terminal": True,
        # DQN specific params to be picked up by MCTSEnv
        "dqn_state_dim": args.dqn_state_dim,
        "dqn_hidden_dim": args.dqn_hidden_dim,
        "num_throttle_bins": args.num_throttle_bins,
        "num_steering_bins": args.num_steering_bins,
        "dqn_learning_rate": args.dqn_learning_rate,
        "dqn_gamma": args.dqn_gamma,
        "dqn_target_update": args.dqn_target_update_frequency, # Renamed for clarity
        "dqn_replay_buffer_capacity": args.dqn_replay_buffer_capacity,
        "dqn_batch_size": args.dqn_batch_size,
        "device": device.type, # Pass device type as string or handle torch.device in MCTSEnv
        "DQN_SUB_STEPS": args.dqn_sub_steps,
        "USE_LLM_FOR_DQN_GUIDANCE": args.dqn_use_llm_guidance,
        "DQN_P_FOLLOW_LLM": args.dqn_p_follow_llm,
    }

    env = MCTSEnv(env_id='highway-v0', model=model_vlm, config=env_config_dict)
    
    if not hasattr(env, 'dqn_agent') or not isinstance(env.dqn_agent, DQN):
        print("Error: MCTSEnv not correctly initialized with a DQN agent.")
        env.env.close() # Close the underlying gym env
        return

    if args.dqn_model_load_path and os.path.exists(args.dqn_model_load_path):
        try:
            env.dqn_agent.q_net.load_state_dict(torch.load(args.dqn_model_load_path, map_location=device))
            env.dqn_agent.target_q_net.load_state_dict(torch.load(args.dqn_model_load_path, map_location=device))
            print(f"Loaded DQN model from {args.dqn_model_load_path}")
        except Exception as e:
            print(f"Could not load DQN model: {e}")

    current_epsilon = args.dqn_epsilon_start
    episode_rewards_hl = []

    if not os.path.exists(os.path.dirname(args.dqn_model_save_path)):
        os.makedirs(os.path.dirname(args.dqn_model_save_path), exist_ok=True)

    for episode in tqdm(range(args.num_episodes), desc="Training DQN Episodes"):
        hl_obs, _ = env.reset(episode=episode, step=0) # MCTSEnv reset
        env.dqn_agent.epsilon = current_epsilon
        
        # High-level MCTS for this episode (re-init or ensure it's reset)
        high_level_args_ns = argparse.Namespace(seed=42, round=0, exploration_constant=2.0, max_depth=10, discount_factor=0.95, simulation_num=5)
        high_level_mcts = HighLevelMCTS(model_vlm, env=env, args=high_level_args_ns)
        env.connect_to_MCTS(high_level_mcts) # Connect MCTS agent to env

        current_episode_reward_hl = 0
        
        for hl_step in range(args.max_steps_per_episode):
            # MCTSEnv's execute_policy_step (or similar) will now:
            # 1. Get high-level action (using its self.high_level_mcts)
            # 2. Get LLM guidance for DQN (if enabled)
            # 3. Run DQN for args.dqn_sub_steps, updating DQN model internally
            # This function needs to be implemented in MCTSEnv
            if not hasattr(env, 'execute_policy_step'):
                print("ERROR: MCTSEnv must have an 'execute_policy_step' method for DQN training.")
                env.env.close()
                return

            next_hl_obs, hl_reward, hl_done, info = env.execute_policy_step()
            
            current_episode_reward_hl += hl_reward
            hl_obs = next_hl_obs

            if hl_done:
                break
        
        episode_rewards_hl.append(current_episode_reward_hl)
        current_epsilon = max(args.dqn_epsilon_end, current_epsilon * args.dqn_epsilon_decay)
        
        if (episode + 1) % args.dqn_save_model_frequency == 0:
            torch.save(env.dqn_agent.q_net.state_dict(), f"{args.dqn_model_save_path}_ep{episode+1}.pth")
            print(f"DQN model saved at episode {episode+1}")

        tqdm.write(f"Episode {episode+1}: High-Level Reward: {current_episode_reward_hl:.2f}, Epsilon: {current_epsilon:.3f}, DQN Steps: {env.dqn_agent.count}")

    torch.save(env.dqn_agent.q_net.state_dict(), args.dqn_model_save_path)
    print(f"Final DQN model saved to {args.dqn_model_save_path}")
    
    env.env.close() # Close the underlying gym env
    # Optional: Plot high-level rewards
    plt.figure()
    plt.plot(episode_rewards_hl)
    plt.title("High-Level Episode Rewards during DQN Training")
    plt.xlabel("Episode")
    plt.ylabel("Total High-Level Reward")
    plt.savefig(os.path.join(os.path.dirname(args.dqn_model_save_path), "dqn_training_hl_rewards.png"))
    # plt.show()


def test_dqn_planner(args, model_vlm):
    """
    Tests the trained DQN low-level motion planner.
    """
    print("Starting DQN Low-Level Planner Testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_config_dict = {
        "observation": {"type": "Kinematics", "vehicles_count": 5, "features": ["presence", "x", "y", "vx", "vy"], "absolute": True, "normalize": False, "see_behind": True},
        "action": {"type": "ContinuousAction", "acceleration_range": [-3.0, 3.0], "steering_range": [-np.pi/4, np.pi/4]},
        "lanes_count": 5, "vehicles_count": 5, "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": args.max_steps_per_episode * args.dqn_sub_steps, 
        "initial_spacing": 2, "collision_reward": -20, "simulation_frequency": 10, 'render_mode': 'rgb_array',
        "show_trajectories": True, "render_agent": True, "offroad_terminal": True,
        "dqn_state_dim": args.dqn_state_dim, "dqn_hidden_dim": args.dqn_hidden_dim,
        "num_throttle_bins": args.num_throttle_bins, "num_steering_bins": args.num_steering_bins,
        "device": device.type, "DQN_SUB_STEPS": args.dqn_sub_steps,
        "USE_LLM_FOR_DQN_GUIDANCE": args.dqn_use_llm_guidance, # LLM guidance can also be used in testing
        "DQN_P_FOLLOW_LLM": args.dqn_p_follow_llm,
    }
    env = MCTSEnv(env_id='highway-v0', model=model_vlm, config=env_config_dict)

    if not hasattr(env, 'dqn_agent') or not isinstance(env.dqn_agent, DQN):
        print("Error: MCTSEnv not correctly initialized with a DQN agent for testing.")
        env.env.close()
        return
        
    model_path = args.dqn_model_load_path if args.dqn_model_load_path else args.dqn_model_save_path
    if os.path.exists(model_path):
        env.dqn_agent.q_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"DQN model loaded from {model_path}")
    else:
        print(f"Error: DQN model not found at {model_path}")
        env.env.close()
        return

    env.dqn_agent.q_net.eval()
    env.dqn_agent.epsilon = 0.01 # Low epsilon for testing

    test_episode_rewards_hl = []
    all_metrics_lists = [] # To store metrics from all episodes for overall report

    for episode in tqdm(range(args.num_episodes), desc="Testing DQN Episodes"):
        hl_obs, _ = env.reset(episode=episode, step=0)
        
        high_level_args_ns = argparse.Namespace(seed=42, round=0, exploration_constant=2.0, max_depth=10, discount_factor=0.95, simulation_num=5)
        high_level_mcts = HighLevelMCTS(model_vlm, env=env, args=high_level_args_ns)
        env.connect_to_MCTS(high_level_mcts)

        current_episode_reward_hl = 0
        # episode_metrics_list = [] # Metrics for the current episode

        for hl_step in range(args.max_steps_per_episode):
            if not hasattr(env, 'execute_policy_step'):
                print("ERROR: MCTSEnv must have an 'execute_policy_step' method.")
                env.env.close()
                return
            
            next_hl_obs, hl_reward, hl_done, info = env.execute_policy_step()
            
            current_episode_reward_hl += hl_reward
            hl_obs = next_hl_obs
            
            # step_metric_detail = {
            #     'hl_step': hl_step, 'hl_reward': hl_reward, 'hl_action': info.get('high_level_action', 'N/A'),
            #     'hl_completed': info.get('high_level_completed', False),
            #     # Add more detailed metrics from info if MCTSEnv provides them
            # }
            # episode_metrics_list.append(step_metric_detail)

            if hl_done:
                break
        
        test_episode_rewards_hl.append(current_episode_reward_hl)
        # all_metrics_lists.append(episode_metrics_list)
        tqdm.write(f"Test Episode {episode+1}: High-Level Reward: {current_episode_reward_hl:.2f}")
        
        # Save per-episode results if needed (using existing save_test_results logic)
        # result_dir_ep = create_directories(args.output_dir, f"test_dqn_ep_{episode}")
        # save_test_results(result_dir_ep, hl_step + 1, episode_metrics_list, {}) # No images for now

    avg_reward_hl = np.mean(test_episode_rewards_hl) if test_episode_rewards_hl else 0
    print(f"Average High-Level Reward over {args.num_episodes} test episodes: {avg_reward_hl:.2f}")
    
    env.env.close()
    # Plotting test rewards
    plt.figure()
    plt.plot(test_episode_rewards_hl)
    plt.title("High-Level Episode Rewards during DQN Testing")
    plt.xlabel("Episode")
    plt.ylabel("Total High-Level Reward")
    plt.savefig(os.path.join(args.output_dir, "dqn_testing_hl_rewards.png"))


def run_highway_env_test(model_name, num_episodes, max_steps_per_episode, output_dir, args_all): # Added args_all
    """
    Run hierarchical MCTS with Highway Environment (Original test function, adapted slightly)
    This function will now use the DQN-based low-level planner if MCTSEnv is modified.
    """
    print(f"Running original run_highway_env_test with potentially DQN-powered MCTSEnv.")
    result_dir = None
    # curvature_means = # This was commented out, keep as is or re-enable if needed
    
    model = Qwen(model_name=model_name)
    
    # Use args_all to pass DQN parameters if MCTSEnv expects them
    env_config_dict = {
        "observation": {"type": "Kinematics", "features": ["presence", "x", "y", "vx", "vy"], "absolute": True, "normalize": False, "vehicles_count": 5, "see_behind": True},
        "action": {"type": "ContinuousAction", "acceleration_range": [-3.0, 3.0], "steering_range": [-np.pi/4, np.pi/4]},
        "lanes_count": 5, "vehicles_count": 5, "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": max_steps_per_episode * getattr(args_all, 'dqn_sub_steps', 10), # Adjust duration based on sub_steps
        "initial_spacing": 2, "collision_reward": -20, "simulation_frequency": 10, 'render_mode': 'rgb_array',
        "show_trajectories": True, "render_agent": True, "offroad_terminal": True,
        # Pass relevant args from args_all to env_config_dict for MCTSEnv
        "dqn_state_dim": args_all.dqn_state_dim,
        "dqn_hidden_dim": args_all.dqn_hidden_dim,
        "num_throttle_bins": args_all.num_throttle_bins,
        "num_steering_bins": args_all.num_steering_bins,
        "device": "cuda" if torch.cuda.is_available() else "cpu", # Simplified device passing
        "DQN_SUB_STEPS": args_all.dqn_sub_steps,
        "USE_LLM_FOR_DQN_GUIDANCE": args_all.dqn_use_llm_guidance,
        "DQN_P_FOLLOW_LLM": args_all.dqn_p_follow_llm,
         # Ensure MCTSEnv's __init__ can handle these if it's the DQN version
    }

    env = MCTSEnv(env_id='highway-v0', model=model, config=env_config_dict)
    # frames = env_config_dict['simulation_frequency'] # Not directly used in the new loop

    high_level_args_ns = argparse.Namespace(seed=42, round=0, exploration_constant=2.0, max_depth=10, discount_factor=0.95, simulation_num=5)
    
    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes} in run_highway_env_test")
        hl_obs, _ = env.reset(episode=episode, step=0)
        
        # Re-initialize or ensure MCTS agent is fresh for the episode
        high_level_mcts = HighLevelMCTS(model, env=env, args=high_level_args_ns)
        env.connect_to_MCTS(high_level_mcts) # Connect MCTS agent to env

        total_reward_hl = 0
        # done_hl = False # Renamed from done
        metrics_list = []
        result_dir = create_directories(output_dir, f"original_test_ep_{episode}")

        for hl_step_count in range(max_steps_per_episode):
            if not hasattr(env, 'execute_policy_step'):
                print("ERROR: MCTSEnv must have an 'execute_policy_step' method for this test flow.")
                env.env.close()
                return

            # This call now encapsulates high-level planning and low-level DQN execution
            next_hl_obs, hl_reward, hl_done, info = env.execute_policy_step()
            
            total_reward_hl += hl_reward
            
            # Log metrics (simplified for this example, adapt as needed)
            step_metrics = {
                'hl_step': hl_step_count,
                'hl_reward': float(hl_reward),
                'total_hl_reward': float(total_reward_hl),
                'high_level_action': info.get('high_level_action', 'N/A'),
                'high_level_completed': bool(info.get('high_level_completed', False)),
                'dqn_steps_taken_in_hl_step': info.get('dqn_steps_executed', args_all.dqn_sub_steps), # Assuming info provides this
                'on_road': bool(getattr(env.ego_vehicle, 'on_road', True)),
                'vehicle_speed_kmh': float(getattr(env.ego_vehicle, 'speed', 0.0) * 3.6),
            }
            metrics_list.append(step_metrics)
            print(f"HL Step {hl_step_count}, HL Reward: {hl_reward:.2f}, Total HL Reward: {total_reward_hl:.2f}, HL Action: {info.get('high_level_action', 'N/A')}")

            hl_obs = next_hl_obs
            if hl_done:
                print(f"Episode {episode+1} finished early at HL step {hl_step_count}.")
                break
        
        save_test_results(result_dir, hl_step_count + 1, metrics_list, {}) # No images for now
        print(f"Episode {episode+1} finished with total HL reward: {total_reward_hl:.2f}")
        env.env.close() # Close after each episode

    print(f"Original highway_env_test completed. Results saved to {output_dir}")
    env.env.close() # Final close


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize VLM model once if used by multiple functions
    vlm_model = Qwen(model_name=args.model)

    if args.train_dqn:
        train_dqn_planner(args, vlm_model)
    
    if args.test_dqn:
        if not args.dqn_model_load_path and not args.train_dqn:
            print("Warning: Testing DQN without specifying a model to load (--dqn_model_load_path) or training it first.")
            print(f"Attempting to load from default save path: {args.dqn_model_save_path}")
        test_dqn_planner(args, vlm_model)
    
    # Decide whether to run the original test based on some condition or if no DQN tasks are specified
    if not args.train_dqn and not args.test_dqn:
        print("No DQN specific task requested, running original run_highway_env_test.")
        # The original test will use the MCTSEnv, which should now be the DQN-integrated one.
        run_highway_env_test(args.model, args.num_episodes, args.max_steps_per_episode, args.output_dir, args)
    
    # Visualize metrics if output_dir is standard
    if os.path.exists(args.output_dir) and (not args.train_dqn and not args.test_dqn) : # Only run for original test for now
         visualize_metrics(args.output_dir, args.max_steps_per_episode) # max_steps might need adjustment depending on what visualize_metrics expects
    elif args.train_dqn or args.test_dqn:
        print(f"Metrics for DQN training/testing are saved as plots (e.g., dqn_training_hl_rewards.png).")
        print(f"Detailed per-step metrics for original run_highway_env_test (if run) are in {args.output_dir}/<ep_num>/results.json")


    print("All requested operations completed.")

