import sys
import os

# from sympy import Intersection # Not used
import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import cv2 # Not used directly in this refactored version
# import logging # Not used
# import random # Used implicitly by numpy/torch if seeds are set
from datetime import datetime
import torch 

# Import custom modules with absolute imports
from src.MCTS.dqn import DQN, ReplayBuffer 
from src.HighwayEnv.goal_checker import GoalChecker
# from src.HighwayEnv.utils import * # This was empty, can be removed or populated if needed
from src.HighwayEnv.mcts_env import MCTSEnv
from src.MCTS.high_level_mcts import MCTSAgent as HighLevelMCTS
# from src.MCTS.utils import get_vehicle_parameters, highway_lanes_to_path # Not directly used in main
from src.VLM.qwen import Qwen
from src.utils import save_test_results, visualize_metrics, create_directories # Assuming these are general utils
# from src.HighwayEnv.envScenario import EnvScenario # Used within MCTSEnv

# Helper functions (idx2action, normalize_action, inverse_lmap) are not directly used
# in the refactored main training/testing loops as actions are handled by DQN/MCTSEnv.
# They might be useful for analysis or specific action conversion if needed elsewhere.

def parse_args():
    parser = argparse.ArgumentParser(description="Run MCTS-based VLM Numerical Prediction Planner")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='VLM Model name for high-level policy and mid-level guidance')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max_steps_per_episode', type=int, default=200, help='Maximum high-level steps per episode')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for results')
    
    # Arguments for DQN
    parser.add_argument('--train_dqn', default=True, help='Train the DQN low-level planner')
    parser.add_argument('--test_dqn', default=False, help='Test the trained DQN low-level planner')
    parser.add_argument('--dqn_model_load_path', type=str, default=None, help='Path to load a pre-trained DQN model for testing or continued training')
    parser.add_argument('--dqn_model_save_path', type=str, default='./model/dqn_planner.pth', help='Path to save the trained DQN model')
    parser.add_argument('--dqn_save_model_frequency', type=int, default=10, help='Save DQN model every N episodes during training')

    # DQN Hyperparameters
    # Many of these will be passed to MCTSEnv's config
    parser.add_argument('--dqn_state_dim', type=int, default=None, help='State dimension for DQN (if not inferred from env)') # MCTSEnv infers this
    parser.add_argument('--dqn_hidden_dim', type=int, default=256, help='Hidden dimension for DQN Q-network')
    parser.add_argument('--num_throttle_bins', type=int, default=13, help='Number of discrete bins for throttle')
    parser.add_argument('--num_steering_bins', type=int, default=13, help='Number of discrete bins for steering')
    parser.add_argument('--dqn_learning_rate', type=float, default=1e-4, help='Learning rate for DQN optimizer')
    parser.add_argument('--dqn_gamma', type=float, default=0.99, help='Discount factor for DQN')
    parser.add_argument('--dqn_epsilon_start', type=float, default=0.9, help='DQN Epsilon start value for training')
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.05, help='DQN Epsilon end value for training')
    parser.add_argument('--dqn_epsilon_decay', type=float, default=0.995, help='DQN Epsilon decay rate per episode')
    parser.add_argument('--dqn_target_update_frequency', type=int, default=20, help='Frequency (in low-level steps) to update DQN target network')
    parser.add_argument('--dqn_replay_buffer_capacity', type=int, default=50000, help='Capacity of DQN replay buffer')
    parser.add_argument('--dqn_batch_size', type=int, default=128, help='Batch size for DQN training')
    parser.add_argument('--dqn_sub_steps', type=int, default=10, help='Number of DQN low-level steps per high-level environment step')

    parser.add_argument('--use_llm_guidance_for_dqn', default=True, help="Whether DQN uses LLM mid-level guidance")
    parser.add_argument('--dqn_p_follow_llm', type=float, default=0.3, help="Probability for DQN to follow LLM guidance if active")
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--render_env', default=True, help="Render the environment during execution (can slow down training)")


    return parser.parse_args()

def setup_environment(args, model_vlm):
    """Initializes and configures the MCTSEnv."""
    device_str = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    env_config = {
        # HighwayEnv specific configurations (can be overridden by args if needed)
        "observation": {
            "type": "Kinematics", "vehicles_count": 5, 
            "features": ["presence", "x", "y", "vx", "vy", "heading"], # Added heading
            "absolute": True, "normalize": False, "see_behind": True, "order": "sorted"
        },
        "action": { # For the base highway-env
            "type": "ContinuousAction",
            "acceleration_range": [-3.0, 3.0], # DQN will output normalized actions, env handles scaling
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "lanes_count": 3, "vehicles_count": 10, # Reduced for potentially faster simulation
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": args.max_steps_per_episode * args.dqn_sub_steps + 100, # Env's internal max duration, main loop controls HL steps
        "initial_spacing": 2, "collision_reward": -20, 
        "simulation_frequency": 15, # Higher sim frequency for smoother physics if policy is slower
        "policy_frequency": 1, # This is for the environment's internal controller if used, MCTS/DQN have their own logic
        'render_mode': 'rgb_array', # Always rgb_array for video recording
        "offscreen_rendering": not args.render_env, # If not rendering to screen, use offscreen
        "manual_control": False,
        "real_time_rendering": args.render_env,


        # DQN configurations to be picked up by MCTSEnv
        "dqn_config": {
            "state_dim": args.dqn_state_dim, # MCTSEnv will try to infer if None
            "hidden_dim": args.dqn_hidden_dim,
            "learning_rate": args.dqn_learning_rate,
            "gamma": args.dqn_gamma,
            "epsilon": args.dqn_epsilon_start,
            "epsilon_decay": args.dqn_epsilon_decay, # MCTSEnv handles inter-episode decay
            "min_epsilon": args.dqn_epsilon_end,
            "target_update": args.dqn_target_update_frequency,
            "device": device_str,
            "buffer_capacity": args.dqn_replay_buffer_capacity,
            "batch_size": args.dqn_batch_size,
            "num_throttle_bins": args.num_throttle_bins,
            "num_steering_bins": args.num_steering_bins,
            "p_follow_llm": args.dqn_p_follow_llm,
        },
        "DQN_SUB_STEPS": args.dqn_sub_steps,
        "USE_LLM_FOR_DQN_GUIDANCE": args.use_llm_guidance_for_dqn,
        "seed": args.seed,
    }
    
    env = MCTSEnv(env_id='highway-v0', model=model_vlm, config=env_config)
    
    # Set seed for reproducibility
    # env.env.reset(seed=args.seed) # Reset the underlying gym env with seed
    # np.random.seed(args.seed) # Numpy seed
    # torch.manual_seed(args.seed) # PyTorch seed
    # if device_str == "cuda":
    #     torch.cuda.manual_seed_all(args.seed)
        
    return env

def train_dqn_planner(args, model_vlm):
    print("Starting DQN Low-Level Planner Training...")
    env = setup_environment(args, model_vlm)
    
    if not hasattr(env, 'dqn_agent') or not isinstance(env.dqn_agent, DQN):
        print("Error: MCTSEnv not correctly initialized with a DQN agent.")
        env.close()
        return

    if args.dqn_model_load_path and os.path.exists(args.dqn_model_load_path):
        try:
            env.dqn_agent.q_net.load_state_dict(torch.load(args.dqn_model_load_path, map_location=env.dqn_config["device"]))
            env.dqn_agent.target_q_net.load_state_dict(torch.load(args.dqn_model_load_path, map_location=env.dqn_config["device"]))
            print(f"Loaded DQN model from {args.dqn_model_load_path} for continued training.")
        except Exception as e:
            print(f"Could not load DQN model: {e}. Starting from scratch.")

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(args.dqn_model_save_path), exist_ok=True)

    episode_rewards_hl = []
    # Epsilon is managed by MCTSEnv.reset() based on its dqn_config
    # We can set the initial epsilon for the agent here if needed, but MCTSEnv should handle it.
    env.dqn_agent.epsilon = args.dqn_epsilon_start # Explicitly set initial epsilon for the agent

    for episode in tqdm.trange(args.num_episodes, desc="Training DQN Episodes"):
        hl_obs_text, _ = env.reset(episode=episode, step=0, seed=args.seed) # Seed per episode
        
        # High-level MCTS for this episode
        high_level_args_ns = argparse.Namespace(
            seed=args.seed, round=0, exploration_constant=1.5, # Adjusted exploration
            max_depth=5, discount_factor=0.9, simulation_num=3 # Reduced for faster training iterations
        )
        high_level_mcts = HighLevelMCTS(model_vlm, env=env, args=high_level_args_ns, use_llm=True) # Assuming VLM is always used for HL
        env.connect_to_MCTS(high_level_mcts)

        current_episode_reward_hl = 0
        
        for hl_step in range(args.max_steps_per_episode):
            if args.render_env: env.render(mode='rgb_array') # Optional rendering

            next_hl_obs_text, hl_reward_accum, hl_done, info = env.execute_policy_step(
                episode=episode, hl_step_num=hl_step
            )
            
            current_episode_reward_hl += hl_reward_accum
            hl_obs_text = next_hl_obs_text

            if hl_done:
                break
        
        episode_rewards_hl.append(current_episode_reward_hl)
        # Epsilon decay is handled by env.reset() for the next episode
        
        if (episode + 1) % args.dqn_save_model_frequency == 0:
            save_path = f"{os.path.splitext(args.dqn_model_save_path)[0]}_ep{episode+1}.pth"
            torch.save(env.dqn_agent.q_net.state_dict(), save_path)
            print(f"DQN model saved to {save_path}")

        tqdm.tqdm.write(f"Episode {episode+1}: HL Reward: {current_episode_reward_hl:.2f}, Epsilon: {env.dqn_agent.epsilon:.3f}, Total LL Steps: {env.timestep}, DQN Updates: {env.dqn_agent.count}")

    final_save_path = args.dqn_model_save_path
    torch.save(env.dqn_agent.q_net.state_dict(), final_save_path)
    print(f"Final DQN model saved to {final_save_path}")
    
    env.close()
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards_hl)
    plt.title("High-Level Episode Rewards during DQN Training")
    plt.xlabel("Episode")
    plt.ylabel("Total High-Level Reward")
    plt.savefig(os.path.join(os.path.dirname(args.dqn_model_save_path), "dqn_training_hl_rewards.png"))
    plt.close()
    print("Training finished.")


def test_dqn_planner(args, model_vlm):
    print("Starting DQN Low-Level Planner Testing...")
    env = setup_environment(args, model_vlm) # Uses args for config

    if not hasattr(env, 'dqn_agent') or not isinstance(env.dqn_agent, DQN):
        print("Error: MCTSEnv not correctly initialized with a DQN agent for testing.")
        env.close()
        return
        
    model_path_to_load = args.dqn_model_load_path if args.dqn_model_load_path else args.dqn_model_save_path
    if os.path.exists(model_path_to_load):
        env.dqn_agent.q_net.load_state_dict(torch.load(model_path_to_load, map_location=env.dqn_config["device"]))
        print(f"DQN model loaded from {model_path_to_load}")
    else:
        print(f"Error: DQN model not found at {model_path_to_load}. Cannot test.")
        env.close()
        return

    env.dqn_agent.q_net.eval() # Set DQN to evaluation mode
    env.dqn_agent.epsilon = args.dqn_epsilon_end # Use min epsilon for testing (more deterministic)

    test_episode_rewards_hl = []
    all_episodes_metrics_lists = []

    for episode in tqdm.trange(args.num_episodes, desc="Testing DQN Episodes"):
        hl_obs_text, _ = env.reset(episode=episode, step=0, seed=args.seed) # Different seed for test
        # High-level MCTS for this episode
        # Using the same seed for HL MCTS as the episode seed for consistency
        high_level_args_ns = argparse.Namespace(
            seed=args.seed, round=0, exploration_constant=1.0, # Lower exploration for testing MCTS
            max_depth=5, discount_factor=0.9, simulation_num=3
        )
        high_level_mcts = HighLevelMCTS(model_vlm, env=env, args=high_level_args_ns, use_llm=True)
        env.connect_to_MCTS(high_level_mcts)

        current_episode_reward_hl = 0
        episode_metrics_list = [] 

        for hl_step in range(args.max_steps_per_episode):
            if args.render_env: env.render(mode='rgb_array')

            next_hl_obs_text, hl_reward_accum, hl_done, info = env.execute_policy_step(
                episode=episode, hl_step_num=hl_step
            )
            
            current_episode_reward_hl += hl_reward_accum
            hl_obs_text = next_hl_obs_text
            
            step_metric_detail = {
                'hl_step': hl_step, 'hl_reward_accum': hl_reward_accum,
                'total_hl_reward_episode': current_episode_reward_hl,
                'high_level_action': info.get('high_level_action', 'N/A'),
                'high_level_completed': bool(info.get('high_level_completed', False)),
                'dqn_steps_in_hl_step': info.get('dqn_steps_executed', args.dqn_sub_steps),
                'ego_speed': info.get('ego_speed', 0.0),
                'on_road': info.get('on_road', True),
                'planning_time_total': info.get('total_time', 0.0),
            }
            episode_metrics_list.append(step_metric_detail)

            if hl_done:
                break
        
        test_episode_rewards_hl.append(current_episode_reward_hl)
        all_episodes_metrics_lists.append(episode_metrics_list)
        tqdm.tqdm.write(f"Test Episode {episode+1}: HL Reward: {current_episode_reward_hl:.2f}")
        
        # Save per-episode results
        result_dir_ep = create_directories(args.output_dir, f"test_dqn_ep_{episode}")
        save_test_results(result_dir_ep, hl_step + 1, episode_metrics_list, {}) 

    avg_reward_hl = np.mean(test_episode_rewards_hl) if test_episode_rewards_hl else 0
    print(f"Average High-Level Reward over {args.num_episodes} test episodes: {avg_reward_hl:.2f}")
    
    env.close()

    plt.figure(figsize=(10,5))
    plt.plot(test_episode_rewards_hl)
    plt.title("High-Level Episode Rewards during DQN Testing")
    plt.xlabel("Episode")
    plt.ylabel("Total High-Level Reward")
    plt.savefig(os.path.join(args.output_dir, "dqn_testing_hl_rewards.png"))
    plt.close()
    print("Testing finished. Detailed metrics saved in output directory.")


def run_original_highway_env_test_adapted(args, model_vlm):
    """
    Adapted version of the original test function, now using the DQN-integrated MCTSEnv.
    This serves as a general test run if not specifically training or testing DQN.
    """
    print(f"Running adapted original test with DQN-powered MCTSEnv.")
    env = setup_environment(args, model_vlm) # Uses args for config
    
    # If a DQN model is specified, load it for this test run, otherwise DQN uses its initial weights.
    if args.dqn_model_load_path and os.path.exists(args.dqn_model_load_path):
        try:
            env.dqn_agent.q_net.load_state_dict(torch.load(args.dqn_model_load_path, map_location=env.dqn_config["device"]))
            print(f"Loaded DQN model from {args.dqn_model_load_path} for adapted test.")
            env.dqn_agent.q_net.eval() # Set to eval mode
            env.dqn_agent.epsilon = args.dqn_epsilon_end # Use low epsilon
        except Exception as e:
            print(f"Could not load DQN model for adapted test: {e}. DQN will use initial/random weights.")
    else:
        print("No DQN model specified or found for adapted test. DQN will use initial/random weights.")
        env.dqn_agent.epsilon = args.dqn_epsilon_start # Or some other default exploration

    all_episodes_rewards_hl = []
    
    for episode in tqdm.trange(args.num_episodes, desc="Adapted Original Test Episodes"):
        hl_obs_text, _ = env.reset(episode=episode, step=0, seed=args.seed)
        
        high_level_args_ns = argparse.Namespace(
            seed=args.seed, round=0, exploration_constant=1.0, 
            max_depth=5, discount_factor=0.9, simulation_num=3
        )
        high_level_mcts = HighLevelMCTS(model_vlm, env=env, args=high_level_args_ns, use_llm=True)
        env.connect_to_MCTS(high_level_mcts)

        current_episode_reward_hl = 0
        episode_metrics_list = []
        result_dir_ep = create_directories(args.output_dir, f"adapted_test_ep_{episode}")

        for hl_step in range(args.max_steps_per_episode):
            if args.render_env: env.render(mode='rgb_array')

            next_hl_obs_text, hl_reward_accum, hl_done, info = env.execute_policy_step(
                episode=episode, hl_step_num=hl_step
            )
            
            current_episode_reward_hl += hl_reward_accum
            hl_obs_text = next_hl_obs_text
            
            step_metric_detail = {
                'hl_step': hl_step, 'hl_reward_accum': hl_reward_accum,
                'total_hl_reward_episode': current_episode_reward_hl,
                'high_level_action': info.get('high_level_action', 'N/A'),
                'high_level_completed': bool(info.get('high_level_completed', False)),
                'dqn_steps_in_hl_step': info.get('dqn_steps_executed', args.dqn_sub_steps),
                'ego_speed': info.get('ego_speed', 0.0),
                'on_road': info.get('on_road', True),
                'planning_time_total': info.get('total_time', 0.0),
            }
            episode_metrics_list.append(step_metric_detail)

            if hl_done:
                break
        
        all_episodes_rewards_hl.append(current_episode_reward_hl)
        tqdm.tqdm.write(f"Adapted Test Episode {episode+1}: HL Reward: {current_episode_reward_hl:.2f}")
        save_test_results(result_dir_ep, hl_step + 1, episode_metrics_list, {})

    avg_reward_hl = np.mean(all_episodes_rewards_hl) if all_episodes_rewards_hl else 0
    print(f"Average High-Level Reward over {args.num_episodes} adapted test episodes: {avg_reward_hl:.2f}")
    
    env.close()
    
    plt.figure(figsize=(10,5))
    plt.plot(all_episodes_rewards_hl)
    plt.title("High-Level Episode Rewards during Adapted Original Test")
    plt.xlabel("Episode")
    plt.ylabel("Total High-Level Reward")
    plt.savefig(os.path.join(args.output_dir, "adapted_test_hl_rewards.png"))
    plt.close()
    print("Adapted original test finished. Detailed metrics saved.")


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize VLM model once
    # VLM model is only needed if high-level MCTS uses LLM or if DQN uses LLM guidance.
    vlm_model = None
    if args.use_llm_guidance_for_dqn or True: # Assuming HL MCTS always uses VLM for now
        print(f"Loading VLM model: {args.model}")
        vlm_model = Qwen(model_name=args.model)
        print("VLM model loaded.")
    else:
        print("VLM model not loaded as per configuration (no LLM guidance for DQN, HL MCTS might not need it).")


    if args.train_dqn:
        train_dqn_planner(args, vlm_model)
    
    if args.test_dqn:
        if not args.dqn_model_load_path and not args.train_dqn: # If testing without training first
            print("Warning: Testing DQN without specifying a model to load (--dqn_model_load_path) and without prior training in this run.")
            print(f"Attempting to load from default save path: {args.dqn_model_save_path}")
        test_dqn_planner(args, vlm_model)
    
    # Run the adapted original test if no specific DQN task is requested
    if not args.train_dqn and not args.test_dqn:
        print("No DQN training/testing task specified. Running the adapted general test.")
        run_original_highway_env_test_adapted(args, vlm_model)
    
    # Visualization of metrics is handled by save_test_results within each test/train function's episode loop.
    # General summary plots are saved at the end of train_dqn_planner and test_dqn_planner.

    print("All requested operations completed.")

