"""
Evaluation script for Double DQN + PER Atari Boxing agent.
Runs trained policy without learning and reports performance.
"""

import numpy as np
import torch
import gymnasium as gym

from config import Config
from env import make_env
from agent import DoubleDQNAgent


def evaluate(model_path, n_episodes=10, render=False, record=False):
    """
    Evaluate a trained Double DQN agent.

    Args:
        model_path: str, path to saved model weights
        n_episodes: int, number of evaluation episodes
        render: bool, render environment live
        record: bool, record video (optional)

    Returns:
        List of episode rewards
    """

    config = Config()

    # ----- Environment -----
    env, preprocessor, stacker = make_env(
        config.ENV_NAME,
        config.FRAME_SIZE,
        config.FRAME_STACK
    )

    # Optional video recording
    if record:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos/",
            episode_trigger=lambda ep: True
        )

    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Agent -----
    agent = DoubleDQNAgent(
        n_frames=config.FRAME_STACK,
        n_actions=n_actions,
        config=config,
        device=device
    )

    agent.load(model_path)
    agent.online_net.eval()
    agent.target_net.eval()

    # IMPORTANT: disable exploration during evaluation
    agent.epsilon_override = 0.0 if hasattr(agent, "epsilon_override") else None

    episode_rewards = []

    # ----- Evaluation Loop -----
    for ep in range(n_episodes):

        obs, _ = env.reset()
        state = stacker.reset(preprocessor.process(obs))

        done = False
        total_reward = 0.0

        while not done:

            # Greedy action selection (no learning, minimal randomness)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
                q_values = agent.online_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = stacker.append(preprocessor.process(next_obs))

            state = next_state
            total_reward += reward

            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{n_episodes} - Reward: {total_reward:.2f}")

    env.close()

    return episode_rewards


# ----- CLI Entry starting command line -----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", action="store_true")

    args = parser.parse_args()

    rewards = evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        record=args.record
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Mean Reward: {np.mean(rewards):.2f}")
    print(f"Std Dev:     {np.std(rewards):.2f}")
    #python evaluate.py --model checkpointbetter_500000.pth --render