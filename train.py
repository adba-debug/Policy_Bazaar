"""
Main training loop.
Handles environment interaction, agent learning,
logging, and checkpoint saving.
"""

import numpy as np
from config import Config
from env import make_env
from agent import DoubleDQNAgent


def train():
    """
    Run the full training loop:
      1. Create environment and agent.
      2. For each step:
         a. Select action (epsilon-greedy).
         b. Step environment.
         c. Store transition.
         d. Call agent.learn() if enough samples.
         e. Sync target network periodically.
         f. Log episode rewards.
         g. Save checkpoints.
    """
    config = Config()

    # ----- setup -----
    env, preprocessor, stacker = make_env(
        config.ENV_NAME, config.FRAME_SIZE, config.FRAME_STACK
    )
    n_actions = env.action_space.n
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    agent = DoubleDQNAgent(
        n_frames=config.FRAME_STACK,
        n_actions=n_actions,
        config=config,
        device=device,
    )

    # ----- metrics -----
    episode_rewards = []
    episode_reward = 0.0
    episode_count = 0
    losses = []

    # ----- initial reset -----
    obs, info = env.reset()
    state = stacker.reset(preprocessor.process(obs))

    # ----- training loop -----
    for step in range(1, config.TOTAL_STEPS + 1):
        # Select action
        action = agent.select_action(state)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Process next observation and update frame stack
        next_state = stacker.append(
            preprocessor.process(next_obs)
        )

        # Store transition
        agent.store_transition(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn if enough samples are available
        if len(agent.buffer) >= config.TRAIN_START  and step%4==0 :
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

        # Update target network
        if step % config.TARGET_UPDATE == 0:
            agent.sync_target()

        # Save checkpoints
        if step % config.SAVE_EVERY == 0:
            agent.save(f"checkpoint_{step}.pth")

        # Update reward
        episode_reward += reward

        # Move to next state
        state = next_state

        # Episode finished
        if done:

            episode_rewards.append(episode_reward)

            episode_count += 1

            print(
                f"Episode {episode_count} | "
                f"Reward: {episode_reward:.2f}"
            )

            episode_reward = 0.0

            obs, info = env.reset()

            state = stacker.reset(
                preprocessor.process(obs)
            )
    env.close()
    return episode_rewards


if __name__ == "__main__":
    rewards = train()