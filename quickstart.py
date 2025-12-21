"""Example of using AgentRing in both local and remote modes.

For remote mode, start the gym-mcp-server first:
    python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000

Then set REMOTE_MODE=True below to use remote server instead of local.
"""

import agentring as gym

# Set to True to use remote server (requires gym-mcp-server running on localhost:8000)
REMOTE_MODE = False


def main():
    """Run a simple CartPole example using either local or remote mode."""
    mode = "remote" if REMOTE_MODE else "local"
    kwargs = {}

    if REMOTE_MODE:
        # Remote mode requires the gym-mcp-server to be running
        kwargs.update({
            "gym_server_url": "http://localhost:8000",
        })

    # Using context manager for automatic cleanup (recommended)
    with gym.make("CartPole-v1", mode=mode, render_mode="rgb_array", **kwargs) as env:
        print(f"Environment: {env}")
        print(f"Mode: {mode}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print()

        # Run a few episodes
        num_episodes = 3

        for episode in range(num_episodes):
            observation, info = env.reset(seed=42 + episode)
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False

            print(f"Episode {episode + 1}:")
            print(f"  Initial observation: {observation}")

            while not (terminated or truncated):
                # Take a random action
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                if steps >= 200:  # Limit steps
                    break

            print(f"  Total reward: {total_reward}")
            print(f"  Steps: {steps}")
            print()

    # env.close() is called automatically when exiting the context
    print("Done!")


if __name__ == "__main__":
    main()
