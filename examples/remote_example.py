"""Example of using GymMCPClient in remote mode.

Before running this example, start the gym-mcp-server:
    python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
"""

from agentring import GymMCPClient


def main():
    """Run a simple CartPole example using remote mode."""
    # Create a remote environment
    # Make sure the gym-mcp-server is running on http://localhost:8000
    env = GymMCPClient(
        "CartPole-v1",
        mode="remote",
        gym_server_url="http://localhost:8000",
        render_mode="rgb_array",
    )

    print(f"Environment: {env}")
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

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
