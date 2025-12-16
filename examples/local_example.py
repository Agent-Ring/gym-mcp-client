"""Example of using GymMCPClient in local mode."""

from agentring import GymMCPClient


def main():
    """Run a simple CartPole example using local mode."""
    # Create a local environment
    env = GymMCPClient("CartPole-v1", mode="local", render_mode="rgb_array")

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
