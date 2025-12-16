"""Example showing the use of GymMCPClient with context managers."""

from agentring import GymMCPClient


def run_with_context_manager(mode: str, **kwargs):
    """Run an example using context manager (automatically handles cleanup)."""
    print(f"\n{'=' * 50}")
    print(f"Running in {mode.upper()} mode")
    print(f"{'=' * 50}\n")

    with GymMCPClient("CartPole-v1", mode=mode, **kwargs) as env:
        print(f"Environment: {env}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}\n")

        # Reset and run one episode
        observation, info = env.reset(seed=42)
        total_reward = 0
        steps = 0

        for _ in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print("Episode completed:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward}")

    # env.close() is called automatically when exiting the context


def main():
    """Demonstrate context manager usage with both local and remote modes."""

    # Local mode example
    run_with_context_manager(mode="local", render_mode="rgb_array")

    # Remote mode example (uncomment if server is running)
    # run_with_context_manager(
    #     mode="remote",
    #     gym_server_url="http://localhost:8000",
    #     render_mode="rgb_array"
    # )


if __name__ == "__main__":
    main()
