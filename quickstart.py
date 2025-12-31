"""
Example demonstrating agentring compatibility with gymnasium.

This example shows that agentring is fully compatible with the standard
gymnasium API. You can use agentring as a drop-in replacement for gymnasium.

The code follows the standard gymnasium pattern:
1. Create environment with gym.make()
2. Reset to get initial observation
3. Loop: sample action, step, check if done, reset if needed
4. Close environment

This works identically whether using local or remote mode.
"""

import agentring as gym


def main():
    """Demonstrate agentring compatibility with gymnasium."""
    print("=" * 60)
    print("AgentRing Gymnasium Compatibility Example")
    print("=" * 60)
    print()
    print("This example demonstrates that agentring is fully compatible")
    print("with the standard gymnasium API.")
    print()

    # Initialise the environment
    # You can use any gymnasium environment - agentring is fully compatible!
    # Examples: "CartPole-v1", "LunarLander-v3", "FrozenLake-v1", etc.
    # Note: LunarLander-v3 requires pygame for render_mode="human"
    # 
    # For remote mode, set mode="remote" and provide gym_server_url:
    # env = gym.make("FrozenLake-v1", mode="remote", gym_server_url="http://localhost:8001")
    env = gym.make(
        "FrozenLake-v1",
        mode="local",
        render_mode="rgb_array",
        gym_server_url="http://localhost:8000"
    )

    try:
        # Reset the environment to generate the first observation
        observation, info = env.reset(seed=42)
        print(f"Environment initialized: {env}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print()
        print("Running episodes (standard gymnasium pattern)...")
        print()

        episode_count = 0
        total_steps = 0

        for step in range(1000):
            # this is where you would insert your policy
            action = env.action_space.sample()

            # step (transition) through the environment with the action
            # receiving the next observation, reward and if the episode has terminated or truncated
            observation, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # If the episode has ended then we can reset to start a new episode
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} completed: {step + 1} steps")
                observation, info = env.reset()

        print()
        print("=" * 60)
        print(f"Completed {episode_count} episodes in {total_steps} total steps")
        print("=" * 60)
        print()
        print("✓ AgentRing is fully compatible with gymnasium!")
        print("  You can use it as a drop-in replacement.")
        print()
        
        # Print run statistics
        print(env.run_manager.format_statistics(show_episode_details=False))

    finally:
        # Close the environment
        env.close()


if __name__ == "__main__":
    main()
