"""Tests for GymMCPClient."""

import numpy as np
import pytest

from gym_mcp_client import GymMCPClient


class TestLocalMode:
    """Tests for local mode functionality."""

    def test_initialization(self):
        """Test basic initialization in local mode."""
        env = GymMCPClient("CartPole-v1", mode="local")

        assert env.mode == "local"
        assert env.env_id == "CartPole-v1"
        assert env.observation_space is not None
        assert env.action_space is not None

        env.close()

    def test_reset(self):
        """Test reset functionality."""
        env = GymMCPClient("CartPole-v1", mode="local")

        observation, info = env.reset(seed=42)

        assert observation is not None
        assert isinstance(info, dict)
        assert observation.shape == (4,)  # CartPole observation space

        env.close()

    def test_step(self):
        """Test step functionality."""
        env = GymMCPClient("CartPole-v1", mode="local")

        observation, info = env.reset(seed=42)
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        assert observation is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_render(self):
        """Test render functionality."""
        env = GymMCPClient("CartPole-v1", mode="local", render_mode="rgb_array")

        env.reset()
        render_output = env.render()

        assert render_output is not None
        assert isinstance(render_output, np.ndarray)
        assert len(render_output.shape) == 3  # RGB image

        env.close()

    def test_context_manager(self):
        """Test context manager functionality."""
        with GymMCPClient("CartPole-v1", mode="local") as env:
            observation, info = env.reset()
            assert observation is not None

        # Environment should be closed after exiting context

    def test_full_episode(self):
        """Test running a full episode."""
        env = GymMCPClient("CartPole-v1", mode="local")

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

        assert steps > 0
        assert total_reward > 0

        env.close()

    def test_repr(self):
        """Test string representation."""
        env = GymMCPClient("CartPole-v1", mode="local")
        repr_str = repr(env)

        assert "GymMCPClient" in repr_str
        assert "CartPole-v1" in repr_str
        assert "local" in repr_str

        env.close()

    def test_unwrapped_property(self):
        """Test unwrapped property returns underlying environment."""
        env = GymMCPClient("CartPole-v1", mode="local")

        # unwrapped should return the underlying gym environment
        unwrapped_env = env.unwrapped
        assert unwrapped_env is not None
        assert unwrapped_env != env

        # Should have the same observation and action spaces
        assert unwrapped_env.observation_space == env.observation_space
        assert unwrapped_env.action_space == env.action_space

        env.close()

    def test_unwrapped_recursive(self):
        """Test that unwrapped recursively unwraps nested wrappers."""
        # CartPole-v1 is wrapped by TimeLimit, so unwrapping should get the base CartPoleEnv
        env = GymMCPClient("CartPole-v1", mode="local")

        # unwrapped should recursively unwrap to the base CartPole env
        unwrapped = env.unwrapped
        assert unwrapped is not None
        # Should be different from the wrapped env (TimeLimit)
        assert unwrapped != env.env
        # Should be the actual CartPoleEnv without any wrappers
        import gymnasium.envs.classic_control.cartpole as cartpole

        assert isinstance(unwrapped, cartpole.CartPoleEnv)

        env.close()

    def test_getattr_forwarding(self):
        """Test that __getattr__ forwards to underlying environment."""
        env = GymMCPClient("CartPole-v1", mode="local")

        # Standard Gymnasium attributes should work
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert hasattr(env, "metadata")

        # Should be able to access the underlying env's spec
        if hasattr(env.unwrapped, "spec"):
            assert env.unwrapped.spec is not None

        env.close()

    def test_getattr_raises_for_dunder(self):
        """Test that __getattr__ doesn't forward dunder methods."""
        env = GymMCPClient("CartPole-v1", mode="local")

        # Should raise AttributeError for dunder methods
        with pytest.raises(AttributeError):
            _ = env.__nonexistent_dunder__

        env.close()

    def test_getattr_raises_for_nonexistent(self):
        """Test that __getattr__ raises for non-existent attributes."""
        env = GymMCPClient("CartPole-v1", mode="local")

        # Should raise AttributeError for non-existent attributes
        with pytest.raises(AttributeError):
            _ = env.nonexistent_attribute_12345

        env.close()


class TestRemoteMode:
    """Tests for remote mode functionality."""

    def test_initialization_requires_url(self):
        """Test that remote mode requires gym_server_url."""
        with pytest.raises(ValueError, match="gym_server_url is required"):
            GymMCPClient("CartPole-v1", mode="remote")

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Mode must be"):
            GymMCPClient("CartPole-v1", mode="invalid")

    # Note: Additional remote tests would require a running gym-mcp-server
    # These are integration tests and should be run separately


class TestSpaceParsing:
    """Tests for space parsing functionality."""

    def test_box_space(self):
        """Test Box space creation in local mode."""
        env = GymMCPClient("CartPole-v1", mode="local")

        from gymnasium.spaces import Box

        assert isinstance(env.observation_space, Box)

        env.close()

    def test_discrete_space(self):
        """Test Discrete space creation in local mode."""
        env = GymMCPClient("CartPole-v1", mode="local")

        from gymnasium.spaces import Discrete

        assert isinstance(env.action_space, Discrete)

        env.close()


class TestActionSerialization:
    """Tests for action serialization."""

    def test_serialize_int_action(self):
        """Test serialization of integer actions."""
        env = GymMCPClient("CartPole-v1", mode="local")

        action = 1
        serialized = env._serialize_action(action)
        assert serialized == 1

        env.close()

    def test_serialize_numpy_action(self):
        """Test serialization of numpy array actions."""
        env = GymMCPClient("CartPole-v1", mode="local")

        action = np.array([0.5, 0.3])
        serialized = env._serialize_action(action)
        assert isinstance(serialized, list)
        assert serialized == [0.5, 0.3]

        env.close()

    def test_serialize_list_action(self):
        """Test serialization of list actions."""
        env = GymMCPClient("CartPole-v1", mode="local")

        action = [1, 2, 3]
        serialized = env._serialize_action(action)
        assert serialized == [1, 2, 3]

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
