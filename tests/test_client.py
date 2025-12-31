"""Tests for AgentRingEnv."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# When running from the monorepo root, the agentring package isn't installed.
# Add the subproject root to sys.path so `import agentring` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentring.env import AgentRingEnv


class TestLocalMode:
    """Tests for local mode functionality."""

    def test_initialization(self):
        """Test basic initialization in local mode."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        assert env.mode == "local"
        assert env.env_id == "CartPole-v1"
        assert env.observation_space is not None
        assert env.action_space is not None

        env.close()

    def test_reset(self):
        """Test reset functionality."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        observation, info = env.reset(seed=42)

        assert observation is not None
        assert isinstance(info, dict)
        assert observation.shape == (4,)  # CartPole observation space

        env.close()

    def test_step(self):
        """Test step functionality."""
        env = AgentRingEnv("CartPole-v1", mode="local")

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
        env = AgentRingEnv("CartPole-v1", mode="local", render_mode="rgb_array")

        env.reset()
        render_output = env.render()

        assert render_output is not None
        assert isinstance(render_output, np.ndarray)
        assert len(render_output.shape) == 3  # RGB image

        env.close()

    def test_context_manager(self):
        """Test context manager functionality."""
        with AgentRingEnv("CartPole-v1", mode="local") as env:
            observation, info = env.reset()
            assert observation is not None

        # Environment should be closed after exiting context

    def test_full_episode(self):
        """Test running a full episode."""
        env = AgentRingEnv("CartPole-v1", mode="local")

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
        env = AgentRingEnv("CartPole-v1", mode="local")
        repr_str = repr(env)

        assert "AgentRingEnv" in repr_str
        assert "CartPole-v1" in repr_str
        assert "local" in repr_str

        env.close()

    def test_unwrapped_property(self):
        """Test unwrapped property returns underlying environment."""
        env = AgentRingEnv("CartPole-v1", mode="local")

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
        env = AgentRingEnv("CartPole-v1", mode="local")

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
        env = AgentRingEnv("CartPole-v1", mode="local")

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
        env = AgentRingEnv("CartPole-v1", mode="local")

        # Should raise AttributeError for dunder methods
        with pytest.raises(AttributeError):
            _ = env.__nonexistent_dunder__

        env.close()

    def test_getattr_raises_for_nonexistent(self):
        """Test that __getattr__ raises for non-existent attributes."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        # Should raise AttributeError for non-existent attributes
        with pytest.raises(AttributeError):
            _ = env.nonexistent_attribute_12345

        env.close()


class TestRemoteMode:
    """Tests for remote mode functionality."""

    def test_initialization_requires_url(self):
        """Test that remote mode requires gym_server_url."""
        with pytest.raises(ValueError, match="gym_server_url is required"):
            AgentRingEnv("CartPole-v1", mode="remote")

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Mode must be"):
            AgentRingEnv("CartPole-v1", mode="invalid")

    def test_env_checker_remote(self):
        """Test that remote environment passes gymnasium's env_checker validation."""
        from gymnasium.utils.env_checker import check_env

        gym_server_url = "http://localhost:8001"

        # Mock HTTP responses for remote environment
        def create_mock_response(response_data):
            """Create a mock httpx response."""
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200
            return mock_response

        def mock_get(url, **kwargs):
            """Mock GET requests."""
            path = url.split(gym_server_url)[1] if gym_server_url in str(url) else str(url)
            if "/info" in path:
                return create_mock_response(
                    {
                        "env_info": {
                            "observation_space": {
                                "type": "Discrete",
                                "n": 16,
                            },
                            "action_space": {
                                "type": "Discrete",
                                "n": 4,
                            },
                            "reward_range": [-float("inf"), float("inf")],
                            "metadata": {
                                "render_modes": ["rgb_array", "ansi"],
                                "render_fps": 4,
                            },
                        },
                    }
                )
            elif "/run/status" in path:
                return create_mock_response(
                    {
                        "run_id": "test-run-123",
                        "state": "running",
                        "current_episode": 1,
                    }
                )
            elif "/run/statistics" in path:
                return create_mock_response(
                    {
                        "run_id": "test-run-123",
                        "state": "running",
                        "current_episode": 1,
                        "completed_episodes": 0,
                        "total_steps": 0,
                        "total_reward": 0.0,
                        "average_reward": 0.0,
                        "average_steps": 0.0,
                        "success_rate": 0.0,
                        "successful_episodes": 0,
                        "episodes": [],
                    }
                )
            mock_response = MagicMock()
            mock_response.json.return_value = {"success": False, "error": "Not found"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 404
            return mock_response

        def mock_post(url, **kwargs):
            """Mock POST requests."""
            path = url.split(gym_server_url)[1] if gym_server_url in str(url) else str(url)
            if "/reset" in path:
                return create_mock_response(
                    {
                        "success": True,
                        "observation": 0,
                        "info": {},
                        "run_progress": {"episode_num": 1},
                    }
                )
            elif "/step" in path:
                return create_mock_response(
                    {
                        "success": True,
                        "observation": 1,
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "info": {},
                    }
                )
            elif "/render" in path:
                # Create a proper RGB array with uint8 values
                # Shape: (400, 600, 3) for RGB image
                render_array = np.zeros((400, 600, 3), dtype=np.uint8)
                # Convert to list for JSON serialization (numpy arrays are converted to lists)
                return create_mock_response(
                    {
                        "success": True,
                        "render": render_array.tolist(),  # Mock RGB array as list
                    }
                )
            mock_response = MagicMock()
            mock_response.json.return_value = {"success": False, "error": "Not found"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 404
            return mock_response

        # Create a mock client
        mock_client = MagicMock()
        mock_client.get = MagicMock(side_effect=mock_get)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client.close = MagicMock()

        # Create remote environment with mocked HTTP client
        with patch("httpx.Client", return_value=mock_client):
            env = AgentRingEnv(
                "FrozenLake-v1",
                mode="remote",
                render_mode="rgb_array",
                gym_server_url=gym_server_url,
            )

            try:
                # Run the environment checker
                # This validates:
                # - Observation and action spaces
                # - Reset and step methods
                # - Render method (if render_mode is set)
                # - Other Gymnasium API compliance checks
                check_env(env, skip_render_check=False)
            finally:
                env.close()


class TestSpaceParsing:
    """Tests for space parsing functionality."""

    def test_box_space(self):
        """Test Box space creation in local mode."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        from gymnasium.spaces import Box

        assert isinstance(env.observation_space, Box)

        env.close()

    def test_discrete_space(self):
        """Test Discrete space creation in local mode."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        from gymnasium.spaces import Discrete

        assert isinstance(env.action_space, Discrete)

        env.close()


class TestActionSerialization:
    """Tests for action serialization."""

    def test_serialize_int_action(self):
        """Test serialization of integer actions."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        action = 1
        serialized = env._serialize_action(action)
        assert serialized == 1

        env.close()

    def test_serialize_numpy_action(self):
        """Test serialization of numpy array actions."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        action = np.array([0.5, 0.3])
        serialized = env._serialize_action(action)
        assert isinstance(serialized, list)
        assert serialized == [0.5, 0.3]

        env.close()

    def test_serialize_list_action(self):
        """Test serialization of list actions."""
        env = AgentRingEnv("CartPole-v1", mode="local")

        action = [1, 2, 3]
        serialized = env._serialize_action(action)
        assert serialized == [1, 2, 3]

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
