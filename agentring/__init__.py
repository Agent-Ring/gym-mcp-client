"""AgentRing - A unified interface for local and remote Gymnasium environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as _gym

if TYPE_CHECKING:
    from agentring.env import AgentRingEnv

__version__ = "0.1.0"

# Re-export key gymnasium components for full API compatibility
spaces = _gym.spaces
wrappers = _gym.wrappers
register = _gym.register
registry = _gym.registry
spec = _gym.spec
envs = _gym.envs
error = _gym.error
logger = _gym.logger
utils = _gym.utils


def make(
    id: str,
    mode: str = "local",
    render_mode: str | None = None,
    gym_server_url: str | None = None,
    gym_server_key: str | None = None,
    **kwargs: Any,
) -> AgentRingEnv:
    """
    Create a Gymnasium environment.

    This is the main entry point for creating environments in AgentRing.
    It supports both local and remote modes, providing a unified interface
    that is compatible with Gymnasium.

    Args:
        id: The environment ID (e.g., "CartPole-v1")
        mode: Either "local" or "remote"
        render_mode: The render mode for the environment
        gym_server_url: URL for remote gym-mcp-server (required for remote mode, optional if GYM_SERVER_URL env var is set)
        gym_server_key: Optional API key for remote server authentication
        **kwargs: Additional arguments passed to the environment

    Returns:
        An AgentRingEnv instance that provides the standard Gymnasium API

    Examples:
        # Local environment
        env = gym.make("CartPole-v1")

        # Remote environment with explicit URL
        env = gym.make("CartPole-v1", mode="remote", gym_server_url="http://localhost:8000")

        # Remote environment using environment variable
        # export GYM_SERVER_URL="http://localhost:8000"
        env = gym.make("CartPole-v1", mode="remote")  # Uses GYM_SERVER_URL env var
    """
    from agentring.env import AgentRingEnv

    return AgentRingEnv(
        env_id=id,
        mode=mode,
        render_mode=render_mode,
        gym_server_url=gym_server_url,
        gym_server_key=gym_server_key,
        **kwargs,
    )


__all__ = [
    "make",
    "spaces",
    "wrappers",
    "register",
    "registry",
    "spec",
    "envs",
    "error",
    "logger",
    "utils",
]
