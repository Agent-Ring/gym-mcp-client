"""AgentRingClient - A unified interface for local and remote Gymnasium environments."""

from typing import Any, SupportsFloat, cast

import gymnasium as gym
import httpx
import numpy as np


class AgentRingClient:
    """
    A Gymnasium-compatible client that works with both local and remote environments.

    Supports two modes:
    1. Local mode: A thin wrapper around any Gymnasium environment
    2. Remote mode: Makes HTTP calls to a gym-mcp-server instance

    This allows users to develop code locally and seamlessly switch to remote environments
    without changing their code.

    Args:
        env_id: The Gymnasium environment ID (e.g., "CartPole-v1")
        mode: Either "local" or "remote"
        render_mode: The render mode for the environment (e.g., "rgb_array", "human")
        gym_server_url: The URL of the remote gym-mcp-server (required for remote mode)
        gym_server_key: Optional API key for authentication (for remote mode)
        **kwargs: Additional keyword arguments passed to gym.make() in local mode

    Examples:
        # Local mode
        env = AgentRingClient("CartPole-v1", mode="local")

        # Remote mode
        env = AgentRingClient(
            "CartPole-v1",
            mode="remote",
            gym_server_url="http://localhost:8000",
            gym_server_key="your-api-key"
        )

        # Use the same API for both
        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        env_id: str,
        mode: str = "local",
        render_mode: str | None = None,
        gym_server_url: str | None = None,
        gym_server_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.env_id = env_id
        self.mode = mode.lower()
        self.render_mode = render_mode
        self.gym_server_url = gym_server_url
        self.gym_server_key = gym_server_key
        self.kwargs = kwargs

        # Validate mode
        if self.mode not in ["local", "remote"]:
            raise ValueError(f"Mode must be 'local' or 'remote', got '{mode}'")

        # Validate remote mode requirements
        if self.mode == "remote":
            if not gym_server_url:
                raise ValueError("gym_server_url is required for remote mode")
            # Ensure URL doesn't end with a slash
            self.gym_server_url = gym_server_url.rstrip("/")

        # Initialize the appropriate backend
        if self.mode == "local":
            self._init_local()
        else:
            self._init_remote()

    def _init_local(self) -> None:
        """Initialize local Gymnasium environment."""
        make_kwargs = self.kwargs.copy()
        if self.render_mode:
            make_kwargs["render_mode"] = self.render_mode
        self.env = gym.make(self.env_id, **make_kwargs)

        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = getattr(self.env, "reward_range", (-float("inf"), float("inf")))
        self.spec = self.env.spec
        self.metadata = getattr(self.env, "metadata", {})

    def _init_remote(self) -> None:
        """Initialize remote connection and fetch environment info."""
        self.client = httpx.Client(timeout=30.0)
        self._headers = {}
        if self.gym_server_key:
            self._headers["Authorization"] = f"Bearer {self.gym_server_key}"

        # Fetch environment info from remote server
        try:
            result = self._call_remote_tool("get_env_info", {})
            # Handle both REST response format (with env_info) and direct env_info
            if "env_info" in result:
                env_info = result["env_info"]
            else:
                env_info = result

            if isinstance(env_info, dict) and not env_info.get("success", True):
                raise RuntimeError(f"Failed to get environment info: {env_info.get('error')}")

            # Parse and store environment properties
            self._setup_remote_spaces(env_info)
            reward_range = env_info.get("reward_range")
            if reward_range is None:
                self.reward_range = (-float("inf"), float("inf"))
            elif isinstance(reward_range, (list, tuple)) and len(reward_range) == 2:
                self.reward_range = tuple(reward_range)
            else:
                self.reward_range = (-float("inf"), float("inf"))
            self.metadata = env_info.get("metadata", {}) or {}
            self.spec = None  # Remote env spec not directly accessible

        except Exception as e:
            self.client.close()
            raise RuntimeError(f"Failed to initialize remote environment: {e}") from e

    def _setup_remote_spaces(self, env_info: dict[str, Any]) -> None:
        """Setup observation and action spaces from remote environment info."""
        # Handle string representation of spaces (from REST API)
        obs_space_str = env_info.get("observation_space", "")
        action_space_str = env_info.get("action_space", "")

        # For Text spaces, create appropriate Gymnasium spaces
        if isinstance(obs_space_str, str) and "Text(" in obs_space_str:
            # Extract parameters from Text(1, 1000000, ...)
            import re

            match = re.search(r"Text\((\d+),\s*(\d+),", obs_space_str)
            if match:
                min_length = int(match.group(1))
                max_length = int(match.group(2))
                # Use Text space if available
                try:
                    from gymnasium.spaces import Text

                    self.observation_space = Text(min_length=min_length, max_length=max_length)
                except (ImportError, AttributeError):
                    # Fallback for older gymnasium versions
                    self.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=(max_length,), dtype=np.uint8
                    )
            else:
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(1000,), dtype=np.uint8
                )
        else:
            # Try to parse as structured space info
            obs_space_info = env_info.get("observation_space", {})
            if isinstance(obs_space_info, dict):
                self.observation_space = self._parse_space(obs_space_info)
            else:
                # Fallback
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(1000,), dtype=np.uint8
                )

        if isinstance(action_space_str, str) and "Text(" in action_space_str:
            import re

            match = re.search(r"Text\((\d+),\s*(\d+),", action_space_str)
            if match:
                min_length = int(match.group(1))
                max_length = int(match.group(2))
                try:
                    from gymnasium.spaces import Text

                    self.action_space = Text(min_length=min_length, max_length=max_length)
                except (ImportError, AttributeError):
                    self.action_space = gym.spaces.Box(
                        low=0, high=255, shape=(max_length,), dtype=np.uint8
                    )
            else:
                self.action_space = gym.spaces.Box(low=0, high=255, shape=(100,), dtype=np.uint8)
        else:
            action_space_info = env_info.get("action_space", {})
            if isinstance(action_space_info, dict):
                self.action_space = self._parse_space(action_space_info)
            else:
                self.action_space = gym.spaces.Box(low=0, high=255, shape=(100,), dtype=np.uint8)

    def _parse_space(self, space_info: dict[str, Any]) -> gym.Space[Any]:
        """Parse space information from JSON to Gymnasium Space objects."""
        space_type = space_info.get("type", "")

        if space_type == "Box":
            low = np.array(space_info["low"])
            high = np.array(space_info["high"])
            shape = tuple(space_info["shape"])
            dtype_str = space_info.get("dtype", "float32")
            # Convert string dtype to numpy type for gymnasium compatibility
            dtype: type[np.floating[Any]] | type[np.integer[Any]]
            if dtype_str == "float32":
                dtype = np.float32
            elif dtype_str == "float64":
                dtype = np.float64
            elif dtype_str == "int32":
                dtype = np.int32
            elif dtype_str == "int64":
                dtype = np.int64
            else:
                dtype = np.float32  # fallback
            return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        elif space_type == "Discrete":
            n = space_info["n"]
            start = space_info.get("start", 0)
            return gym.spaces.Discrete(n=n, start=start)

        elif space_type == "MultiBinary":
            n = space_info["n"]
            return gym.spaces.MultiBinary(n=n)

        elif space_type == "MultiDiscrete":
            nvec = np.array(space_info["nvec"])
            return gym.spaces.MultiDiscrete(nvec=nvec)

        elif space_type == "Tuple":
            spaces = [self._parse_space(s) for s in space_info["spaces"]]
            return gym.spaces.Tuple(spaces)

        elif space_type == "Dict":
            spaces_dict = {k: self._parse_space(v) for k, v in space_info["spaces"].items()}
            return gym.spaces.Dict(spaces_dict)

        else:
            raise ValueError(f"Unknown space type: {space_type}")

    def _call_remote_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP call to the remote gym-mcp-server."""
        # Map tool names to REST endpoints
        endpoint_map = {
            "get_env_info": "/info",
            "reset_env": "/reset",
            "step_env": "/step",
            "render_env": "/render",
            "close_env": "/close",
        }

        # Use REST endpoint if available, otherwise fall back to MCP
        if tool_name in endpoint_map:
            url = f"{self.gym_server_url}{endpoint_map[tool_name]}"
            method = "GET" if tool_name == "get_env_info" else "POST"

            try:
                if method == "GET":
                    response = self.client.get(url, headers=self._headers)
                else:
                    response = self.client.post(url, json=params, headers=self._headers)
                response.raise_for_status()
                result = cast(dict[str, Any], response.json())
                # For get_env_info, return the full result (contains env_info key)
                return result
            except httpx.HTTPError as e:
                raise RuntimeError(f"Remote call failed: {e}") from e
        else:
            # Fallback to MCP endpoint
            url = f"{self.gym_server_url}/mcp/v1/tools/{tool_name}/call"
            payload = {"params": params}

            try:
                response = self.client.post(url, json=payload, headers=self._headers)
                response.raise_for_status()
                result = cast(dict[str, Any], response.json())
                return result
            except httpx.HTTPError as e:
                raise RuntimeError(f"Remote call failed: {e}") from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            observation: The initial observation
            info: Additional information dictionary
        """
        if self.mode == "local":
            return self.env.reset(seed=seed, options=options)
        else:
            params = {}
            if seed is not None:
                params["seed"] = seed

            result = self._call_remote_tool("reset_env", params)

            if not result.get("success"):
                raise RuntimeError(f"Reset failed: {result.get('error')}")

            observation = self._deserialize_observation(result.get("observation"))
            info = result.get("info", {})

            return observation, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The action to take

        Returns:
            observation: The observation after taking the action
            reward: The reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information dictionary
        """
        if self.mode == "local":
            return self.env.step(action)
        else:
            # Serialize action for remote call
            action_serialized = self._serialize_action(action)

            result = self._call_remote_tool("step_env", {"action": action_serialized})

            if not result.get("success"):
                raise RuntimeError(f"Step failed: {result.get('error')}")

            observation = self._deserialize_observation(result.get("observation"))
            reward = float(result.get("reward", 0.0))
            terminated = bool(result.get("terminated", False))
            truncated = bool(result.get("truncated", False))
            info = result.get("info", {})

            return observation, reward, terminated, truncated, info

    def render(self) -> Any | None:
        """
        Render the environment.

        Returns:
            The rendered output (depends on render_mode)
        """
        if self.mode == "local":
            return self.env.render()
        else:
            params = {}
            if self.render_mode:
                params["mode"] = self.render_mode

            result = self._call_remote_tool("render_env", params)

            if not result.get("success"):
                raise RuntimeError(f"Render failed: {result.get('error')}")

            # Handle different render modes
            render_data = result.get("render")

            # If it's image data (rgb_array), decode it
            if isinstance(render_data, list):
                return np.array(render_data)

            return render_data

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.mode == "local":
            self.env.close()
        else:
            try:
                self._call_remote_tool("close_env", {})
            finally:
                self.client.close()

    def _serialize_action(self, action: Any) -> Any:
        """Serialize action for remote transmission."""
        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, (list, tuple)):
            return [self._serialize_action(a) for a in action]
        elif isinstance(action, dict):
            return {k: self._serialize_action(v) for k, v in action.items()}
        else:
            return action

    def _deserialize_observation(self, observation: Any) -> Any:
        """Deserialize observation from remote response."""
        if observation is None:
            return None

        # If observation space is Box, convert to numpy array
        if isinstance(self.observation_space, gym.spaces.Box):
            return np.array(observation, dtype=self.observation_space.dtype)

        # For discrete spaces, return as-is (integer)
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            return int(observation)

        # For MultiBinary, convert to numpy array
        elif isinstance(self.observation_space, gym.spaces.MultiBinary):
            return np.array(observation, dtype=np.int8)

        # For MultiDiscrete, convert to numpy array
        elif isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            return np.array(observation, dtype=np.int64)

        # For Tuple spaces, recursively deserialize
        elif isinstance(self.observation_space, gym.spaces.Tuple):
            return tuple(observation)

        # For Dict spaces, recursively deserialize
        elif isinstance(self.observation_space, gym.spaces.Dict):
            return {k: np.array(v) if isinstance(v, list) else v for k, v in observation.items()}

        return observation

    def __enter__(self) -> "AgentRingClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @property
    def unwrapped(self) -> Any:
        """
        Return the underlying unwrapped environment, recursively unwrapping nested wrappers.

        This property provides compatibility with Gymnasium's standard API
        where environments expose an `unwrapped` attribute to access the
        base environment without any wrappers.

        Returns:
            The underlying unwrapped environment (either from local mode or remote mode)
        """
        if self.mode == "local":
            env = self.env
            # Recursively unwrap if the env itself has unwrapped
            # Stop if unwrapped points back to itself (base environment)
            while hasattr(env, "unwrapped"):
                unwrapped = env.unwrapped
                if unwrapped is env:
                    break
                env = unwrapped
            return env
        elif self.mode == "remote":
            # For remote mode, there is no underlying environment to unwrap
            return None
        else:
            return self.env

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the underlying environment.

        This allows transparent access to custom attributes and methods
        of the wrapped environment, making AgentRingClient fully compatible
        with code that expects direct access to environment internals.

        Args:
            name: The attribute name to access

        Returns:
            The requested attribute from the underlying environment

        Raises:
            AttributeError: If the attribute doesn't exist on either the
                           client or the underlying environment
        """
        # Don't forward dunder methods or attributes that exist on self
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Don't forward unwrapped to avoid infinite recursion
        if name == "unwrapped":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Forward to underlying environment (only in local mode)
        if self.mode == "local" and hasattr(self, "env") and self.env is not None:
            return getattr(self.env, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"AgentRingClient(env_id='{self.env_id}', mode='{self.mode}')"
