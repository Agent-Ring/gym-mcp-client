"""GymMCPClient - A unified interface for local and remote Gymnasium environments."""

from typing import Any, SupportsFloat

import gymnasium as gym
import httpx
import numpy as np


class GymMCPClient:
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
        env = GymMCPClient("CartPole-v1", mode="local")

        # Remote mode
        env = GymMCPClient(
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
            env_info = self._call_remote_tool("get_env_info", {})
            if not env_info.get("success"):
                raise RuntimeError(f"Failed to get environment info: {env_info.get('error')}")

            # Parse and store environment properties
            self._setup_remote_spaces(env_info)
            self.reward_range = tuple(env_info.get("reward_range", (-float("inf"), float("inf"))))
            self.metadata = env_info.get("metadata", {})
            self.spec = None  # Remote env spec not directly accessible

        except Exception as e:
            self.client.close()
            raise RuntimeError(f"Failed to initialize remote environment: {e}") from e

    def _setup_remote_spaces(self, env_info: dict[str, Any]) -> None:
        """Setup observation and action spaces from remote environment info."""
        # Parse observation space
        obs_space_info = env_info.get("observation_space", {})
        self.observation_space = self._parse_space(obs_space_info)

        # Parse action space
        action_space_info = env_info.get("action_space", {})
        self.action_space = self._parse_space(action_space_info)

    def _parse_space(self, space_info: dict[str, Any]) -> gym.Space[Any]:
        """Parse space information from JSON to Gymnasium Space objects."""
        space_type = space_info.get("type", "")

        if space_type == "Box":
            low = np.array(space_info["low"])
            high = np.array(space_info["high"])
            shape = tuple(space_info["shape"])
            dtype = np.dtype(space_info.get("dtype", "float32"))
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
        url = f"{self.gym_server_url}/mcp/v1/tools/{tool_name}/call"

        payload = {"params": params}

        try:
            response = self.client.post(url, json=payload, headers=self._headers)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
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
            self.env.close()  # type: ignore[no-untyped-call]
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

    def __enter__(self) -> "GymMCPClient":
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
            # For remote mode, unwrapped might be the same as env or None
            # depending on the remote server's implementation
            return self.env if hasattr(self.env, "unwrapped") else None
        else:
            return self.env

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the underlying environment.

        This allows transparent access to custom attributes and methods
        of the wrapped environment, making GymMCPClient fully compatible
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
        return f"GymMCPClient(env_id='{self.env_id}', mode='{self.mode}')"
