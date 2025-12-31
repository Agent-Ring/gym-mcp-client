"""AgentRingEnv - A unified interface for local and remote Gymnasium environments."""

import uuid
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import httpx
import numpy as np

from .run_manager import RunManager


class AgentRingEnv(gym.Env[Any, Any]):
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
        env = AgentRingEnv("CartPole-v1", mode="local")

        # Remote mode
        env = AgentRingEnv(
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
        super().__init__()
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
            # Check environment variable if no URL provided
            if not gym_server_url:
                import os

                env_url = os.getenv("GYM_SERVER_URL")
                if not env_url:
                    raise ValueError(
                        "gym_server_url is required for remote mode. "
                        "Either pass gym_server_url parameter or set GYM_SERVER_URL environment variable."
                    )
                gym_server_url = env_url

            # Ensure URL doesn't end with a slash
            self.gym_server_url = gym_server_url.rstrip("/")

        # Initialize run manager
        self.run_manager = RunManager(mode=self.mode)

        # Initialize the appropriate backend
        if self.mode == "local":
            self._init_local(render_mode, **kwargs)
        else:
            try:
                self._init_remote(render_mode)
            except RuntimeError as e:
                # Check if this is a connection error
                if "Failed to connect to gym-mcp-server" in str(e):
                    import sys

                    print(str(e), file=sys.stderr)
                    sys.exit(1)
                raise
            # Set up sync callback for remote mode
            self.run_manager.set_sync_callback(self._get_server_statistics)

    def _init_local(self, render_mode: str | None, **kwargs: Any) -> None:
        """Initialize local Gymnasium environment."""
        make_kwargs = kwargs.copy()
        if render_mode:
            make_kwargs["render_mode"] = render_mode
        self.env = gym.make(self.env_id, **make_kwargs)

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Observation space of the environment."""
        if self.mode == "local":
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: gym.Space[Any]) -> None:
        """Set observation space (remote mode only)."""
        if self.mode == "local":
            raise AttributeError("Cannot set observation_space in local mode")
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space[Any]:
        """Action space of the environment."""
        if self.mode == "local":
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, value: gym.Space[Any]) -> None:
        """Set action space (remote mode only)."""
        if self.mode == "local":
            raise AttributeError("Cannot set action_space in local mode")
        self._action_space = value

    @property
    def reward_range(self) -> tuple[float, float]:
        """Reward range of the environment."""
        if self.mode == "local":
            reward_range = getattr(self.env, "reward_range", (-float("inf"), float("inf")))
            return cast(tuple[float, float], reward_range)
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: tuple[float, float]) -> None:
        """Set reward range (remote mode only)."""
        if self.mode == "local":
            raise AttributeError("Cannot set reward_range in local mode")
        self._reward_range = value

    @property
    def spec(self) -> Any:
        """Environment specification."""
        if self.mode == "local":
            return self.env.spec
        return self._spec

    @spec.setter
    def spec(self, value: Any) -> None:
        """Set spec (remote mode only)."""
        if self.mode == "local":
            raise AttributeError("Cannot set spec in local mode")
        self._spec = value

    @property
    def metadata(self) -> dict[str, Any]:
        """Environment metadata."""
        if self.mode == "local":
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Set metadata (remote mode only)."""
        if self.mode == "local":
            raise AttributeError("Cannot set metadata in local mode")
        self._metadata = value

    def _init_remote(self, render_mode: str | None) -> None:
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
                self._reward_range = (-float("inf"), float("inf"))
            elif isinstance(reward_range, (list, tuple)) and len(reward_range) == 2:
                self._reward_range = tuple(reward_range)
            else:
                self._reward_range = (-float("inf"), float("inf"))
            self._metadata = env_info.get("metadata", {}) or {}
            # Try to get spec from gymnasium registry if env_id is registered
            try:
                self._spec = gym.spec(self.env_id)
            except Exception:
                self._spec = None

        except Exception as e:
            self.client.close()
            # Check if this is a connection error (either directly or wrapped)
            is_connection_error = isinstance(e, httpx.ConnectError) or (
                isinstance(e, RuntimeError)
                and e.__cause__ is not None
                and isinstance(e.__cause__, httpx.ConnectError)
            )

            if is_connection_error:
                connect_error = e if isinstance(e, httpx.ConnectError) else e.__cause__
                error_msg = (
                    f"Failed to connect to gym-mcp-server at {self.gym_server_url}.\n"
                    f"Please make sure the server is running.\n\n"
                    f"To start the server, run:\n"
                    f"  python -m gym_mcp_server --env {self.env_id}\n\n"
                    f"Connection error: {connect_error}"
                )
                raise RuntimeError(error_msg) from None
            else:
                raise RuntimeError(f"Failed to initialize remote environment: {e}") from e

    def _setup_remote_spaces(self, env_info: dict[str, Any]) -> None:
        """Setup observation and action spaces from remote environment info."""
        # Handle string representation of spaces (from REST API)
        obs_space_str = env_info.get("observation_space", "")
        action_space_str = env_info.get("action_space", "")

        # Parse observation space
        if isinstance(obs_space_str, str):
            import re

            # Handle Discrete spaces
            if "Discrete(" in obs_space_str:
                match = re.search(r"Discrete\((\d+)", obs_space_str)
                if match:
                    n = int(match.group(1))
                    self._observation_space = gym.spaces.Discrete(n=n)
                else:
                    # Fallback
                    self._observation_space = gym.spaces.Box(
                        low=0, high=255, shape=(1000,), dtype=np.uint8
                    )
            # Handle Text spaces
            elif "Text(" in obs_space_str:
                # Extract parameters from Text(1, 1000000, ...)
                match = re.search(r"Text\((\d+),\s*(\d+),", obs_space_str)
                if match:
                    min_length = int(match.group(1))
                    max_length = int(match.group(2))
                    # Use Text space if available
                    try:
                        from gymnasium.spaces import Text

                        self._observation_space = Text(min_length=min_length, max_length=max_length)
                    except (ImportError, AttributeError):
                        # Fallback for older gymnasium versions
                        self._observation_space = gym.spaces.Box(
                            low=0, high=255, shape=(max_length,), dtype=np.uint8
                        )
                else:
                    self._observation_space = gym.spaces.Box(
                        low=0, high=255, shape=(1000,), dtype=np.uint8
                    )
            else:
                # Fallback for unrecognized string formats
                self._observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(1000,), dtype=np.uint8
                )
        else:
            # Try to parse as structured space info
            obs_space_info = env_info.get("observation_space", {})
            if isinstance(obs_space_info, dict):
                self._observation_space = self._parse_space(obs_space_info)
            else:
                # Fallback
                self._observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(1000,), dtype=np.uint8
                )

        if isinstance(action_space_str, str):
            import re

            # Handle Text spaces
            if "Text(" in action_space_str:
                match = re.search(r"Text\((\d+),\s*(\d+),", action_space_str)
                if match:
                    min_length = int(match.group(1))
                    max_length = int(match.group(2))
                    try:
                        from gymnasium.spaces import Text

                        self._action_space = Text(min_length=min_length, max_length=max_length)
                    except (ImportError, AttributeError):
                        self._action_space = gym.spaces.Box(
                            low=0, high=255, shape=(max_length,), dtype=np.uint8
                        )
                    return
            # Handle Discrete spaces
            elif "Discrete(" in action_space_str:
                match = re.search(r"Discrete\((\d+)", action_space_str)
                if match:
                    n = int(match.group(1))
                    self._action_space = gym.spaces.Discrete(n=n)
                    return

            # Fallback for unrecognized string formats
            self._action_space = gym.spaces.Box(low=0, high=255, shape=(100,), dtype=np.uint8)
        else:
            action_space_info = env_info.get("action_space", {})
            if isinstance(action_space_info, dict):
                self._action_space = self._parse_space(action_space_info)
            else:
                self._action_space = gym.spaces.Box(low=0, high=255, shape=(100,), dtype=np.uint8)

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
        # Call super().reset() to initialize _np_random for env_checker compatibility
        super().reset(seed=seed)

        if self.mode == "local":
            observation, info = self.env.reset(seed=seed, options=options)
            # Track episode start in run manager (local mode)
            if self.run_manager.stats is None:
                run_id = str(uuid.uuid4())
                self.run_manager.start_run(run_id, self.env_id)
            # Get next episode number
            if self.run_manager.stats is not None:
                next_episode = self.run_manager.stats.current_episode + 1
                self.run_manager.start_episode(next_episode)
        else:
            params: dict[str, Any] = {}
            if seed is not None:
                params["seed"] = seed
            if options is not None:
                params["options"] = options

            result = self._call_remote_tool("reset_env", params)

            if not result.get("success"):
                raise RuntimeError(f"Reset failed: {result.get('error')}")

            observation = self._deserialize_observation(result.get("observation"))
            info = result.get("info", {})

            # Track episode start in run manager
            # Get run status from server to get the actual run_id
            try:
                run_status = self.get_run_status()
                if run_status and run_status.get("run_id"):
                    run_id = run_status["run_id"]
                    # Initialize run if not started or if run_id changed
                    if self.run_manager.stats is None or self.run_manager.stats.run_id != run_id:
                        self.run_manager.start_run(run_id, self.env_id)

                    # Get episode number from run_progress or run_status
                    run_progress = result.get("run_progress", {})
                    episode_num = run_progress.get("episode_num") or run_status.get(
                        "current_episode", 0
                    )
                    if episode_num > 0:
                        self.run_manager.start_episode(episode_num)
            except Exception:
                # If getting run status fails, try to use run_progress
                run_progress = result.get("run_progress", {})
                if run_progress:
                    episode_num = run_progress.get("episode_num", 0)
                    if episode_num > 0:
                        if self.run_manager.stats is None:
                            run_id = str(uuid.uuid4())
                            self.run_manager.start_run(run_id, self.env_id)
                        self.run_manager.start_episode(episode_num)

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
            observation, reward, terminated, truncated, info = self.env.step(action)
            # Track step in run manager
            self.run_manager.record_step(float(reward), terminated, truncated)
            # Finalize episode if done
            if terminated or truncated:
                self.run_manager._finalize_episode()
            return observation, reward, terminated, truncated, info
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

            # Track step in run manager
            self.run_manager.record_step(reward, terminated, truncated)

            # Finalize episode if done (this will sync with server in remote mode)
            if terminated or truncated:
                self.run_manager._finalize_episode()

            return observation, reward, terminated, truncated, info

    def render(self) -> Any | None:
        """Render the environment."""
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
                render_array = np.array(render_data)
                # Convert to uint8 for RGB arrays (gymnasium expects uint8 for rgb_array mode)
                if self.render_mode == "rgb_array" and render_array.dtype != np.uint8:
                    # Ensure values are in [0, 255] range and convert to uint8
                    render_array = np.clip(render_array, 0, 255).astype(np.uint8)
                return render_array

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

    def start_run(self) -> dict[str, Any]:
        """
        Start a new run on the server.

        This initializes the run manager on the server and begins tracking episodes.
        All subsequent episodes (via reset()) will be logged under this run until
        reset_run() is called.

        Note: This method only works in remote mode. In local mode, it's a no-op.

        Returns:
            Dictionary with run information (run_id, state, num_episodes, etc.)

        Raises:
            RuntimeError: If called in local mode or if the server call fails
        """
        if self.mode == "local":
            # In local mode, run management is not available
            return {
                "success": False,
                "error": "start_run() is only available in remote mode",
            }

        try:
            url = f"{self.gym_server_url}/run/start"
            response = self.client.post(url, headers=self._headers)
            response.raise_for_status()
            result = cast(dict[str, Any], response.json())
            return result
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to start run: {e}") from e

    def reset_run(self) -> dict[str, Any]:
        """
        Reset the run on the server and start a new run.

        This clears all statistics and starts a fresh run. All subsequent episodes
        (via reset()) will be logged under this new run.

        Note: This method only works in remote mode. In local mode, it's a no-op.

        Returns:
            Dictionary with run information (run_id, state, num_episodes, etc.)

        Raises:
            RuntimeError: If called in local mode or if the server call fails
        """
        if self.mode == "local":
            # In local mode, run management is not available
            return {
                "success": False,
                "error": "reset_run() is only available in remote mode",
            }

        try:
            url = f"{self.gym_server_url}/run/reset"
            response = self.client.post(url, headers=self._headers)
            response.raise_for_status()
            result = cast(dict[str, Any], response.json())
            return result
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to reset run: {e}") from e

    def get_run_status(self) -> dict[str, Any]:
        """
        Get current run status from the server.

        Note: This method only works in remote mode. In local mode, it's a no-op.

        Returns:
            Dictionary with current run status (run_id, state, progress, etc.)

        Raises:
            RuntimeError: If called in local mode or if the server call fails
        """
        if self.mode == "local":
            return {
                "success": False,
                "error": "get_run_status() is only available in remote mode",
            }

        try:
            url = f"{self.gym_server_url}/run/status"
            response = self.client.get(url, headers=self._headers)
            response.raise_for_status()
            result = cast(dict[str, Any], response.json())
            return result
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to get run status: {e}") from e

    def get_run_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive run statistics.

        In remote mode, returns statistics from the server.
        In local mode, returns local statistics.

        Returns:
            Dictionary with comprehensive run statistics (episodes, rewards, etc.)

        Raises:
            RuntimeError: If called in remote mode and the server call fails
        """
        if self.mode == "local":
            return self.run_manager.get_statistics()
        else:
            try:
                url = f"{self.gym_server_url}/run/statistics"
                response = self.client.get(url, headers=self._headers)
                response.raise_for_status()
                result = cast(dict[str, Any], response.json())
                return result
            except httpx.HTTPError as e:
                raise RuntimeError(f"Failed to get run statistics: {e}") from e

    def _get_server_statistics(self) -> dict[str, Any]:
        """Internal method to get server statistics for sync callback."""
        try:
            url = f"{self.gym_server_url}/run/statistics"
            response = self.client.get(url, headers=self._headers)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except Exception:
            return {}

    def _serialize_action(self, action: Any) -> Any:
        """Serialize action for remote transmission."""
        # Handle numpy scalars (int64, float64, etc.) - convert to Python native types
        if isinstance(action, (np.integer, np.floating)):
            return action.item()
        elif isinstance(action, np.ndarray):
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

    def __enter__(self) -> "AgentRingEnv":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @property
    def unwrapped(self) -> Any:
        """Return the underlying unwrapped environment."""
        if self.mode == "local":
            return self.env.unwrapped
        else:
            # For remote mode, return self so env_checker can access _np_random
            return self

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying environment (local mode only)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self.mode == "local" and hasattr(self, "env") and self.env is not None:
            return getattr(self.env, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"AgentRingEnv(env_id='{self.env_id}', mode='{self.mode}')"
