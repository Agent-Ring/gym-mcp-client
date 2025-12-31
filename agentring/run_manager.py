"""RunManager - Manages run statistics for AgentRing environments."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    episode_num: int
    total_steps: int = 0
    total_reward: float = 0.0
    done: bool = False
    truncated: bool = False
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class RunStats:
    """Statistics for a complete run."""

    run_id: str
    env_id: str
    state: str = "idle"  # idle, running, completed, error
    current_episode: int = 0
    completed_episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    average_steps: float = 0.0
    success_rate: float = 0.0
    successful_episodes: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    episodes: list[EpisodeStats] = field(default_factory=list)


class RunManager:
    """
    Manages run statistics for AgentRing environments.

    In remote mode, syncs with the server after each episode to ensure
    data consistency. Server data always overrides local data.
    """

    def __init__(self, mode: str = "local") -> None:
        """Initialize the run manager.

        Args:
            mode: Either "local" or "remote"
        """
        self.mode = mode.lower()
        self.stats: RunStats | None = None
        self._current_episode_stats: EpisodeStats | None = None
        self._sync_callback: Any | None = None

    def set_sync_callback(self, callback: Any) -> None:
        """Set a callback function to sync with server (remote mode only).

        Args:
            callback: Function that returns server statistics dict
        """
        self._sync_callback = callback

    def start_run(self, run_id: str, env_id: str) -> None:
        """Start a new run.

        Args:
            run_id: Unique identifier for the run
            env_id: Environment ID
        """
        self.stats = RunStats(
            run_id=run_id,
            env_id=env_id,
            state="running",
            start_time=datetime.now(),
        )
        self._current_episode_stats = None

    def start_episode(self, episode_num: int) -> None:
        """Start a new episode.

        Args:
            episode_num: Episode number
        """
        if self.stats is None:
            raise RuntimeError("Run not started. Call start_run() first.")

        # Finalize previous episode if exists
        if self._current_episode_stats is not None:
            self._finalize_episode()

        self._current_episode_stats = EpisodeStats(
            episode_num=episode_num,
            start_time=datetime.now(),
        )
        self.stats.current_episode = episode_num

    def record_step(
        self,
        reward: float,
        done: bool = False,
        truncated: bool = False,
    ) -> None:
        """Record a step in the current episode.

        Args:
            reward: Reward received
            done: Whether episode is done
            truncated: Whether episode was truncated
        """
        if self._current_episode_stats is None:
            return

        self._current_episode_stats.total_steps += 1
        self._current_episode_stats.total_reward += reward
        self._current_episode_stats.done = done
        self._current_episode_stats.truncated = truncated

        if self.stats:
            self.stats.total_steps += 1
            self.stats.total_reward += reward

    def _finalize_episode(self) -> None:
        """Finalize the current episode."""
        if self._current_episode_stats is None or self.stats is None:
            return

        self._current_episode_stats.end_time = datetime.now()
        self.stats.episodes.append(self._current_episode_stats)
        self.stats.completed_episodes += 1

        # Update aggregated stats
        if self.stats.completed_episodes > 0:
            self.stats.average_reward = self.stats.total_reward / self.stats.completed_episodes
            self.stats.average_steps = self.stats.total_steps / self.stats.completed_episodes

        # Count successful episodes (done and not truncated)
        if self._current_episode_stats.done and not self._current_episode_stats.truncated:
            self.stats.successful_episodes += 1

        if self.stats.completed_episodes > 0:
            self.stats.success_rate = self.stats.successful_episodes / self.stats.completed_episodes

        # In remote mode, sync with server after episode completion
        if self.mode == "remote" and self._sync_callback:
            self._sync_from_server()

        self._current_episode_stats = None

    def _sync_from_server(self) -> None:
        """Sync statistics from server (server always wins)."""
        if not self._sync_callback or self.stats is None:
            return

        try:
            server_stats = self._sync_callback()
            if not server_stats or not isinstance(server_stats, dict):
                return

            # Update run-level stats from server
            if "run_id" in server_stats:
                self.stats.run_id = server_stats["run_id"]
            if "state" in server_stats:
                self.stats.state = server_stats["state"]
            if "current_episode" in server_stats:
                self.stats.current_episode = server_stats["current_episode"]
            if "completed_episodes" in server_stats:
                self.stats.completed_episodes = server_stats["completed_episodes"]
            if "total_steps" in server_stats:
                self.stats.total_steps = server_stats["total_steps"]
            if "total_reward" in server_stats:
                self.stats.total_reward = server_stats["total_reward"]
            if "average_reward" in server_stats:
                self.stats.average_reward = server_stats["average_reward"]
            if "average_steps" in server_stats:
                self.stats.average_steps = server_stats["average_steps"]
            if "success_rate" in server_stats:
                self.stats.success_rate = server_stats["success_rate"]
            if "successful_episodes" in server_stats:
                self.stats.successful_episodes = server_stats["successful_episodes"]

            # Parse start_time if provided
            if "start_time" in server_stats and server_stats["start_time"]:
                try:
                    if isinstance(server_stats["start_time"], str):
                        self.stats.start_time = datetime.fromisoformat(
                            server_stats["start_time"].replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    pass

            # Parse end_time if provided
            if "end_time" in server_stats and server_stats["end_time"]:
                try:
                    if isinstance(server_stats["end_time"], str):
                        self.stats.end_time = datetime.fromisoformat(
                            server_stats["end_time"].replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    pass

            # Update episode list from server (server always wins)
            if "episodes" in server_stats and isinstance(server_stats["episodes"], list):
                self.stats.episodes = []
                for ep_data in server_stats["episodes"]:
                    if isinstance(ep_data, dict):
                        ep_stats = EpisodeStats(
                            episode_num=ep_data.get("episode_num", 0),
                            total_steps=ep_data.get("total_steps", 0),
                            total_reward=ep_data.get("total_reward", 0.0),
                            done=ep_data.get("done", False),
                            truncated=ep_data.get("truncated", False),
                        )

                        # Parse timestamps if available
                        if "start_time" in ep_data and ep_data["start_time"]:
                            try:
                                if isinstance(ep_data["start_time"], str):
                                    ep_stats.start_time = datetime.fromisoformat(
                                        ep_data["start_time"].replace("Z", "+00:00")
                                    )
                            except (ValueError, TypeError):
                                pass

                        if "end_time" in ep_data and ep_data["end_time"]:
                            try:
                                if isinstance(ep_data["end_time"], str):
                                    ep_stats.end_time = datetime.fromisoformat(
                                        ep_data["end_time"].replace("Z", "+00:00")
                                    )
                            except (ValueError, TypeError):
                                pass

                        self.stats.episodes.append(ep_stats)

        except Exception:
            # If sync fails, continue with local stats
            pass

    def get_statistics(self) -> dict[str, Any]:
        """Get current run statistics.

        Returns:
            Dictionary with run statistics
        """
        if self.stats is None:
            return {
                "run_id": None,
                "state": "idle",
                "current_episode": 0,
                "completed_episodes": 0,
                "total_steps": 0,
                "total_reward": 0.0,
                "average_reward": 0.0,
                "average_steps": 0.0,
                "success_rate": 0.0,
                "successful_episodes": 0,
            }

        return {
            "run_id": self.stats.run_id,
            "env_id": self.stats.env_id,
            "state": self.stats.state,
            "current_episode": self.stats.current_episode,
            "completed_episodes": self.stats.completed_episodes,
            "total_steps": self.stats.total_steps,
            "total_reward": self.stats.total_reward,
            "average_reward": self.stats.average_reward,
            "average_steps": self.stats.average_steps,
            "success_rate": self.stats.success_rate,
            "successful_episodes": self.stats.successful_episodes,
            "start_time": (self.stats.start_time.isoformat() if self.stats.start_time else None),
            "end_time": (self.stats.end_time.isoformat() if self.stats.end_time else None),
            "episodes": [
                {
                    "episode_num": ep.episode_num,
                    "total_steps": ep.total_steps,
                    "total_reward": ep.total_reward,
                    "done": ep.done,
                    "truncated": ep.truncated,
                    "start_time": (ep.start_time.isoformat() if ep.start_time else None),
                    "end_time": ep.end_time.isoformat() if ep.end_time else None,
                }
                for ep in self.stats.episodes
            ],
        }

    def format_statistics(self, show_episode_details: bool = False) -> str:
        """Format run statistics as a human-readable string.

        Args:
            show_episode_details: If True, show all episode details. If False, show summary only.

        Returns:
            Formatted string with run statistics
        """
        stats = self.get_statistics()
        lines = []

        lines.append("=" * 60)
        lines.append("Run Statistics:")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Run ID: {stats.get('run_id', 'N/A')}")
        lines.append(f"Environment: {stats.get('env_id', 'N/A')}")
        lines.append(f"State: {stats.get('state', 'N/A').upper()}")
        lines.append("")

        lines.append("Summary:")
        lines.append(f"  Completed Episodes: {stats.get('completed_episodes', 0)}")
        lines.append(f"  Total Steps: {stats.get('total_steps', 0)}")
        lines.append(f"  Total Reward: {stats.get('total_reward', 0.0):.2f}")
        lines.append("")

        if stats.get("completed_episodes", 0) > 0:
            lines.append("Averages:")
            lines.append(f"  Average Reward per Episode: {stats.get('average_reward', 0.0):.4f}")
            lines.append(f"  Average Steps per Episode: {stats.get('average_steps', 0.0):.2f}")
            lines.append(f"  Success Rate: {stats.get('success_rate', 0.0):.1%}")
            lines.append(f"  Successful Episodes: {stats.get('successful_episodes', 0)}")
            lines.append("")

        # Show episode details if requested
        episodes = stats.get("episodes", [])
        if show_episode_details and episodes:
            lines.append("Episode Details:")
            for ep in episodes:
                reward_str = (
                    f"{ep.get('total_reward', 0.0):.2f}"
                    if ep.get("total_reward", 0.0) > 0
                    else "0.00"
                )
                status = "✓" if ep.get("done") and not ep.get("truncated") else "✗"
                lines.append(
                    f"  Episode {ep.get('episode_num', 0):3d}: {ep.get('total_steps', 0):3d} steps, "
                    f"reward {reward_str}, {status}"
                )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the run manager."""
        self.stats = None
        self._current_episode_stats = None
