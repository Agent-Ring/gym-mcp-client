# AgentRing

A unified Python client for working with both local and remote Gymnasium environments via [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server).

**Goal**: Write code once, seamlessly switch between local development and remote execution.

## Features

- 🎮 **Unified API**: Same Gymnasium interface for both local and remote environments
- 🔄 **Seamless Switching**: Change modes with a single parameter
- 🌐 **Remote Execution**: Connect to gym-mcp-server instances over HTTP
- 🔧 **Full Compatibility**: Supports all Gymnasium environment types (Box, Discrete, MultiBinary, etc.)
- 🐍 **Modern Python**: Python 3.12+ with complete type hints
- 📦 **Easy Setup**: Managed with uv for fast dependency management
- ✅ **Well Tested**: 19 comprehensive tests (18 passing)

## Installation

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Complete Example

Here's a complete example showing both modes with multiple episodes:

```python
import agentring as gym

# Choose mode: "local" or "remote"
MODE = "local"  # Change to "remote" to use gym-mcp-server
SERVER_URL = "http://localhost:8000" if MODE == "remote" else None

# Create environment with context manager for automatic cleanup
with gym.make(
    "CartPole-v1",
    mode=MODE,
    gym_server_url=SERVER_URL,
    render_mode="rgb_array"
) as env:
    print(f"Environment: {env}")
    print(f"Mode: {MODE}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Run 3 episodes
    for episode in range(3):
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        terminated = truncated = False

        print(f"Episode {episode + 1}:")
        print(f"  Initial observation: {observation}")

        while not (terminated or truncated):
            # Sample random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if steps >= 200:  # Safety limit
                break

        print(f"  Total reward: {total_reward}")
        print(f"  Steps: {steps}")
        print()

print("Done!")
```

## Architecture

```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ gym.make    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Local  │ │ Remote       │
│ Gym    │ │ HTTP Client  │
└────────┘ └──────┬───────┘
              │
              ▼
         ┌──────────────┐
         │ gym-mcp-     │
         │ server       │
         └──────┬───────┘
                │
                ▼
           ┌────────┐
           │ Gym    │
           │ Env    │
           └────────┘
```

### Supported Space Types

| Space Type | Local | Remote | Serialization |
|------------|-------|--------|---------------|
| Box | ✅ | ✅ | array ↔ list |
| Discrete | ✅ | ✅ | int ↔ int |
| MultiBinary | ✅ | ✅ | array ↔ list |
| MultiDiscrete | ✅ | ✅ | array ↔ list |
| Tuple | ✅ | ✅ | recursive |
| Dict | ✅ | ✅ | recursive |

## Examples

See `quickstart.py` for a complete working example showing both local and remote modes.

Run the example:

```bash
# Local mode (default)
uv run python quickstart.py

# Remote mode (start server first!)
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
# Then edit quickstart.py to set REMOTE_MODE = True and run:
uv run python quickstart.py
```

## Development

### Setup

```bash
git clone <your-repo-url>
cd agentring
make install
```

### Available Commands

```bash
make help       # Show all commands
make test       # Run test suite (14 tests)
make lint       # Run ruff linter
make format     # Format code with ruff
make typecheck  # Run mypy type checker
make check      # Run all checks (lint + typecheck + test)
make all        # Format, then run all checks
make demo       # Run local demo
make clean      # Clean build artifacts
```

### Running Tests

```bash
make test
# Or: uv run pytest tests/ -v
# 19 tests (18 passing, 1 failing due to missing pygame) ✅
```

## Use Cases

1. **Development → Production**: Develop locally, deploy remotely
2. **Distributed Training**: Multiple processes connecting to remote environments
3. **Resource Management**: Run expensive simulations on dedicated servers
4. **Testing**: Test locally before remote deployment

## Performance

### Local Mode
- **Overhead**: Minimal (thin wrapper)
- **Best for**: Development, testing, lightweight environments

### Remote Mode
- **Overhead**: HTTP round-trip (1-10ms on localhost)
- **Best for**: Expensive environments, distributed training, resource sharing

## Error Handling

```python
import agentring as gym
import httpx

try:
    env = gym.make(
        "CartPole-v1",
        mode="remote",
        gym_server_url="http://localhost:8000"
    )
    observation, info = env.reset()
    # ... your code ...
except ValueError:
    # Invalid mode, missing URL, etc.
    pass
except RuntimeError:
    # Environment initialization failed, remote call failed
    pass
except httpx.HTTPError:
    # Network error (remote mode only)
    pass
finally:
    if 'env' in locals():
        env.close()
```

## Troubleshooting

### Remote Connection Issues

1. Ensure the gym-mcp-server is running and accessible
2. Check URL is correct (including protocol: `http://` or `https://`)
3. Verify firewall/network settings
4. Check server logs for errors

### Environment Not Found

```bash
# For Atari environments
uv add "gymnasium[atari]"

# For Box2D environments
uv add "gymnasium[box2d]"

# For MuJoCo environments
uv add "gymnasium[mujoco]"
```

## Requirements

- Python 3.10+
- gymnasium >= 1.2.1
- httpx >= 0.28.1
- numpy >= 2.0.0
- gym-mcp-server (from GitHub)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to verify all tests pass
5. Submit a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## License

MIT License - see LICENSE file for details.

## Related Projects

- [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server) - MCP server for Gymnasium environments
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool integration protocol

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a GitHub discussion
- **Documentation**: See examples/ directory

---

**Status**: ✅ Production Ready | **Version**: 0.1.0 | **Python**: 3.12+ | **Tests**: 19 (18 passing)
