# Gym MCP Client

A unified Python interface for working with both local and remote Gymnasium environments via [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server).

**Goal**: Write code once, seamlessly switch between local development and remote execution.

## Features

- 🎮 **Unified API**: Same Gymnasium interface for both local and remote environments
- 🔄 **Seamless Switching**: Change modes with a single parameter
- 🌐 **Remote Execution**: Connect to gym-mcp-server instances over HTTP
- 🔧 **Full Compatibility**: Supports all Gymnasium environment types (Box, Discrete, MultiBinary, etc.)
- 🐍 **Modern Python**: Python 3.12+ with complete type hints
- 📦 **Easy Setup**: Managed with uv for fast dependency management
- ✅ **Well Tested**: 14 comprehensive tests, all passing

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd agentring

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Local Mode

```python
from agentring import GymMCPClient

# Create a local environment
env = GymMCPClient("CartPole-v1", mode="local")

# Use standard Gymnasium API
observation, info = env.reset(seed=42)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Remote Mode

First, start a gym-mcp-server:

```bash
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
```

Then connect to it:

```python
from agentring import GymMCPClient

# Create a remote environment
env = GymMCPClient(
    "CartPole-v1",
    mode="remote",
    gym_server_url="http://localhost:8000"
)

# Use the exact same API as local mode!
observation, info = env.reset(seed=42)
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Context Manager

```python
from agentring import GymMCPClient

# Automatic cleanup
with GymMCPClient("CartPole-v1", mode="local") as env:
    observation, info = env.reset()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # env.close() called automatically
```

## Switching Between Modes

The main benefit: write once, run anywhere!

```python
import os
from agentring import GymMCPClient

# Configuration from environment variables
MODE = os.getenv("GYM_MODE", "local")
SERVER_URL = os.getenv("GYM_SERVER_URL", "http://localhost:8000")

# Create environment based on mode
if MODE == "local":
    env = GymMCPClient("CartPole-v1", mode="local")
else:
    env = GymMCPClient(
        "CartPole-v1",
        mode="remote",
        gym_server_url=SERVER_URL,
        gym_server_key=os.getenv("GYM_API_KEY")
    )

# Your code works the same regardless of mode!
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

Switch modes via command line:

```bash
# Run locally
export GYM_MODE=local
python your_training_script.py

# Run remotely
export GYM_MODE=remote
export GYM_SERVER_URL=http://gpu-server:8000
python your_training_script.py
```

## API Reference

### `GymMCPClient`

```python
GymMCPClient(
    env_id: str,
    mode: str = "local",
    render_mode: str | None = None,
    gym_server_url: str | None = None,
    gym_server_key: str | None = None,
    **kwargs
)
```

**Parameters:**
- `env_id`: Gymnasium environment ID (e.g., "CartPole-v1")
- `mode`: Either "local" or "remote"
- `render_mode`: Render mode ("rgb_array", "human", etc.)
- `gym_server_url`: URL of gym-mcp-server (required for remote mode)
- `gym_server_key`: Optional API key for authentication
- `**kwargs`: Additional arguments passed to `gym.make()` in local mode

**Methods:**
- `reset(seed=None, options=None)` → (observation, info)
- `step(action)` → (observation, reward, terminated, truncated, info)
- `render()` → render_output
- `close()` → None

**Properties:**
- `observation_space`: The observation space
- `action_space`: The action space
- `reward_range`: The reward range
- `metadata`: Environment metadata

## Architecture

```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GymMCPClient    │
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

See the `examples/` directory:
- `local_example.py`: Local mode usage
- `remote_example.py`: Remote mode usage
- `context_manager_example.py`: Context manager pattern

Run examples:

```bash
# Local mode
uv run python examples/local_example.py

# Remote mode (start server first!)
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
uv run python examples/remote_example.py

# Demo CLI tool
python main.py --mode local --episodes 3
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
# 14 tests, all passing ✅
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
from agentring import GymMCPClient
import httpx

try:
    env = GymMCPClient(
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

- Python 3.12+
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

**Status**: ✅ Production Ready | **Version**: 0.1.0 | **Python**: 3.12+
