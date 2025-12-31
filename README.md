# AgentRing

A unified Python client for working with both local and remote Gymnasium environments via [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server).

**Goal**: Write code once, seamlessly switch between local development and remote execution.

## Features

- 🎮 **Unified API**: Same Gymnasium interface for both local and remote environments
- 🔄 **Seamless Switching**: Change modes with a single parameter
- 🌐 **Remote Execution**: Connect to gym-mcp-server instances over HTTP
- 🤖 **MCP Extensions**: SDK-agnostic tool generation with autocomplete-enabled access (SDK-agnostic)
- 🔧 **Full Compatibility**: Supports all Gymnasium environment types (Box, Discrete, MultiBinary, etc.)
- 🐍 **Modern Python**: Python 3.10+ with complete type hints
- 📦 **Easy Setup**: Managed with uv for fast dependency management
- 🌐 **Unified Configuration**: Environment variables for server URLs
- 💡 **IDE Autocomplete**: ToolCollection provides both `tools["name"]` and `tools.name` access
- 📊 **Result Analysis**: Comprehensive episode statistics and export capabilities
- ✅ **Well Tested**: 19 core tests + 7 MCP extension test suites

## Installation

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Environment Variables

AgentRing supports environment variables for configuration:

### GYM_SERVER_URL
Set the default MCP server URL for both `gym.make()` and `gym_mcp.create_tools()`:

```bash
export GYM_SERVER_URL="http://localhost:8070"
```

This allows you to:
- Use `gym.make("CartPole-v1", mode="remote")` without specifying `gym_server_url`
- Use `gym_mcp.create_tools()` without specifying `server_url`
- Change server URLs across your entire project by updating one environment variable

## Quick Start

### Basic Usage

#### Local Mode (Standard Gymnasium)
```python
import agentring as gym

# Create local environment (same as gymnasium.make)
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
action = env.action_space.sample()  # Random action
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

#### Remote Mode (MCP Server)
```python
import agentring as gym

# Create remote environment via MCP server
env = gym.make(
    "CartPole-v1",
    mode="remote",
    gym_server_url="http://localhost:8000"
)

observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### MCP Extensions Quick Start

AgentRing's MCP extensions dramatically simplify agent development:

#### Environment Variable Configuration (Recommended)

Set the server URL once for your entire project:
```bash
export GYM_SERVER_URL="http://localhost:8070"
```

#### 1. Generate Tools (SDK-Agnostic)
```python
import agentring.mcp as gym_mcp

# One line to get all environment tools (uses GYM_SERVER_URL)
tools = gym_mcp.create_tools()

# Or specify URL explicitly
tools = gym_mcp.create_tools("http://localhost:8070")

# Tools are accessed by name (dict return)
reset_result = tools["reset_env"](seed=42)  # reset_env
step_result = tools["step_env"](action="go north")  # step_env
```

#### 2. Use with Any SDK
```python
# With CrewAI
from crewai import Agent
from crewai.tools import tool

@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action: str):
    return tools["step_env"](action=action)

agent = Agent(tools=[reset_env, step_env], ...)
```

#### 3. SDK-Native Episode Execution
```python
# Agent SDKs handle episode execution natively
# AgentRing provides tools and result collection

# Example with CrewAI
from crewai import Agent, Task, Crew

agent = Agent(tools=[reset_env, step_env, ...], ...)
task = Task(description="Complete the quest", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

# Collect results with AgentRing
episode_results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.8, 12, True)
])
print(episode_results.summary())
```

### Complete Examples by SDK

#### CrewAI + TextWorld
```python
import agentring.mcp as gym_mcp
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# Wrap for CrewAI
@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action: str):
    return tools["step_env"](action=action)

# Create agent
agent = Agent(
    role="Text Adventure Agent",
    goal="Complete quests in text environments",
    backstory="You are skilled at solving puzzles and exploring.",
    tools=[reset_env, step_env],
    verbose=True
)

# Run task
task = Task(
    description="Find the treasure and escape the dungeon",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

#### LangGraph + ALFWorld
```python
import agentring.mcp as gym_mcp
from langchain_core.tools import tool
from langgraph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8090")

# Wrap for LangChain
@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action: str):
    return tools["step_env"](action=action)

# Create LangGraph workflow
def agent_node(state):
    # Your LLM logic here
    return {"messages": state["messages"] + ["response"]}

workflow = StateGraph()
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([reset_env, step_env]))
workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", lambda s: END if s.get("done") else "tools")

app = workflow.compile()
result = app.invoke({"messages": ["Complete the household task"]})
```

#### Google ADK + WebShop
```python
import agentring.mcp as gym_mcp
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8002")

# Wrap for Google ADK
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

def step_env(action: str):
    return tools["step_env"](action=action)

reset_tool = FunctionTool(reset_env)
step_tool = FunctionTool(step_env)

# Create agent
agent = LlmAgent(
    name="ShoppingAgent",
    description="An agent that shops efficiently",
    model="gemini-2.0-flash-exp",
    instruction=gym_mcp.templates.SHOPPING_INSTRUCTIONS,
    tools=[reset_tool, step_tool],
)

# Run episode
import asyncio
async for result in agent.run_async("Buy the best laptop for under $1000"):
    print(result.text)
```

#### Multi-Server Example
```python
import agentring.mcp as gym_mcp

# Connect to multiple environments
multi_client = gym_mcp.MultiServerClient()
multi_client.add_server("textworld", "http://localhost:8070")
multi_client.add_server("alfworld", "http://localhost:8090")
multi_client.add_server("webshop", "http://localhost:8002")

# Get tools from all servers
all_tools = multi_client.get_all_tools()

# Check server health
health = multi_client.health_check_all()
print(f"Server health: {health}")

# Run agents across different environments
textworld_tools = multi_client.get_tools("textworld")
alfworld_tools = multi_client.get_tools("alfworld")

# SDKs handle episode execution natively
# Use AgentRing for result collection and analysis

# Example: Run TextWorld agent
# result = your_textworld_agent.run("Complete the quest")
# Collect results with AgentRing
tw_results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.8, 12, True),
    # ... more episode results
])

aw_results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.6, 15, True),
    # ... more episode results
])

print("TextWorld:", tw_results.success_percentage, "% success")
print("ALFWorld:", aw_results.success_percentage, "% success")
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

## MCP Extensions for Agent Development

AgentRing now includes powerful MCP (Model Context Protocol) extensions that dramatically simplify agent development with MCP servers. These extensions provide generic, SDK-agnostic tools and utilities.

### Features

- 🤖 **Generic Tool Factory**: Auto-generate callable tools from MCP servers (works with any agent SDK)
- 🎯 **SDK Agnostic**: No SDK dependencies - tools are standard Python callables
- 🚀 **Episode Runner**: Unified episode execution and result collection
- 🔧 **Format Converters**: Convert tool definitions to JSON Schema, OpenAPI, and SDK-specific formats
- 🌐 **Multi-Server Support**: Work with multiple MCP servers simultaneously
- 📊 **Result Analysis**: Comprehensive episode result statistics and export capabilities

### Quick Start

```python
import agentring.mcp as gym_mcp

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")
# Returns: List of callable Python functions

# 2. Use with any agent SDK
# Example with CrewAI:
from crewai import Agent, Task, Crew
from crewai.tools import tool

@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action):
    return tools["step_env"](action=action)

agent = Agent(tools=[reset_env, step_env], ...)
task = Task(description="Complete the household task", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

# 3. SDKs handle episode execution natively
# AgentRing provides tools and result collection
# Use AgentRing's EpisodeResults for analysis:
results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.8, 12, True),
    # ... collect results from SDK execution
])
print(results.summary())
```

### MCP Tool Factory

The `create_tools()` function automatically discovers available tools from MCP servers and returns a `ToolCollection` that provides both dict-style and attribute-style access with IDE autocomplete:

```python
# Generate all tools from server
tools = gym_mcp.create_tools("http://localhost:8070")

# Generate specific tools
tools = gym_mcp.create_tools("http://localhost:8070", ["reset_env", "step_env"])

# Tools support both dict-style and attribute-style access
result = tools["reset_env"](seed=42)  # Dict-style access
result = tools.reset_env(seed=42)     # Attribute access with autocomplete!

# Both access methods provide the same callable function
assert tools["reset_env"] is tools.reset_env  # True
```

### SDK-Native Episode Execution

Agent SDKs handle episode execution natively. AgentRing provides tools and result collection:

```python
# Example with CrewAI
from crewai import Agent, Task, Crew

agent = Agent(tools=[reset_env, step_env], ...)
task = Task(description="Complete the task", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()  # SDK handles execution

# Use AgentRing for result collection and analysis
from agentring.mcp import results, types

episode_results = results.EpisodeResults([
    types.EpisodeResult(1, 0.8, 12, True),
    # ... collect from actual runs
])
print(f"Success rate: {episode_results.success_percentage:.1f}%")
```

### Format Converters

Convert tool definitions to various formats for SDK integration:

```python
from agentring.mcp import formats

# Convert to JSON Schema (OpenAI style)
json_schema = formats.to_json_schema(tool_definition)

# Convert to OpenAPI spec
openapi_spec = formats.to_openapi_spec(tool_definition)

# Convert to SDK-specific formats
crewai_format = formats.to_crewai_tool(tool_definition)
langchain_format = formats.to_langchain_tool(tool_definition)
```

### Multi-Server Support

Work with multiple MCP servers:

```python
multi_client = gym_mcp.MultiServerClient()
multi_client.add_server("textworld", "http://localhost:8070")
multi_client.add_server("alfworld", "http://localhost:8090")

# Get tools from all servers
all_tools = multi_client.get_all_tools()

# Health check all servers
health = multi_client.health_check_all()
print(f"Healthy servers: {multi_client.get_healthy_servers()}")
```

### Agent Templates

Pre-built instruction templates for common agent patterns:

```python
from agentring.mcp import templates

# Get templates
text_adventure_prompt = templates.TEXT_ADVENTURE_INSTRUCTIONS
shopping_prompt = templates.SHOPPING_INSTRUCTIONS
household_prompt = templates.HOUSEHOLD_INSTRUCTIONS

# Create complete agent configurations
config = templates.create_text_adventure_config(
    max_steps=50,
    custom_instructions="Always examine objects before using them."
)
```

### Result Analysis

Comprehensive episode result analysis and export:

```python
results = runner.run_episodes(episodes=20)

# Statistics
print(f"Success rate: {results.success_percentage:.1f}%")
print(f"Average reward: {results.average_reward:.2f}")
print(f"Average steps: {results.average_steps:.1f}")

# Export results
results.save_json("results.json")
results.save_csv("results.csv")

# Filter and analyze
successful_episodes = results.filter_by_success(successful_only=True)
high_reward_episodes = results.filter_by_reward(min_reward=1.0)
```

### SDK Integration Examples

#### With CrewAI

```python
import agentring.mcp as gym_mcp
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# Wrap for CrewAI
@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action):
    return tools["step_env"](action=action)

agent = Agent(
    role="Text Adventure Agent",
    goal="Complete quests in text worlds",
    backstory="You excel at solving puzzles and exploring environments.",
    tools=[reset_env, step_env]
)

task = Task(description="Find the treasure and escape the dungeon", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

#### With LangGraph

```python
import agentring.mcp as gym_mcp
from langchain_core.tools import tool
from langgraph import StateGraph

# Generate and wrap tools
tools = gym_mcp.create_tools("http://localhost:8070")

@tool
def reset_env(seed=None):
    return tools["reset_env"](seed=seed)

@tool
def step_env(action):
    return tools["step_env"](action=action)

# Use in LangGraph workflow
# ... workflow definition ...
```

#### With Generic Agent

```python
import agentring.mcp as gym_mcp

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# Define custom agent class
class MyCustomAgent:
    def __init__(self, tools):
        self.tools = tools
        self.episode_results = []

    def run_episode(self, prompt: str, max_steps=10):
        # Custom episode execution logic
        # Use tools["reset_env"](), tools["step_env"](), etc.
        total_reward = 0.0
        steps = 0

        # Reset environment
        reset_result = self.tools["reset_env"](seed=42)

        # Your agent logic here
        for step in range(max_steps):
            # Agent decision making
            action = "look"  # Your logic here

            # Execute action
            step_result = self.tools["step_env"](action=action)
            reward = step_result.get("reward", 0)
            total_reward += reward
            steps += 1

            if step_result.get("done"):
                break

        # Store result
        result = gym_mcp.types.EpisodeResult(1, total_reward, steps, total_reward > 0)
        self.episode_results.append(result)
        return result

    def get_results(self):
        return gym_mcp.results.EpisodeResults(self.episode_results)

# Use the custom agent
agent = MyCustomAgent(tools)
agent.run_episode("Complete the task")
results = agent.get_results()
```

## SDK Integration Guide

AgentRing's MCP extensions work with any agent SDK. Here are comprehensive examples for popular frameworks:

### CrewAI Integration

CrewAI agents can use AgentRing MCP tools with minimal code changes.

#### Basic CrewAI Agent

```python
import agentring.mcp as gym_mcp
from crewai import Agent, Task, Crew
from crewai.tools import tool

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Wrap tools for CrewAI (simple adapters)
@tool
def reset_env(seed=None):
    """Reset the environment to start a new episode."""
    return tools["reset_env"](seed=seed)

@tool
def step_env(action: str):
    """Take an action in the environment."""
    return tools["step_env"](action=action)

@tool
def get_env_info():
    """Get information about the environment."""
    return tools["get_env_info"]()

# 3. Create CrewAI agent
agent = Agent(
    role="Text Adventure Agent",
    goal="Complete quests and solve puzzles in text-based environments",
    backstory="""You are an expert at playing text adventure games.
    You carefully read descriptions, make logical decisions, and
    systematically explore environments to achieve objectives.""",
    tools=[reset_env, step_env, get_env_info],
    verbose=True,
    allow_delegation=False
)

# 4. Create and run task
task = Task(
    description="""Navigate the environment, find the treasure,
    and return it to the starting location. Be methodical and
    examine objects before using them.""",
    agent=agent,
    expected_output="A summary of the completed quest"
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)
result = crew.kickoff()
```

#### Advanced CrewAI with Multiple Agents

```python
import agentring.mcp as gym_mcp
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Generate tools from multiple servers
textworld_tools = gym_mcp.create_tools("http://localhost:8070")
alfworld_tools = gym_mcp.create_tools("http://localhost:8090")

# Wrap tools
@tool
def reset_textworld(seed=None):
    return textworld_tools["reset_env"](seed=seed)

@tool
def step_textworld(action: str):
    return textworld_tools["step_env"](action=action)

@tool
def reset_alfworld(seed=None):
    return alfworld_tools["reset_env"](seed=seed)

@tool
def step_alfworld(action: str):
    return alfworld_tools["step_env"](action=action)

# Create specialized agents
textworld_agent = Agent(
    role="Text Adventure Expert",
    goal="Solve text-based puzzles and quests",
    tools=[reset_textworld, step_textworld],
    verbose=True
)

alfworld_agent = Agent(
    role="Household Task Expert",
    goal="Complete household tasks efficiently",
    tools=[reset_alfworld, step_alfworld],
    verbose=True
)

# Create crew with multiple agents
crew = Crew(
    agents=[textworld_agent, alfworld_agent],
    tasks=[
        Task(description="Solve the treasure quest", agent=textworld_agent),
        Task(description="Clean the kitchen", agent=alfworld_agent)
    ],
    verbose=True
)

result = crew.kickoff()
```

### LangGraph Integration

LangGraph workflows can use AgentRing MCP tools as LangChain tools.

#### Basic LangGraph Agent

```python
import agentring.mcp as gym_mcp
from langchain_core.tools import tool
from langgraph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Wrap tools for LangChain
@tool
def reset_env(seed: int = None) -> str:
    """Reset the environment to start a new episode."""
    result = tools["reset_env"](seed=seed)
    return f"Environment reset: {result}"

@tool
def step_env(action: str) -> str:
    """Take an action in the environment."""
    result = tools["step_env"](action=action)
    return f"Action result: {result}"

# 3. Define state
class AgentState(TypedDict):
    messages: list
    step_count: int
    total_reward: float
    done: bool

# 4. Create workflow
def agent_node(state: AgentState):
    # Agent logic here (LLM call with tools)
    messages = state["messages"]
    # ... LLM call with tool calling ...
    return {"messages": messages, "step_count": state["step_count"] + 1}

def should_continue(state: AgentState) -> str:
    if state["done"] or state["step_count"] >= 50:
        return END
    return "tools"

# 5. Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode([reset_env, step_env]))

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue)

app = workflow.compile()

# 6. Run workflow
initial_state = {
    "messages": [{"role": "user", "content": "Complete the text adventure quest"}],
    "step_count": 0,
    "total_reward": 0.0,
    "done": False
}

result = app.invoke(initial_state)
print(f"Workflow completed with {result['step_count']} steps")
```

#### LangGraph with AgentRing MCP Runner

```python
import agentring.mcp as gym_mcp
import asyncio
from langchain_core.language_models import BaseLanguageModel

# 1. Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Create LangGraph agent interface
async def langgraph_agent(prompt: str) -> str:
    # Your LangGraph agent logic here
    # This would integrate with your LangGraph setup
    return "Agent response with tool calls"

# 3. Integrate with LangGraph workflow
# Add tools to your LangGraph workflow nodes
# Your LangGraph handles episode execution natively

# 4. Collect results
results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.85, 14, True),
    # ... collect from LangGraph execution
])
print(results.summary())
```

### Google ADK Integration

Google ADK (Agent Development Kit) works seamlessly with AgentRing MCP tools.

#### Basic Google ADK Agent

```python
import agentring.mcp as gym_mcp
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Wrap tools for Google ADK
def reset_env_adk(seed=None):
    """Reset the environment to start a new episode."""
    return tools["reset_env"](seed=seed)

def step_env_adk(action: str):
    """Take an action in the environment."""
    return tools["step_env"](action=action)

def get_env_info_adk():
    """Get information about the environment."""
    return tools["get_env_info"]()

# Create FunctionTool instances
reset_tool = FunctionTool(reset_env_adk)
step_tool = FunctionTool(step_env_adk)
info_tool = FunctionTool(get_env_info_adk)

# 3. Create ADK agent
agent = LlmAgent(
    name="TextWorldAgent",
    description="An agent that plays TextWorld text adventure games",
    model="gemini-2.0-flash-exp",
    instruction=gym_mcp.templates.TEXT_ADVENTURE_INSTRUCTIONS,
    tools=[reset_tool, step_tool, info_tool],
)

# 4. Run episodes (async)
async def run_adk_episodes():
    results = []
    for episode in range(3):
        result = await agent.run_async(f"Episode {episode + 1}: Complete the quest")
        results.append(result)
    return results

# Run the episodes
import asyncio
episode_results = asyncio.run(run_adk_episodes())
```

#### Advanced Google ADK with AgentRing Runner

```python
import agentring.mcp as gym_mcp
import asyncio

# 1. Generate tools and create ADK agent
tools = gym_mcp.create_tools("http://localhost:8070")

# Create ADK agent (same as above)
agent = LlmAgent(
    name="TextWorldAgent",
    description="An agent that plays TextWorld text adventure games",
    model="gemini-2.0-flash-exp",
    instruction=gym_mcp.templates.TEXT_ADVENTURE_INSTRUCTIONS,
    tools=[reset_tool, step_tool, info_tool],
)

# 2. Create ADK agent interface for AgentRing runner
async def adk_agent_interface(prompt: str) -> str:
    """Adapter to use ADK agent with AgentRing runner."""
    # Handle ADK's async generator response
    async for result in agent.run_async(prompt):
        return result.text
    return "No response from agent"

# 3. Run episodes (ADK handles execution natively)
# ADK agents execute episodes using their built-in run_async method
# with integrated tool calling

# 4. Collect results for analysis
results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.82, 16, True),
    # ... collect from ADK execution
])
print(results.summary())
```

### OpenAI Agents SDK Integration

OpenAI Agents SDK has native MCP support, making integration even simpler.

#### Basic OpenAI Agents with MCP

```python
from agents import Agent, Runner, ModelSettings
from agents.mcp import MCPServerStreamableHttp

# 1. Create MCP server connection (no AgentRing needed for basic usage)
server = MCPServerStreamableHttp(
    name="Gym Environment",
    params={"url": "http://localhost:8070/mcp", "timeout": 10},
)

# 2. Create agent with MCP server
agent = Agent(
    name="GymAgent",
    instructions="You are an agent playing in a Gym environment. Complete tasks efficiently.",
    mcp_servers=[server],
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.1),
)

# 3. Run agent
result = await Runner.run(agent, "Complete the text adventure quest")
print(result.final_output)
```

#### OpenAI Agents with AgentRing MCP Extensions

```python
import agentring.mcp as gym_mcp
from agents import Agent, Runner, ModelSettings

# 1. Use AgentRing to get MCP server connection
client = gym_mcp.MCPServerClient("http://localhost:8070")
server = MCPServerStreamableHttp(
    name="Gym Environment",
    params={"url": f"{client.server_url}/mcp", "timeout": 10},
)

# 2. Create agent
agent = Agent(
    name="GymAgent",
    instructions=gym_mcp.templates.TEXT_ADVENTURE_INSTRUCTIONS,
    mcp_servers=[server],
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.1),
)

# 3. Run episodes (OpenAI Agents SDK handles execution natively)
# The OpenAI Agents SDK manages episode execution with built-in tool calling
async def run_openai_agent(prompt: str):
    result = await Runner.run(agent, prompt)
    return result.final_output

# Use AgentRing for result collection and analysis
results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.88, 13, True),
    # ... collect from OpenAI Agents execution
])
```

### Letta Integration

Letta agents can use AgentRing MCP tools as function definitions.

#### Basic Letta Agent

```python
import agentring.mcp as gym_mcp
from letta_client import Letta

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Convert to Letta tool format
letta_tools = []
for tool in tools:
    tool_def = gym_mcp.formats.to_letta_tool(tool)
    letta_tools.append(tool_def)

# 3. Create Letta client
client = Letta(api_key="your-letta-api-key")

# 4. Create Letta agent with tools
agent_state = client.agents.create(
    model="openai/gpt-4o-mini",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {
            "label": "persona",
            "value": gym_mcp.templates.TEXT_ADVENTURE_INSTRUCTIONS
        }
    ],
    # Letta handles tool registration differently
)

# 5. Run agent
response = client.agents.messages.create(
    agent_id=agent_state.id,
    input="Start a new text adventure episode and complete the quest"
)
```

### Custom SDK Integration

For any custom or unsupported SDK, you can use AgentRing's generic tools directly.

#### Generic Agent with Custom SDK

```python
import agentring.mcp as gym_mcp

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")

# 2. Use tools directly in your custom agent
class MyCustomAgent:
    def __init__(self, tools):
        self.tools = tools

    def run_episode(self, instructions: str):
        # Reset environment
        reset_result = self.tools["reset_env"](seed=42)
        print(f"Environment reset: {reset_result}")

        # Your custom agent logic here
        # Use self.tools["step_env"] for step_env, etc.

        return {"success": True, "steps": 5, "reward": 1.0}

# 3. Create and run agent
agent = MyCustomAgent(tools)
result = agent.run_episode("Complete the text adventure quest")
```

### Multi-SDK Agent System

You can even create agents that use multiple SDKs for different tasks.

#### Multi-SDK System

```python
import agentring.mcp as gym_mcp

# 1. Set up multiple MCP servers
multi_client = gym_mcp.MultiServerClient()
multi_client.add_server("textworld", "http://localhost:8070")
multi_client.add_server("alfworld", "http://localhost:8090")
multi_client.add_server("webshop", "http://localhost:8002")

# 2. Get tools from all servers
all_tools = multi_client.get_all_tools()

# 3. Create agents for different domains
textworld_tools = multi_client.get_tools("textworld")
alfworld_tools = multi_client.get_tools("alfworld")
webshop_tools = multi_client.get_tools("webshop")

# 4. Use different SDKs for different environments
def create_specialized_agent(sdk_name: str, tools: list, env_name: str):
    """Create an agent using the specified SDK for a specific environment."""

    if sdk_name == "crewai":
        from crewai import Agent
        from crewai.tools import tool

        # Wrap tools for CrewAI
        @tool
        def reset_env(seed=None):
            return tools["reset_env"](seed=seed)

        @tool
        def step_env(action: str):
            return tools["step_env"](action=action)

        return Agent(
            role=f"{env_name} Specialist",
            goal=f"Excel at tasks in {env_name}",
            tools=[reset_env, step_env],
            verbose=True
        )

    elif sdk_name == "langgraph":
        # LangGraph implementation
        pass

    # Add other SDKs...

    else:
        raise ValueError(f"Unsupported SDK: {sdk_name}")

# 5. Create specialized agents
textworld_agent = create_specialized_agent("crewai", textworld_tools, "TextWorld")
alfworld_agent = create_specialized_agent("crewai", alfworld_tools, "ALFWorld")

# 6. Run agents on their respective environments (SDK-native execution)
# textworld_result = textworld_agent.run("Complete TextWorld quest")
# alfworld_result = alfworld_agent.run("Complete ALFWorld task")

# 7. Collect and compare performance with AgentRing results
textworld_results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.85, 14, True),
    # ... collect results from actual runs
])

alfworld_results = gym_mcp.results.EpisodeResults([
    gym_mcp.types.EpisodeResult(1, 0.72, 18, True),
    # ... collect results from actual runs
])

print("TextWorld Results:")
print(textworld_results.summary())
print("\nALFWorld Results:")
print(alfworld_results.summary())
```

## API Reference

### Core AgentRing API

#### `agentring.make(id, mode="local", **kwargs)`
Create a Gymnasium environment.

**Parameters:**
- `id` (str): Environment ID (e.g., "CartPole-v1")
- `mode` (str): "local" or "remote"
- `render_mode` (str, optional): Render mode
- `gym_server_url` (str, optional): MCP server URL for remote mode
- `**kwargs`: Additional arguments

**Returns:** AgentRingClient instance

### MCP Extensions API

#### `agentring.mcp.create_tools(server_url, tool_names=None, client=None)`
Create callable tools from MCP server.

**Parameters:**
- `server_url` (str): MCP server URL
- `tool_names` (list, optional): Specific tool names to create
- `client` (MCPServerClient, optional): Pre-configured client

**Returns:** ToolCollection with tools accessible by name (`tools["reset_env"]`) or attribute (`tools.reset_env`) for IDE autocomplete


#### `agentring.mcp.MCPServerClient(server_url, **kwargs)`
MCP server client with connection management.

**Parameters:**
- `server_url` (str): Server URL
- `timeout` (float): Request timeout
- `max_retries` (int): Maximum retries
- `health_check_interval` (float): Health check interval

**Methods:**
- `health_check()`: Check server health
- `call_tool(tool_name, params)`: Call a tool
- `get_server_info()`: Get server information

#### `agentring.mcp.MultiServerClient()`
Manage multiple MCP servers.

**Methods:**
- `add_server(name, url)`: Add a server
- `get_tools(server_name)`: Get tools from server
- `get_all_tools()`: Get tools from all servers
- `health_check_all()`: Check all servers

#### `agentring.mcp.EpisodeResults(results)`
Episode results collection and analysis.

**Methods:**
- `summary()`: Get comprehensive statistics
- `to_json()`: Export to JSON
- `to_csv()`: Export to CSV
- `filter_by_success(successful_only)`: Filter results
- `filter_by_reward(min_reward, max_reward)`: Filter by reward
- `filter_by_steps(min_steps, max_steps)`: Filter by steps

### Utility Functions

#### Format Converters
- `agentring.mcp.formats.to_json_schema(tool)`: Convert to JSON Schema
- `agentring.mcp.formats.to_openapi_spec(tool)`: Convert to OpenAPI
- `agentring.mcp.formats.to_crewai_tool(tool)`: Convert for CrewAI
- `agentring.mcp.formats.to_langchain_tool(tool)`: Convert for LangChain

#### Tool Utilities
- `agentring.mcp.utils.compose_tools(*tool_lists)`: Combine tool lists
- `agentring.mcp.utils.filter_tools(tools, names, include_patterns, exclude_patterns)`: Filter tools
- `agentring.mcp.utils.validate_tool_call(tool, args)`: Validate tool arguments

#### Templates
- `agentring.mcp.templates.TEXT_ADVENTURE_INSTRUCTIONS`: TextWorld instructions
- `agentring.mcp.templates.SHOPPING_INSTRUCTIONS`: WebShop instructions
- `agentring.mcp.templates.HOUSEHOLD_INSTRUCTIONS`: ALFWorld instructions
- `agentring.mcp.templates.GENERIC_INSTRUCTIONS`: Generic gym instructions

## Troubleshooting

### Common Issues

#### Connection Errors
```python
# Check server health
client = gym_mcp.MCPServerClient("http://localhost:8070")
if not client.health_check():
    print("Server is not responding")
```

#### Tool Discovery Failures
```python
# Try with explicit client
client = gym_mcp.MCPServerClient("http://localhost:8070")
try:
    tools = gym_mcp.create_tools("http://localhost:8070", client=client)
except Exception as e:
    print(f"Tool discovery failed: {e}")
```

#### SDK Integration Issues
```python
# Validate tools before use
from agentring.mcp.utils import validate_tool_call

for tool in tools:
    is_valid, error = validate_tool_call(tool, {})
    if not is_valid:
        print(f"Tool {tool.__name__} has issues: {error}")
```

### Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed MCP communication
tools = gym_mcp.create_tools("http://localhost:8070")
```

### Performance Tips

1. **Reuse clients**: Create MCPServerClient once and reuse
2. **Batch operations**: Use `run_episodes()` instead of individual `run_episode()` calls
3. **Caching**: Server info and tools are cached automatically
4. **Health checks**: Use `health_check_all()` for multi-server setups

## Related Projects

- [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server) - MCP server for Gymnasium environments
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool integration protocol

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a GitHub discussion
- **Documentation**: See examples/ directory

---

**Status**: ✅ Production Ready | **Version**: 0.4.0 | **Python**: 3.10+ | **Tests**: 19 core + 7 MCP extension test suites
