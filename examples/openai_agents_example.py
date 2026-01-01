import asyncio
from agents import Agent, Runner, function_tool, SQLiteSession
from agents.exceptions import MaxTurnsExceeded
import agentring as gym
from openai import RateLimitError, BadRequestError


async def main():
    """Run the OpenAI Agents SDK example with AgentRing environment."""
    
    try:
        
        # Create the AgentRing environment
        env = gym.make("FrozenLake-v1", mode="remote", gym_server_url="http://localhost:8000")
    
        # Create SDK-agnostic tools from the environment
        reset_env, step_env, get_env_info = env.create_env_tools()
        
        # Wrap with OpenAI Agents SDK function_tool
        tools = [
            function_tool(reset_env),
            function_tool(step_env),
            function_tool(get_env_info),
        ]
        
        # Create the agent with environment tools
        agent = Agent(
            name="GymAgent",
            instructions=(
                "You are an expert agent navigating a FrozenLake Gymnasium environment.\n"
                "FrozenLake is a 4x4 grid-world where you start at S (top-left) and must reach G (bottom-right).\n\n"
                "RULES:\n"
                "- Actions: 0=left, 1=down, 2=right, 3=up\n"
                "- Reward: 0 per step, 1.0 when reaching goal\n"
                "- Episode ends when: goal reached (reward=1.0, terminated=True) OR hole (terminated=True, reward=0)\n\n"
                "STRATEGY:\n"
                "1. Call get_env_info() ONCE at the start - do not call it repeatedly\n"
                "2. Take MANY actions in EACH turn using step_env() - take 5-10 actions per turn, not just 1-2\n"
                "3. After each step_env() call, check the response for 'Done: True' - if True, STOP and report the result\n"
                "4. If terminated=True and reward=1.0: SUCCESS! Report success and stop.\n"
                "5. If terminated=True and reward=0: Fell in hole. Report failure. You can reset and try again if needed.\n"
                "6. IMPORTANT: Take multiple actions per turn. Don't waste turns with single actions.\n"
                "7. The goal is at position 15 (bottom-right). Move systematically: right (2) and down (1).\n\n"
                "EFFICIENCY CRITICAL: In each turn, call step_env() multiple times (5-10 times) before stopping. "
                "Only stop when Done=True or you've taken many actions. Don't stop after 1-2 actions!"
            ),
            model="gpt-4o-mini",  # Using gpt-4o-mini for higher rate limits and lower cost
            tools=tools,
        )
        
        # Create a session to maintain conversation history across runs
        # This allows the agent to learn from previous attempts
        # The session persists to "agent_history.db" so history is maintained between script runs
        # To start fresh, delete the database file or use: await session.clear_session()
        session = SQLiteSession("frozenlake_agent", "agent_history.db")
        
        # Limit session history to prevent token overflow
        # Keep only the last 20 items to balance learning with token efficiency
        # Note: We clear the session if it gets corrupted (tool call mismatches)
        items = await session.get_items()
        if len(items) > 20:
            print(f"Session has {len(items)} items, limiting to last 20 for efficiency...")
            # Instead of keeping partial history, clear it to avoid tool call mismatches
            # The agent can still learn from the current run
            await session.clear_session()
            items = []
            print("Session history cleared to avoid tool call mismatches")
        
        # Reset the environment before starting the episode
        reset_env()
        
        # Run the agent with a more directive prompt
        prompt = (
            "Navigate from start (S) to goal (G) in FrozenLake.\n"
            "1. Call get_env_info() ONCE\n"
            "2. Take MANY actions per turn (5-10 step_env() calls) - don't stop after 1-2 actions\n"
            "3. Keep going until Done=True (either success with reward=1.0 or failure with reward=0)\n"
            "4. Goal is bottom-right - move right (2) and down (1) systematically"
        )
        
        print(f"Prompt: {prompt}\n")
        print("=" * 60)
        print(f"Session ID: {session.session_id}")
        print(f"Session history: {len(items)} items")
        print("=" * 60)
        
        # Increase max_turns significantly to allow more exploration
        # Pass session to maintain conversation history
        # Add retry logic for rate limits, tool call errors, and max turns
        max_retries = 3
        retry_delay = 5  # seconds
        max_turns = 100  # Increased from 50 to allow more exploration
        
        result = None
        for attempt in range(max_retries):
            try:
                result = await Runner.run(agent, prompt, max_turns=max_turns, session=session)
                break
            except MaxTurnsExceeded as e:
                print(f"\nMax turns ({max_turns}) exceeded.")
                if attempt < max_retries - 1:
                    print(f"Clearing session and retrying with fresh start (attempt {attempt + 1}/{max_retries})...")
                    await session.clear_session()
                    items = []
                    # Reset environment for fresh attempt
                    reset_env()
                else:
                    print("\nMax turns exceeded after multiple retries.")
                    print("The agent may need more efficient instructions or a different strategy.")
                    print(f"Final session history: {len(await session.get_items())} items")
                    raise
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    print(f"\nRate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    print("\nRate limit exceeded after multiple retries.")
                    print("Consider:")
                    print("  1. Waiting a few minutes before trying again")
                    print("  2. Clearing session history: await session.clear_session()")
                    print("  3. Using a different OpenAI API key with higher limits")
                    raise
            except BadRequestError as e:
                # Handle tool call mismatch errors (corrupted session history)
                error_msg = str(e)
                if "No tool call found for function call output" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"\nSession history corrupted (tool call mismatch). Clearing session and retrying...")
                        await session.clear_session()
                        items = []
                        reset_env()
                        print("Session cleared. Retrying...")
                    else:
                        print("\nSession history corrupted and retries exhausted.")
                        print("The session has been cleared. Please run again.")
                        raise
                else:
                    # Other BadRequestError - re-raise
                    raise
        
        if result is None:
            print("\nFailed to complete after all retries.")
            return
        
        print("\n" + "=" * 60)
        print("Agent response:")
        print("=" * 60)
        print(result.final_output)
        
        # Print environment statistics
        print("\n" + "=" * 60)
        print("Environment Statistics:")
        print(env.run_manager.format_statistics(show_episode_details=False))
        
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())