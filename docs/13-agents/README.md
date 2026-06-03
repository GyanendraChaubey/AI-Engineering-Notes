# Agent-Based Systems

> Agents extend LLMs from question-answering to autonomous multi-step task execution.

---

## 1. What is an LLM agent and what strategies exist to implement one?

An **LLM agent** is a system where an LLM acts as a reasoning engine that dynamically decides which actions to take to accomplish a goal — rather than producing a single response.

**Core components:**
- **Brain (LLM):** Decides what to do next
- **Tools:** Functions the agent can call (search, calculator, code executor, APIs)
- **Memory:** Short-term (conversation history) and long-term (vector store)
- **Orchestrator:** Loop that runs LLM → action → observation → LLM

```
Goal: "Find the current stock price of Apple and compare to last year"
  ↓
LLM thinks: "I need to search for AAPL current price"
  ↓ [Tool: web_search("AAPL stock price")]
Observation: "$189.50"
  ↓
LLM thinks: "Now I need last year's price"
  ↓ [Tool: web_search("AAPL stock price June 2023")]
Observation: "$179.21"
  ↓
LLM thinks: "I have both, I can compute and answer"
  ↓
Final answer: "Apple stock is $189.50 today, up ~5.7% from $179.21 a year ago"
```

---

## 2. Why do we need agents and what are the common implementation strategies?

**Why agents:** Most real-world tasks require multiple steps, dynamic information, external tool use, or decisions that depend on intermediate results — all beyond what a single LLM call can do.

**Common strategies:**

| Strategy | Description | Best for |
|---|---|---|
| **ReAct** | Interleave Reasoning + Acting in natural language | General purpose agents |
| **Plan-and-Execute** | Plan all steps upfront, then execute each | Complex multi-step tasks |
| **OpenAI Functions / Tool Use** | Structured JSON tool calls | Production reliability |
| **Self-Critique / Reflexion** | Agent evaluates its own output and retries | Error-prone tasks |
| **Multi-agent** | Multiple specialized agents collaborate | Complex workflows |

---

## 3. Explain ReAct prompting with a code example and its advantages.

**ReAct (Reasoning + Acting)** interleaves thinking and tool use in a loop: the model writes a `Thought`, takes an `Action`, receives an `Observation`, and repeats.

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: a math expression string."""
    return str(eval(expression))

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # In production: call real search API
    return f"Search results for: {query}"

tools = [calculator, web_search]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct prompt structures the Thought/Action/Observation loop
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "What is 15% of Apple's 2023 revenue of $383.3 billion?"
})

# Model internally generates:
# Thought: I need to calculate 15% of 383.3 billion
# Action: calculator
# Action Input: 383.3 * 0.15
# Observation: 57.495
# Thought: I have the answer
# Final Answer: 15% of Apple's 2023 revenue is approximately $57.5 billion
```

**Advantages of ReAct:**
- Transparent reasoning — you can see every thought step
- Dynamic tool selection based on context
- Self-correcting — bad observation leads to revised reasoning
- No upfront planning needed — adapts to intermediate results

**Limitation:** Can get stuck in loops or make too many tool calls for simple tasks.

---

## 4. How does the Plan-and-Execute strategy work?

Plan-and-Execute separates planning from execution:

1. **Planner LLM:** Given the goal, generates a complete multi-step plan
2. **Executor:** Executes each step in order, often using a separate (cheaper) LLM

```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

llm = ChatOpenAI(model="gpt-4o")
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools)

agent = PlanAndExecute(planner=planner, executor=executor)

result = agent.run("Research the top 3 AI companies by valuation and create a comparison table")

# Internally:
# PLAN:
#   Step 1: Search for top AI companies by valuation
#   Step 2: Find valuation of company 1
#   Step 3: Find valuation of company 2
#   Step 4: Find valuation of company 3
#   Step 5: Create comparison table
#
# EXECUTE: Run each step sequentially
```

**Best for:** Tasks with a clear structure that can be planned upfront — research tasks, document creation, multi-step data processing.

**Weakness:** The plan is fixed — if step 2 returns unexpected results, the plan doesn't adapt (unlike ReAct which adapts dynamically).

---

## 5. How do OpenAI function calling / tool use work?

Instead of generating tool calls in natural language (like ReAct), OpenAI function calling outputs **structured JSON** — more reliable and parseable.

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tools with JSON schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

# First call: LLM decides to call a tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Extract tool call
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name      # "get_weather"
arguments = json.loads(tool_call.function.arguments)  # {"city": "Tokyo"}

# Execute the actual function
weather_result = get_weather(**arguments)

# Second call: feed result back to LLM
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    response.choices[0].message,  # Assistant's tool call message
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(weather_result)
    }
]

final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
print(final_response.choices[0].message.content)
```

**Key advantage:** Structured JSON output is predictable and doesn't require parsing natural language — far more reliable for production.

---

## 6. What are the key differences between OpenAI function calling and LangChain Agents?

| Aspect | OpenAI Function Calling | LangChain Agents |
|---|---|---|
| **Output format** | Structured JSON (reliable) | Natural language + structured (varies) |
| **Provider lock-in** | OpenAI / Anthropic specific | Provider-agnostic |
| **Flexibility** | Lower — must use their schema format | Higher — supports custom agents |
| **Reliability** | High — deterministic JSON parsing | Lower — natural language parsing can fail |
| **Abstraction** | Low-level — you manage the loop | High-level — loop managed by LangChain |
| **Multi-step loops** | Manual — you implement the agent loop | Automatic — `AgentExecutor` handles it |
| **Debugging** | More transparent | Can be opaque |

**When to use OpenAI function calling:** Production systems where reliability matters, simple tool-use scenarios.

**When to use LangChain Agents:** Rapid prototyping, complex multi-agent systems, provider flexibility needed.

**In production, prefer function calling** — the structured output eliminates parsing failures that plague ReAct-style agents.

```python
# Production pattern: explicit tool loop with function calling
def run_agent(user_message, max_iterations=10):
    messages = [{"role": "user", "content": user_message}]
    
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        # If no tool calls, we're done
        if not msg.tool_calls:
            return msg.content
        
        # Execute all tool calls
        messages.append(msg)
        for tc in msg.tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })
    
    return "Max iterations reached"
```

---

*Next: [Prompt Hacking →](../14-prompt-hacking/README.md)*
