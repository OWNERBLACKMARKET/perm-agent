# perm-agent

Declarative, JSON-native AI agent framework. Define workflows as data, not code.

```
pip install perm-agent
```

---

## Why perm-agent

Every agent framework asks you to write Python orchestration code. perm-agent takes a different approach: **workflows are JSON specifications** that can be versioned, stored, shared, and composed -- like Terraform for AI pipelines.

But you don't have to start with JSON. The Python API is simple and compiles down to JSON specs automatically.

**Key idea:** simple Python for simple cases, JSON specs for complex/dynamic workflows.

```python
from perm_agent import Agent

agent = Agent(
    name="researcher",
    model="openai/gpt-4o",
    instructions="You are a research assistant. Answer concisely.",
    tools=[search],
)

result = agent.run("What causes aurora borealis?")
```

Under the hood, this compiles to a JSON spec:

```json
[
  {
    "op": "agent_loop",
    "model": "openai/gpt-4o",
    "instructions": "You are a research assistant. Answer concisely.",
    "input": "What causes aurora borealis?",
    "tools": ["search"],
    "max_iterations": 10,
    "path": "/result"
  }
]
```

That JSON spec can be saved to a database, loaded later, modified programmatically, or sent over HTTP.

---

## Features

| Feature | Description |
|---------|-------------|
| **Python-first API** | `Agent`, `Pipeline`, `@agent` decorator |
| **JSON-native workflows** | Serializable, versionable, composable specs |
| **Model-agnostic** | Any LLM via litellm (OpenAI, Anthropic, Gemini, local models) |
| **Multi-agent** | Handoffs, pipelines, agent composition |
| **Async** | Full async support for all handlers |
| **Streaming** | Token-by-token streaming with event system |
| **Observability** | Built-in tracing with spans, events, cost tracking |
| **Structured output** | Pydantic model validation for LLM responses |
| **Guardrails** | Input/output content filtering and validation |
| **Retry** | Configurable retry with exponential backoff |
| **Type-safe tools** | Auto schema from type hints, Pydantic models, enums, docstrings |
| **Error recovery** | Tool errors sent back to LLM, not crashes |

---

## Installation

Requires Python 3.10+.

```bash
pip install perm-agent
```

Or with uv:

```bash
uv add perm-agent
```

### Configure your LLM provider

perm-agent uses [litellm](https://docs.litellm.ai/) under the hood, so it works with 100+ LLM providers. Set the API key for your provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="AIza..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
```

Model names follow litellm conventions: `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `gemini/gemini-2.0-flash`, etc.

---

## Usage guide

### 1. Define tools

Tools are regular Python functions. The `@tool` decorator auto-generates OpenAI-compatible function schemas from type hints and docstrings.

```python
from perm_agent import tool

@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query.
        max_results: Maximum results to return.
    """
    # Your implementation here
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))
```

Supported parameter types: `str`, `int`, `float`, `bool`, `list[T]`, `dict`, `Optional[T]`, `Literal[...]`, `Enum`, `BaseModel`, `@dataclass`.

### 2. Create an agent

```python
from perm_agent import Agent

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",
    instructions="You are a helpful assistant. Use tools when needed.",
    tools=[search, calculate],
)

result = agent.run("What is the mass of the sun in kilograms?")
print(result)
```

The agent will call tools as needed, feeding results back to the LLM until it produces a final answer.

### 3. Decorator style

For a more concise API, use the `@agent` decorator. The docstring becomes the agent's instructions.

```python
from perm_agent import agent

@agent(model="openai/gpt-4o", tools=[search])
def researcher(question: str) -> str:
    """You are a research assistant. Find accurate information."""

answer = researcher("When was the transistor invented?")
```

### 4. Multi-agent pipeline

Chain multiple agents together. Each agent's output becomes the next agent's input.

```python
from perm_agent import Agent, Pipeline

researcher = Agent(
    name="researcher",
    model="openai/gpt-4o",
    instructions="Research the topic thoroughly.",
    tools=[search],
)

writer = Agent(
    name="writer",
    model="openai/gpt-4o",
    instructions="Write a clear summary based on the research.",
)

pipeline = Pipeline("research-and-write")
pipeline.add_step(researcher, output_path="/research")
pipeline.add_step(writer, input_map={"input": "@:/research"}, output_path="/summary")

result = pipeline.run({"input": "History of quantum computing"})
print(result["summary"])
```

### 5. JSON workflows

The real power: define workflows as pure JSON. Store them in a database, load from an API, version with git.

```python
import json
from perm_agent import build_agent_engine

def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

engine = build_agent_engine(tools={"search": search})

workflow = [
    {
        "op": "agent_loop",
        "model": "openai/gpt-4o",
        "instructions": "Research the topic.",
        "input": "${/question}",
        "tools": ["search"],
        "path": "/research",
    },
    {
        "op": "llm",
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "user", "content": "Summarize: ${@:/research}"}
        ],
        "path": "/summary",
    },
    {
        "op": "if",
        "cond": "${@:/summary}",
        "then": [{"op": "set", "path": "/status", "value": "done"}],
        "else": [{"op": "set", "path": "/status", "value": "failed"}],
    },
]

# The workflow is just a list of dicts -- save it anywhere
json.dumps(workflow)

# Execute
result = engine.apply(workflow, source={"question": "What is CRISPR?"}, dest={})
print(result["summary"])
```

### 6. Export and reconstruct agents

Any agent can be exported as a JSON spec and reconstructed later.

```python
import json
from perm_agent import Agent

agent = Agent(
    name="analyst",
    model="openai/gpt-4o",
    instructions="Analyze the data.",
    tools=[search],
)

# Export to JSON
spec = agent.to_spec()
saved = json.dumps(spec)

# Later: reconstruct from JSON
loaded = json.loads(saved)
rebuilt = Agent.from_spec(
    {"name": "analyst", "model": "openai/gpt-4o", "instructions": "Analyze the data.", "tools": ["search"]},
    tools={"search": search},
)

result = rebuilt.run("Analyze trends in AI")
```

---

## Available operations

All operations can be used in JSON workflow specs:

| Operation | Description |
|-----------|-------------|
| `llm` | Single LLM call |
| `tool` | Execute a registered tool |
| `agent_loop` | Full agent loop with tool calling |
| `handoff` | Delegate to another agent spec |
| `set` | Set a value at a path |
| `foreach` | Iterate over a collection |
| `if` | Conditional branching |
| `while` | Conditional loop |
| `streaming_llm` | LLM call with token streaming |
| `streaming_agent_loop` | Agent loop with streaming events |

Plus j-perm built-ins: `copy`, `delete`, `update`, `distinct`, `exec`, `assert`, `$def`/`$func` (reusable functions), `$eval` (isolated sub-workflows), `$or`/`$and`/`$not` (logical operators).

### Template syntax

- `${/path}` -- reference source data
- `${@:/path}` -- reference destination (previously computed) data
- `${int:/age}` -- cast to int
- `/path/-` -- append to array

---

## Advanced workflows

perm-agent supports declarative patterns that no other agent framework offers: reusable workflow functions, conditional fallbacks, and iterative loops -- all as JSON specs.

### Reusable workflow functions (`$def` / `$func`)

Define a workflow once, call it multiple times with different inputs.

```python
engine = build_agent_engine()

workflow = [
    # Define a reusable summarization function
    {
        "$def": "summarize",
        "params": ["text"],
        "body": [
            {
                "op": "llm",
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "user", "content": "Summarize in 2 sentences: ${/text}"},
                ],
                "path": "/result",
            },
        ],
        "return": "/result",
    },
    # Call it for each document
    {"/summary_a": {"$func": "summarize", "args": ["First document content..."]}},
    {"/summary_b": {"$func": "summarize", "args": ["Second document content..."]}},
]

result = engine.apply(workflow, source={}, dest={})
# result["summary_a"] and result["summary_b"] contain independent summaries
```

### Fallback with `$or`

Try a primary path; if it returns empty/falsy, fall back to an alternative.

```python
workflow = [
    {
        "/answer": {
            "$or": [
                [{"op": "set", "path": "/answer", "value": "${/cached_answer}"}],
                [{"op": "llm", "model": "openai/gpt-4o",
                  "messages": [{"role": "user", "content": "Answer: ${/question}"}],
                  "path": "/answer"}],
            ]
        }
    },
]
```

### Iterative refinement with `while`

Loop until a condition is met.

```python
workflow = [
    {"op": "set", "path": "/approved", "value": False},
    {
        "op": "while",
        "path": "@:/approved",
        "equals": False,
        "do": [
            {"op": "llm", "model": "openai/gpt-4o",
             "messages": [{"role": "user", "content": "Generate content"}],
             "path": "/output"},
            {"op": "if", "path": "@:/output",
             "then": [{"op": "set", "path": "/approved", "value": True}]},
        ],
    },
]
```

### Isolated sub-workflows with `$eval`

Run a sub-workflow in an isolated context without polluting the main state.

```python
workflow = [
    {"op": "set", "path": "/data", "value": "raw input"},
    {
        "/transformed": {
            "$eval": [
                {"op": "set", "path": "/x", "value": "processed"},
                {"op": "set", "path": "/temp", "value": "discarded"},
            ],
            "$select": "/x",
        }
    },
]
# result: {"data": "raw input", "transformed": "processed"}
# /temp stayed inside $eval's isolated context
```

---

## Structured output

Validate LLM responses against Pydantic models.

```python
from pydantic import BaseModel
from perm_agent import StructuredOutput

class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str]

output = StructuredOutput(UserProfile)

# Parse LLM response
profile = output.parse('{"name": "Alice", "age": 30, "interests": ["AI", "music"]}')
assert profile.name == "Alice"

# Safe parsing (returns None on failure)
result = output.parse_safe("invalid json")
assert result is None

# Get JSON schema for LLM prompting
schema = output.json_schema()
```

---

## Observability

Built-in tracing inspired by OpenTelemetry. Every handler emits spans automatically.

```python
from perm_agent import build_agent_engine, Tracer, ConsoleTracerHook, CostTracker

cost = CostTracker()
tracer = Tracer(hooks=[ConsoleTracerHook(), cost])

engine = build_agent_engine(tools={"search": search}, tracer=tracer)

spec = [
    {
        "op": "agent_loop",
        "model": "openai/gpt-4o",
        "instructions": "Research the topic.",
        "input": "Quantum computing",
        "tools": ["search"],
        "path": "/result",
    }
]

result = engine.apply(spec, source={}, dest={})

# Inspect spans
for span in tracer.spans:
    print(f"{span.operation}: {span.name} [{span.status}]")

# Token usage
print(f"Input tokens: {cost.total_input_tokens}")
print(f"Output tokens: {cost.total_output_tokens}")

# Export as JSON for log aggregation
trace_data = tracer.to_dict()
```

Traced operations: `llm`, `tool`, `agent_loop`, `handoff`, `streaming_llm`, `streaming_agent_loop`.

---

## Streaming

Token-by-token streaming with an event system.

```python
from perm_agent import build_agent_engine, TokenEvent, ToolCallEvent, AgentCompleteEvent

engine = build_agent_engine(tools={"search": search})

events_log = []

spec = [
    {
        "op": "streaming_agent_loop",
        "model": "openai/gpt-4o",
        "instructions": "Help the user.",
        "input": "Find info about Mars",
        "tools": ["search"],
        "on_event": lambda e: events_log.append(e),
        "path": "/result",
    }
]

result = engine.apply(spec, source={}, dest={})

for event in events_log:
    if isinstance(event, TokenEvent):
        print(event.token, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[calling {event.tool_name}]")
```

---

## Guardrails

Content filtering for inputs and outputs.

```python
from perm_agent import MaxLengthGuardrail, ContentFilterGuardrail, GuardrailPipeline

pipeline = GuardrailPipeline([
    MaxLengthGuardrail(max_length=5000),
    ContentFilterGuardrail(blocked_patterns=["password", r"api[_-]?key"]),
])

result = pipeline.check("This is safe content")
assert result.passed

result = pipeline.check("My password is 12345")
assert not result.passed
print(result.reason)  # "Content matches blocked pattern: 'password'"
```

---

## Retry

Automatic retry with exponential backoff for transient failures.

```python
spec = [
    {
        "op": "llm",
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "retry": {"max_retries": 3, "backoff_factor": 1.0},
        "path": "/answer",
    }
]
```

---

## Async

Full async support for all handlers.

```python
from perm_agent import AsyncLlmHandler, AsyncAgentLoopHandler, AsyncToolHandler

handler = AsyncLlmHandler()
result = await handler.execute(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Advanced tool schemas

The `@tool` decorator supports complex Python types:

```python
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel
from perm_agent import tool

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskInput(BaseModel):
    title: str
    description: str
    priority: Priority

@tool
def create_task(
    input: TaskInput,
    tags: list[str],
    assignee: Optional[str] = None,
    status: Literal["open", "closed"] = "open",
) -> str:
    """Create a new task in the system.

    Args:
        input: The task details.
        tags: Labels for categorization.
        assignee: Person responsible for the task.
        status: Current task status.
    """
    return f"Created: {input.title}"
```

---

## Architecture

```
Python API (Agent, Pipeline, @agent)
         |
         v
    JSON Specs
         |
         v
   j-perm Engine
         |
    +---------+---------+---------+
    |         |         |         |
  llm      tool   agent_loop  handoff
    |         |         |         |
 litellm  registry  loop+tools  delegate
```

- **j-perm** -- JSON transformation engine providing `set`, `foreach`, `if`, `while`, `$def/$func`, template resolution
- **litellm** -- universal LLM API (100+ models)
- **pydantic** -- structured output validation, tool schema generation

---

## Publishing to PyPI

```bash
uv build
uv publish
```

## Development

```bash
git clone https://github.com/denys/perm-agent.git
cd perm-agent
uv sync
uv run pytest
uv run ruff check src/ tests/
```

---

## License

MIT
