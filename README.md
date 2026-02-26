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

## Use cases

### Customer support automation

Build a multi-agent support system where a triage agent classifies incoming tickets, specialist agents handle specific domains, and a QA agent reviews responses before sending.

```python
from perm_agent import Agent, Pipeline, tool

@tool
def lookup_order(order_id: str) -> str:
    """Look up order details in the database."""
    return db.orders.get(order_id).to_json()

@tool
def lookup_account(email: str) -> str:
    """Look up customer account details."""
    return db.accounts.get_by_email(email).to_json()

@tool
def create_ticket(summary: str, priority: str, category: str) -> str:
    """Create a support ticket in the system."""
    return ticketing.create(summary=summary, priority=priority, category=category)

triage = Agent(
    name="triage",
    model="openai/gpt-4o",
    instructions="""Classify the customer request into one of: billing, shipping, technical, general.
    Extract key entities (order IDs, emails, product names).
    Output a structured summary for the specialist agent.""",
)

specialist = Agent(
    name="specialist",
    model="openai/gpt-4o",
    instructions="""You handle customer issues. Look up relevant data using tools.
    Be empathetic, concise, and solution-oriented.
    If you cannot resolve, escalate with a clear summary.""",
    tools=[lookup_order, lookup_account, create_ticket],
)

qa_reviewer = Agent(
    name="qa",
    model="anthropic/claude-sonnet-4-20250514",
    instructions="""Review the draft response for:
    - Accuracy of information
    - Tone and professionalism
    - Whether the issue is actually resolved
    Output the final approved response or request revision.""",
)

pipeline = Pipeline("customer-support", enable_communication=True)
pipeline.add_step(triage, output_path="/classification")
pipeline.add_step(specialist, input_map={"input": "@:/classification"}, output_path="/draft")
pipeline.add_step(qa_reviewer, input_map={"input": "@:/draft"}, output_path="/response")

result = pipeline.run({"input": "I ordered 3 days ago and haven't received shipping info. Order #12345"})
```

### Content generation pipeline

A research-write-edit pipeline for producing high-quality articles, reports, or documentation.

```python
from perm_agent import Agent, Pipeline, tool

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date information."""
    return search_api.query(query, limit=max_results)

@tool
def fetch_page(url: str) -> str:
    """Fetch and extract text from a web page."""
    return scraper.extract_text(url)

researcher = Agent(
    name="researcher",
    model="openai/gpt-4o",
    instructions="""Research the topic thoroughly using web search.
    Gather facts, statistics, and expert opinions.
    Cite sources. Output structured research notes.""",
    tools=[web_search, fetch_page],
)

writer = Agent(
    name="writer",
    model="anthropic/claude-sonnet-4-20250514",
    instructions="""Write a well-structured article based on the research.
    Use clear headings, engaging introduction, and actionable conclusion.
    Target audience: technical professionals.""",
)

editor = Agent(
    name="editor",
    model="openai/gpt-4o",
    instructions="""Edit the article for:
    - Grammar and clarity
    - Logical flow between sections
    - Factual consistency with the research
    - Remove filler words and redundancy
    Output the polished final version.""",
)

pipeline = Pipeline("content-pipeline")
pipeline.add_step(researcher, output_path="/research")
pipeline.add_step(writer, input_map={"input": "@:/research"}, output_path="/draft")
pipeline.add_step(editor, input_map={"input": "@:/draft"}, output_path="/article")

result = pipeline.run({"input": "Write an article about WebAssembly in 2026"})
print(result["article"])
```

### Data extraction and analysis

Extract structured data from unstructured sources, validate it, and produce analysis.

```python
from pydantic import BaseModel
from perm_agent import Agent, StructuredOutput, tool

class CompanyInfo(BaseModel):
    name: str
    industry: str
    revenue_usd: float | None
    employee_count: int | None
    headquarters: str
    key_products: list[str]

@tool
def search_company(name: str) -> str:
    """Search for company information."""
    return company_api.search(name)

agent = Agent(
    name="extractor",
    model="openai/gpt-4o",
    instructions="""Extract company information from the provided data.
    Return a JSON object matching this schema:
    {schema}
    If a field is unknown, use null.""".format(
        schema=StructuredOutput(CompanyInfo).json_schema()
    ),
    tools=[search_company],
)

raw_result = agent.run("Get info about SpaceX")
company = StructuredOutput(CompanyInfo).parse(raw_result)
print(f"{company.name}: {company.industry}, {company.employee_count} employees")
```

### Dynamic workflow engine (workflows-as-data)

Store AI workflows in a database and execute them on demand. This is the core differentiator of perm-agent -- workflows are JSON, not code.

```python
import json
from perm_agent import build_agent_engine

# Workflows stored in your database
WORKFLOWS_DB = {
    "summarize": [
        {
            "op": "agent_loop",
            "model": "openai/gpt-4o",
            "instructions": "Summarize the text in 3 bullet points.",
            "input": "${/text}",
            "tools": [],
            "path": "/summary",
        }
    ],
    "translate": [
        {
            "op": "llm",
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "user", "content": "Translate to ${/target_lang}: ${/text}"}
            ],
            "path": "/translation",
        }
    ],
    "research-and-summarize": [
        {
            "op": "agent_loop",
            "model": "openai/gpt-4o",
            "instructions": "Research the topic using available tools.",
            "input": "${/query}",
            "tools": ["web_search"],
            "path": "/research",
        },
        {
            "op": "llm",
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Write an executive summary:\n${@:/research}"}
            ],
            "path": "/summary",
        },
    ],
}

engine = build_agent_engine(tools={"web_search": web_search})

# Load and execute any workflow dynamically
def run_workflow(workflow_name: str, inputs: dict) -> dict:
    spec = WORKFLOWS_DB[workflow_name]
    return engine.apply(spec, source=inputs, dest={})

# API endpoint, CLI command, or queue worker can call this
result = run_workflow("research-and-summarize", {"query": "AI regulation in Europe"})
```

### Code review assistant

An agent that reviews pull requests, checking for bugs, security issues, and style violations.

```python
from perm_agent import Agent, Pipeline, tool

@tool
def get_diff(pr_number: int) -> str:
    """Get the diff of a pull request."""
    return github.pulls.get(pr_number).diff()

@tool
def get_file_content(path: str, ref: str = "main") -> str:
    """Get file content from the repository."""
    return github.repos.get_content(path, ref=ref)

@tool
def post_review_comment(pr_number: int, body: str, path: str, line: int) -> str:
    """Post an inline review comment on a PR."""
    return github.pulls.create_comment(pr_number, body=body, path=path, line=line)

security_reviewer = Agent(
    name="security",
    model="openai/gpt-4o",
    instructions="""Review the code diff for security vulnerabilities:
    - SQL injection, XSS, command injection
    - Hardcoded secrets or credentials
    - Insecure deserialization
    - Missing input validation
    Report each finding with file path and line number.""",
    tools=[get_diff, get_file_content],
)

logic_reviewer = Agent(
    name="logic",
    model="anthropic/claude-sonnet-4-20250514",
    instructions="""Review the code diff for logic errors:
    - Off-by-one errors, null pointer risks
    - Race conditions, missing error handling
    - Broken edge cases
    - Performance issues (N+1 queries, unnecessary loops)
    Report each finding with file path and line number.""",
    tools=[get_diff, get_file_content],
)

summarizer = Agent(
    name="summarizer",
    model="openai/gpt-4o",
    instructions="""Combine the security and logic reviews into a single
    structured review. Group by severity (critical, warning, info).
    Post inline comments for critical issues.""",
    tools=[post_review_comment],
)

pipeline = Pipeline("code-review", enable_communication=True)
pipeline.add_step(security_reviewer, output_path="/security_review")
pipeline.add_step(logic_reviewer, output_path="/logic_review")
pipeline.add_step(
    summarizer,
    input_map={"input": "Security: ${@:/security_review}\nLogic: ${@:/logic_review}"},
    output_path="/final_review",
)

result = pipeline.run({"input": "Review PR #42"})
```

### Chatbot with tool access and streaming

Build an interactive assistant with real-time token streaming and tool usage events.

```python
from perm_agent import build_agent_engine, TokenEvent, ToolCallEvent, ToolResultEvent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return weather_api.current(city)

@tool
def book_restaurant(name: str, date: str, party_size: int) -> str:
    """Book a table at a restaurant."""
    return booking_api.reserve(name=name, date=date, guests=party_size)

engine = build_agent_engine(tools={
    "get_weather": get_weather,
    "book_restaurant": book_restaurant,
})

def handle_message(user_input: str) -> None:
    def on_event(event):
        if isinstance(event, TokenEvent):
            print(event.token, end="", flush=True)  # stream to UI
        elif isinstance(event, ToolCallEvent):
            print(f"\n  [using {event.tool_name}...]")
        elif isinstance(event, ToolResultEvent):
            print(f"  [done]")

    spec = [
        {
            "op": "streaming_agent_loop",
            "model": "openai/gpt-4o",
            "instructions": """You are a helpful travel assistant.
            You can check weather and book restaurants.""",
            "input": user_input,
            "tools": ["get_weather", "book_restaurant"],
            "on_event": on_event,
            "path": "/response",
        }
    ]

    engine.apply(spec, source={}, dest={})
    print()  # newline after streaming

handle_message("What's the weather in Paris? And book Le Cinq for 2 on Friday.")
```

### Multi-agent coordination with communication

Agents that actively communicate through a shared blackboard and direct messages during execution.

```python
from perm_agent import Agent, Pipeline, tool

@tool
def analyze_metrics(service: str) -> str:
    """Pull performance metrics for a service."""
    return monitoring.get_metrics(service, period="24h")

@tool
def check_logs(service: str, level: str = "error") -> str:
    """Search service logs for errors."""
    return logging.search(service=service, level=level, period="24h")

diagnostician = Agent(
    name="diagnostician",
    model="openai/gpt-4o",
    instructions="""Analyze the service health issue.
    - Pull metrics and logs using tools
    - Post your findings to the board with post_to_board("diagnosis", ...)
    - Send a message to the 'resolver' agent with your top hypothesis.""",
    tools=[analyze_metrics, check_logs],
)

resolver = Agent(
    name="resolver",
    model="openai/gpt-4o",
    instructions="""You receive a diagnosis from the diagnostician.
    - Read the board with read_board("diagnosis") for full context
    - Check your messages with check_messages()
    - Propose a remediation plan
    - Post the plan to the board with post_to_board("plan", ...)""",
    tools=[],
)

pipeline = Pipeline("incident-response", enable_communication=True)
pipeline.add_step(diagnostician, output_path="/diagnosis")
pipeline.add_step(resolver, input_map={"input": "@:/diagnosis"}, output_path="/resolution")

result = pipeline.run({"input": "API latency spike on payment-service"})

# Inspect what agents communicated
hub = pipeline.communication
print("Board:", hub.blackboard.read_all())
print("Messages:", hub.mailbox.all_messages)
```

### Observability and cost monitoring

Track every LLM call, tool execution, and token usage across your pipelines.

```python
from perm_agent import Agent, Pipeline, Tracer, ConsoleTracerHook, CostTracker

cost = CostTracker()
tracer = Tracer(hooks=[ConsoleTracerHook(), cost])

# Tracer integrates with any pipeline via the engine
from perm_agent import build_agent_engine

engine = build_agent_engine(
    tools={"search": search, "calculate": calculate},
    tracer=tracer,
)

spec = [
    {
        "op": "agent_loop",
        "model": "openai/gpt-4o",
        "instructions": "Research and calculate market projections.",
        "input": "Project AI market size for 2030",
        "tools": ["search", "calculate"],
        "path": "/result",
    }
]

result = engine.apply(spec, source={}, dest={})

# Detailed span breakdown
for span in tracer.spans:
    print(f"[{span.operation}] {span.name} — {span.status} ({span.duration_ms:.0f}ms)")

# Cost summary
print(f"\nTotal: {cost.total_input_tokens} in / {cost.total_output_tokens} out tokens")
print(f"Estimated cost: ${cost.total_cost:.4f}")

# Export for Datadog, Grafana, etc.
import json
print(json.dumps(tracer.to_dict(), indent=2))
```

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
git clone https://github.com/Shtomuch/perm-agent.git
cd perm-agent
uv sync
uv run pytest
uv run ruff check src/ tests/
```

---

## License

MIT
