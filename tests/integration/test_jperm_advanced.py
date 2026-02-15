"""Integration tests for j-perm advanced features with perm-agent handlers.

Verifies $def/$func, $or, while, $eval, and composite workflows
work correctly when combined with perm-agent LLM/agent_loop handlers.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from perm_agent import build_agent_engine


def _make_text_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response.choices = [choice]
    return response


def _make_tool_response(tool_calls: list[dict]) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    calls = []
    for tc in tool_calls:
        call = MagicMock()
        call.id = tc["id"]
        call.function.name = tc["name"]
        call.function.arguments = json.dumps(tc["args"])
        calls.append(call)
    choice.message.tool_calls = calls
    choice.message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"]),
                },
            }
            for tc in tool_calls
        ],
    }
    response.choices = [choice]
    return response


def search(query: str) -> str:
    """Search for information"""
    return f"Found: {query}"


# ---------------------------------------------------------------------------
# Test 1: Reusable workflow functions ($def/$func)
# ---------------------------------------------------------------------------


class TestDefFuncWithLlm:
    @patch("litellm.completion")
    def test_reusable_summarize_function(self, mock_completion):
        """$def/$func reuse an LLM-calling function for multiple inputs."""
        mock_completion.side_effect = [
            _make_text_response("Summary of first document"),
            _make_text_response("Summary of second document"),
        ]
        engine = build_agent_engine()

        spec = [
            {
                "$def": "summarize",
                "params": ["text"],
                "body": [
                    {
                        "op": "llm",
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "Summarize: ${/text}"},
                        ],
                        "path": "/result",
                    },
                ],
                "return": "/result",
            },
            {"/summary1": {"$func": "summarize", "args": ["First document"]}},
            {"/summary2": {"$func": "summarize", "args": ["Second document"]}},
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["summary1"] == "Summary of first document"
        assert result["summary2"] == "Summary of second document"
        assert mock_completion.call_count == 2


# ---------------------------------------------------------------------------
# Test 2: Conditional routing with $or (fallback pattern)
# ---------------------------------------------------------------------------


class TestOrFallback:
    def test_or_returns_fallback_when_first_is_falsy(self):
        """$or skips falsy branch (empty dict) and returns the fallback."""
        engine = build_agent_engine()

        # First branch: assert on missing path returns False (falsy)
        # Second branch: produces a truthy dict
        spec = [
            {
                "/output": {
                    "$or": [
                        [{"op": "assert", "path": "/missing", "return": True}],
                        [{"op": "set", "path": "/x", "value": "fallback value"}],
                    ]
                }
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["output"] == {"x": "fallback value"}

    def test_or_returns_first_truthy(self):
        """$or short-circuits on first truthy result."""
        engine = build_agent_engine()

        spec = [
            {
                "/output": {
                    "$or": [
                        [{"op": "set", "path": "/x", "value": "first"}],
                        [{"op": "set", "path": "/x", "value": "second"}],
                    ]
                }
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["output"] == {"x": "first"}

    @patch("litellm.completion")
    def test_or_llm_with_default_fallback(self, mock_completion):
        """$or falls back to default when LLM returns empty."""
        mock_completion.return_value = _make_text_response("")
        engine = build_agent_engine()

        spec = [
            {
                "/output": {
                    "$or": [
                        [
                            {
                                "op": "llm",
                                "model": "test-model",
                                "messages": [
                                    {"role": "user", "content": "Generate something"},
                                ],
                                "path": "/llm_result",
                            },
                            # assert returns False when value is empty string
                            {"op": "assert", "value": "${@:/llm_result}", "return": True},
                        ],
                        [{"op": "set", "path": "/value", "value": "default response"}],
                    ]
                }
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["output"] == {"value": "default response"}
        assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# Test 3: While loop for iterative refinement
# ---------------------------------------------------------------------------


class TestWhileLoop:
    @patch("litellm.completion")
    def test_while_loops_with_llm_and_terminates(self, mock_completion):
        """While loop calls LLM once then terminates via flag."""
        mock_completion.return_value = _make_text_response("refined output")
        engine = build_agent_engine()

        spec = [
            {"op": "set", "path": "/needs_refinement", "value": True},
            {
                "op": "while",
                "path": "@:/needs_refinement",
                "equals": True,
                "do": [
                    {
                        "op": "llm",
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "Refine the output"},
                        ],
                        "path": "/output",
                    },
                    {"op": "set", "path": "/needs_refinement", "value": False},
                ],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["output"] == "refined output"
        assert result["needs_refinement"] is False
        assert mock_completion.call_count == 1

    @patch("litellm.completion")
    def test_while_quality_check_loop(self, mock_completion):
        """While loop retries LLM until quality condition is met."""
        mock_completion.side_effect = [
            _make_text_response("draft"),
            _make_text_response("APPROVED: final version"),
        ]
        engine = build_agent_engine()

        spec = [
            {"op": "set", "path": "/approved", "value": False},
            {
                "op": "while",
                "path": "@:/approved",
                "equals": False,
                "do": [
                    {
                        "op": "llm",
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "Generate approved content"},
                        ],
                        "path": "/output",
                    },
                    {
                        "op": "if",
                        "cond": "${@:/output}",
                        "then": [{"op": "set", "path": "/approved", "value": True}],
                    },
                ],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        # First iteration sets /output to "draft", which is truthy -> approved = True
        assert result["approved"] is True
        assert mock_completion.call_count == 1

    def test_while_with_path_equals(self):
        """While loop terminates when path no longer equals expected."""
        engine = build_agent_engine()

        spec = [
            {"op": "set", "path": "/status", "value": "running"},
            {"op": "set", "path": "/iterations", "value": 0},
            {
                "op": "while",
                "path": "@:/status",
                "equals": "running",
                "do": [
                    {"op": "set", "path": "/iterations", "value": 1},
                    {"op": "set", "path": "/status", "value": "done"},
                ],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["status"] == "done"
        assert result["iterations"] == 1


# ---------------------------------------------------------------------------
# Test 4: $eval for isolated sub-workflows
# ---------------------------------------------------------------------------


class TestEvalIsolation:
    def test_eval_runs_in_isolated_context(self):
        """$eval runs actions with fresh dest and returns selected value."""
        engine = build_agent_engine()

        spec = [
            {"op": "set", "path": "/data", "value": "raw input"},
            {
                "/transformed": {
                    "$eval": [
                        {"op": "set", "path": "/x", "value": "processed"},
                    ],
                    "$select": "/x",
                }
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["transformed"] == "processed"
        assert result["data"] == "raw input"

    @patch("litellm.completion")
    def test_eval_with_llm_inside(self, mock_completion):
        """$eval can contain LLM calls and return selected result."""
        mock_completion.return_value = _make_text_response("evaluated result")
        engine = build_agent_engine()

        spec = [
            {
                "/answer": {
                    "$eval": [
                        {
                            "op": "llm",
                            "model": "test-model",
                            "messages": [
                                {"role": "user", "content": "Process this"},
                            ],
                            "path": "/llm_out",
                        },
                    ],
                    "$select": "/llm_out",
                }
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["answer"] == "evaluated result"


# ---------------------------------------------------------------------------
# Test 5: Composite — agent_loop + if routing
# ---------------------------------------------------------------------------


class TestCompositeAgentLoopWithIf:
    @patch("litellm.completion")
    def test_agent_loop_result_routes_via_if(self, mock_completion):
        """agent_loop result is used in if-condition for routing."""
        mock_completion.return_value = _make_text_response("research findings")
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "test-model",
                "instructions": "Research the topic",
                "input": "AI safety",
                "tools": ["search"],
                "path": "/research",
            },
            {
                "op": "if",
                "path": "@:/research",
                "then": [{"op": "set", "path": "/status", "value": "success"}],
                "else": [{"op": "set", "path": "/status", "value": "empty"}],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["research"] == "research findings"
        assert result["status"] == "success"

    @patch("litellm.completion")
    def test_agent_loop_eval_then_if(self, mock_completion):
        """agent_loop result piped through $eval for extraction, then if routing."""
        mock_completion.return_value = _make_text_response("detailed research output")
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "test-model",
                "instructions": "Research the topic",
                "input": "quantum computing",
                "tools": ["search"],
                "path": "/raw_research",
            },
            # $eval uses source context; copy raw_research to source via copy op
            {"op": "copy", "from": "@:/raw_research", "path": "/raw_copy"},
            {
                "/extracted": {
                    "$eval": [
                        {"op": "set", "path": "/data", "value": "processed"},
                        {"op": "set", "path": "/length", "value": 42},
                    ],
                    "$select": "/data",
                }
            },
            {
                "op": "if",
                "path": "@:/raw_research",
                "then": [{"op": "set", "path": "/status", "value": "extracted"}],
                "else": [{"op": "set", "path": "/status", "value": "empty"}],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["raw_research"] == "detailed research output"
        assert result["extracted"] == "processed"
        assert result["status"] == "extracted"

    @patch("litellm.completion")
    def test_agent_loop_with_tool_then_if(self, mock_completion):
        """agent_loop uses tool, then if routes based on result."""
        mock_completion.side_effect = [
            _make_tool_response([{"id": "c1", "name": "search", "args": {"query": "test"}}]),
            _make_text_response("Based on search: Found: test"),
        ]
        engine = build_agent_engine(tools={"search": search})

        spec = [
            {
                "op": "agent_loop",
                "model": "test-model",
                "instructions": "Search and analyze",
                "input": "Find test data",
                "tools": ["search"],
                "path": "/research",
            },
            {
                "op": "if",
                "path": "@:/research",
                "then": [{"op": "set", "path": "/status", "value": "success"}],
                "else": [{"op": "set", "path": "/status", "value": "empty"}],
            },
        ]

        result = engine.apply(spec, source={}, dest={})

        assert result["research"] == "Based on search: Found: test"
        assert result["status"] == "success"
        assert mock_completion.call_count == 2
