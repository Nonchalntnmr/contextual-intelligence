"""A deterministic stand-in for a tool-using model. NOT a model, and it says so.

Every reply starts with "[mock agent — no model connected]". It exists for two reasons:
tests that need exact tool calls (pass `script=`), and exercising the full workspace loop
(governor, approvals, execution, cross-session briefing) on a machine with no API key.

Without a script it routes a few literal commands, and nothing else:
    "run tests" · "git status" · "list files" · "read <path>" · "search <regex>"
    "write <path>: <content>" · "note decision|question|next step: <text>"
Anything else gets an echo of the project briefing it was given, which is how the mock
shows cross-session continuity. It does not pretend to reason about it.
"""

from __future__ import annotations

import json
import re

from .base import AgentStep, Usage

TAG = "[mock agent — no model connected]"


def _last_request(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    text = content if isinstance(content, str) else " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return text.rsplit("Request:\n", 1)[-1].strip()


def _briefing(messages: list[dict]) -> str:
    first = messages[0]["content"]
    text = first if isinstance(first, str) else ""
    m = re.search(r"<project_briefing[^>]*>\n(.*?)\n</project_briefing", text, re.S)
    return m.group(1) if m else ""


class MockAgentAdapter:
    provider = "mock"

    def __init__(self, model: str = "mock-agent", script: list[dict] | None = None):
        self.model = model
        self.script = list(script) if script else None
        self.requests: list[dict] = []
        self._n = 0

    def _step(self, text: str = "", tool_uses: list[dict] | None = None) -> AgentStep:
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]} for tu in (tool_uses or [])]
        return AgentStep(content=content, text=text, tool_uses=tool_uses or [],
                         stop_reason="tool_use" if tool_uses else "end_turn", usage=Usage(), cost_usd=0.0,
                         model=self.model)

    def _tool(self, name: str, inp: dict) -> AgentStep:
        self._n += 1
        return self._step(f"{TAG} Proposing {name}.", [{"id": f"toolu_mock_{self._n}", "name": name, "input": inp}])

    def agent_step(self, system: str, messages: list[dict], tools: list[dict], max_tokens=None) -> AgentStep:
        self.requests.append({"system": system, "messages": json.loads(json.dumps(messages, default=str))})
        if self.script is not None:
            spec = self.script.pop(0)
            return self._step(spec.get("text", ""), spec.get("tool_uses"))
        last = messages[-1]["content"]
        if isinstance(last, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in last):
            lines = []
            for b in last:
                if b.get("type") == "tool_result":
                    body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                    lines.append(f"- {'error' if b.get('is_error') else 'ok'}: {body[:400]}")
            return self._step(f"{TAG} Tool results:\n" + "\n".join(lines))
        req = _last_request(messages)
        low = req.lower()
        if re.search(r"\brun (the )?tests\b", low):
            return self._tool("run_tests", {"reason": "you asked me to run the tests"})
        if "git status" in low:
            return self._tool("git_status", {})
        if "list files" in low:
            return self._tool("list_files", {"pattern": "**/*"})
        if m := re.match(r"read (\S+)", req, re.I):
            return self._tool("read_file", {"path": m.group(1)})
        if m := re.match(r"search (.+)", req, re.I):
            return self._tool("search_code", {"pattern": m.group(1).strip(), "glob": "**/*"})
        if m := re.match(r"write (\S+?):\s*(.*)", req, re.I | re.S):
            return self._tool("write_file", {"path": m.group(1), "content": m.group(2), "reason": "you asked"})
        if m := re.match(r"note (decision|question|next step):\s*(.+)", req, re.I):
            kind = {"decision": "decision", "question": "open_question", "next step": "next_step"}[m.group(1).lower()]
            return self._tool("record_note", {"kind": kind, "text": m.group(2).strip()})
        b = _briefing(messages)
        return self._step(f"{TAG} I can't reason about this; connect a model provider for that. "
                          f"Here is the project context this session started from:\n{b or '(no briefing)'}")
