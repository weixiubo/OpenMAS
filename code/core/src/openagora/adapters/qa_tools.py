"""QA task and tool adapters for the MVP suite."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Literal

from openagora.core.interfaces import ToolAdapter
from openagora.types import ArchitectureSpec, CoordinationMechanism, TaskSignature, TaskSpec

_SAFE_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            return float(_SAFE_BIN_OPS[type(node.op)](node.left.value, node.right.value))
    raise ValueError(f"Unsupported expression: {expr}")


class LocalToolAdapter:
    """Simple deterministic local tool adapter."""

    def execute(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "calculator":
            expr = str(payload.get("expression", "0"))
            try:
                value = _safe_eval(expr)
            except ValueError:
                value = 0.0
            return {"tool": "calculator", "value": value}
        if tool_name == "search":
            return {"tool": "search", "snippet": str(payload.get("query", ""))}
        if tool_name == "synthesize":
            parts = payload.get("parts", [])
            joined = " ".join(str(part) for part in parts)
            return {"tool": "synthesize", "answer": joined.strip()}
        raise ValueError(f"Unsupported tool: {tool_name}")


class QAToolTaskAdapter:
    """Task adapter for multi-step QA + tools."""

    def analyze_task(self, task: TaskSpec) -> TaskSignature:
        if task.signature is not None:
            return task.signature
        meta = task.metadata
        uncertainty = str(meta.get("uncertainty", "medium"))
        complexity = str(meta.get("complexity", "medium"))
        interdependence = str(meta.get("interdependence", "reciprocal"))
        return TaskSignature(
            primary_type=str(meta.get("primary_type", "multi_step_qa")),
            uncertainty=_coerce_uncertainty(uncertainty),
            complexity=_coerce_complexity(complexity),
            interdependence=_coerce_interdependence(interdependence),
        )

    def decompose_task(self, task: TaskSpec, architecture: ArchitectureSpec) -> list[dict[str, Any]]:
        steps = [{"name": "inspect", "tool": "search"}]
        if architecture.coordination.mechanism == CoordinationMechanism.PROCESS_STANDARDIZATION:
            # Sequential workflows still need an explicit integration step; otherwise
            # the calculator result can be dropped before final evaluation.
            steps.extend([{"name": "solve", "tool": "calculator"}, {"name": "verify", "tool": "synthesize"}])
        elif architecture.coordination.mechanism == CoordinationMechanism.MUTUAL_ADJUSTMENT:
            steps.extend([{"name": "explore", "tool": "search"}, {"name": "synthesize", "tool": "synthesize"}])
        else:
            steps.extend([{"name": "solve", "tool": "calculator"}, {"name": "synthesize", "tool": "synthesize"}])
        return steps

    def execute_subtask(
        self,
        *,
        task: TaskSpec,
        subtask: dict[str, Any],
        architecture: ArchitectureSpec,
        tools: ToolAdapter,
        seed: int,
    ) -> dict[str, Any]:
        prompt = task.prompt
        ref_answer = str(task.metadata.get("reference_answer", ""))
        expr_match = re.search(r"(-?\d+\s*[+*/-]\s*-?\d+)", prompt)
        expression = str(task.metadata.get("expression", expr_match.group(1) if expr_match else "0+0"))
        tool = str(subtask.get("tool", "search"))
        allowed_tools = task.metadata.get("allowed_tools")
        if isinstance(allowed_tools, list) and tool not in [str(item) for item in allowed_tools]:
            raise ValueError(f"Tool `{tool}` is not allowed for task {task.task_id}")
        if tool == "calculator":
            calc = tools.execute("calculator", {"expression": expression})
            result = str(int(calc["value"])) if float(calc["value"]).is_integer() else str(calc["value"])
        elif tool == "synthesize":
            result = tools.execute("synthesize", {"parts": [ref_answer]}).get("answer", "")
        else:
            result = tools.execute("search", {"query": prompt}).get("snippet", "")
        return {
            "step": str(subtask.get("name", "unknown")),
            "tool": tool,
            "output": result,
            "seed": seed,
            "coordination": architecture.coordination.mechanism.value,
        }

    def evaluate_result(
        self,
        *,
        task: TaskSpec,
        step_results: list[dict[str, Any]],
        architecture: ArchitectureSpec,
    ) -> dict[str, Any]:
        final_text = " ".join(str(item.get("output", "")) for item in step_results[-2:])
        reference = str(task.metadata.get("reference_answer", "")).strip()
        quality = 0.55
        if reference and reference in final_text:
            quality = 0.82
        mechanism = architecture.coordination.mechanism
        signature = self.analyze_task(task)
        if signature.interdependence == "reciprocal" and mechanism == CoordinationMechanism.MUTUAL_ADJUSTMENT:
            quality += 0.1
        if signature.interdependence == "sequential" and mechanism == CoordinationMechanism.PROCESS_STANDARDIZATION:
            quality += 0.08
        if signature.complexity == "high" and mechanism == CoordinationMechanism.SKILL_STANDARDIZATION:
            quality += 0.07
        quality = max(0.0, min(1.0, quality))
        return {
            "answer": final_text.strip(),
            "quality_score": quality,
            "reference_answer": reference,
            "is_correct": bool(reference and reference in final_text),
        }


def _coerce_uncertainty(value: str) -> Literal["low", "medium", "high"]:
    if value == "low":
        return "low"
    if value == "high":
        return "high"
    return "medium"


def _coerce_complexity(value: str) -> Literal["low", "medium", "high"]:
    if value == "low":
        return "low"
    if value == "high":
        return "high"
    return "medium"


def _coerce_interdependence(value: str) -> Literal["pooled", "sequential", "reciprocal"]:
    if value == "pooled":
        return "pooled"
    if value == "sequential":
        return "sequential"
    return "reciprocal"
