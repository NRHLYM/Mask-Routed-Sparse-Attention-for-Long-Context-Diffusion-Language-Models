"""Pure helpers for SCBench-VT prompt construction and set scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


VAR_ASSIGNMENT_RE = re.compile(
    r"\bVAR\s+([A-Z]{2,})\s*=\s*(?:VAR\s+)?([A-Z0-9]+)\b"
)
VARIABLE_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")
TARGET_VALUE_RE = re.compile(r"value\s+([A-Z0-9]+)", re.IGNORECASE)
SCBENCH_STOP_WORDS = {
    "ACCORDING",
    "ANSWER",
    "ASSGINED",
    "ASSIGNED",
    "CHAIN",
    "FIND",
    "QUESTION",
    "TEXT",
    "THE",
    "THEY",
    "VALUE",
    "VAR",
    "VARIABLE",
    "VARIABLES",
}


@dataclass(frozen=True)
class Assignment:
    variable: str
    rhs: str
    start: int
    end: int
    text: str


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _flatten_answer_value(value: object) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_answer_value(item)
        return
    yield str(value)


def extract_variable_tokens(
    text: object,
    *,
    allowed: set[str] | None = None,
) -> list[str]:
    tokens: list[str] = []
    for match in VARIABLE_TOKEN_RE.finditer(str(text).upper()):
        token = match.group(0)
        if allowed is not None and token not in allowed:
            continue
        if allowed is None and token in SCBENCH_STOP_WORDS:
            continue
        tokens.append(token)
    return _dedupe_preserve_order(tokens)


def extract_answer_vars(example: Mapping[str, object]) -> list[str]:
    for key in ("answer", "answer_vars", "answers"):
        if key not in example:
            continue
        values: list[str] = []
        for value in _flatten_answer_value(example.get(key)):
            values.extend(extract_variable_tokens(value))
        if values:
            return _dedupe_preserve_order(values)
    return []


def extract_target_value(example: Mapping[str, object]) -> str | None:
    match = TARGET_VALUE_RE.search(str(example.get("input", "")))
    return None if match is None else match.group(1).upper()


def parse_assignments(context: str) -> list[Assignment]:
    assignments: list[Assignment] = []
    for match in VAR_ASSIGNMENT_RE.finditer(context.upper()):
        assignments.append(
            Assignment(
                variable=match.group(1),
                rhs=match.group(2),
                start=match.start(),
                end=match.end(),
                text=" ".join(match.group(0).split()),
            )
        )
    return assignments


def candidate_vars_from_context(context: str) -> list[str]:
    return _dedupe_preserve_order(
        assignment.variable for assignment in parse_assignments(context)
    )


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def build_assignment_window_context(
    context: str,
    assignments: Sequence[Assignment],
    *,
    window_chars: int,
) -> tuple[str, int]:
    if window_chars < 0:
        raise ValueError("window_chars must be non-negative")
    if not assignments:
        return "", 0
    if window_chars == 0:
        return "\n".join(assignment.text for assignment in assignments), len(assignments)

    intervals = _merge_intervals(
        [
            (
                max(0, assignment.start - window_chars),
                min(len(context), assignment.end + window_chars),
            )
            for assignment in assignments
        ]
    )
    return "\n...\n".join(context[start:end].strip() for start, end in intervals), len(
        intervals
    )


def evidence_vars_in_context(
    context: str,
    gold_vars: Sequence[str],
) -> list[str]:
    gold = set(gold_vars)
    retained = [
        assignment.variable
        for assignment in parse_assignments(context)
        if assignment.variable in gold
    ]
    retained_set = set(retained)
    return [variable for variable in gold_vars if variable in retained_set]


def build_oracle_context(
    example: Mapping[str, object],
    *,
    oracle_scope: str = "all_assignments",
) -> tuple[str, dict[str, object]]:
    raw_context = str(example.get("context", ""))
    assignments = parse_assignments(raw_context)
    gold_vars = extract_answer_vars(example)
    gold_set = set(gold_vars)
    gold_assignments = [
        assignment for assignment in assignments if assignment.variable in gold_set
    ]

    if oracle_scope == "all_assignments":
        selected = assignments
    elif oracle_scope == "gold_chain":
        selected = gold_assignments
    else:
        raise ValueError(f"unsupported oracle_scope={oracle_scope}")

    selected_vars = {assignment.variable for assignment in selected}
    missing_gold = [
        variable for variable in gold_vars if variable not in selected_vars
    ]
    metadata: dict[str, object] = {
        "oracle_scope": oracle_scope,
        "raw_assignment_count": len(assignments),
        "oracle_assignment_count": len(selected),
        "gold_assignment_count": len(gold_assignments),
        "gold_vars": gold_vars,
        "oracle_missing_gold_vars": missing_gold,
        "target_value": extract_target_value(example),
    }
    return "\n".join(assignment.text for assignment in selected), metadata


def build_scbench_prompt_parts(
    example: Mapping[str, object],
    *,
    mode: str,
    oracle_scope: str,
) -> tuple[str, str, str, dict[str, object]]:
    prefix = str(example.get("prefix", ""))
    if mode == "oracle_2k":
        context, metadata = build_oracle_context(
            example,
            oracle_scope=oracle_scope,
        )
    elif mode == "long_context":
        context = str(example.get("context", ""))
        metadata = {
            "oracle_scope": "none",
            "raw_assignment_count": len(parse_assignments(context)),
            "oracle_assignment_count": 0,
            "gold_assignment_count": len(
                evidence_vars_in_context(context, extract_answer_vars(example))
            ),
            "gold_vars": extract_answer_vars(example),
            "oracle_missing_gold_vars": [],
            "target_value": extract_target_value(example),
        }
    else:
        raise ValueError(f"unsupported mode={mode}")

    query = "\n\n" + str(example.get("input", "")).lstrip()
    return prefix, context, query, metadata


def tokenize_scbench_prompt(
    tokenizer,
    example: Mapping[str, object],
    *,
    mode: str,
    max_prompt_tokens: int,
    bos_token_id: int | None,
    oracle_scope: str = "all_assignments",
    long_truncation: str = "assignment_windows",
    assignment_window_chars: int = 256,
) -> tuple[list[int], dict[str, object]]:
    prefix, context, query, prompt_metadata = build_scbench_prompt_parts(
        example,
        mode=mode,
        oracle_scope=oracle_scope,
    )
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids
    prefix_ids = list(encode(prefix))
    context_ids = list(encode(context))
    query_ids = list(encode(query))
    bos = [] if bos_token_id is None else [int(bos_token_id)]

    fixed_tokens = len(bos) + len(prefix_ids) + len(query_ids)
    if max_prompt_tokens < fixed_tokens:
        raise ValueError(
            f"prompt budget {max_prompt_tokens} cannot fit {fixed_tokens} "
            "BOS/prefix/query tokens"
        )
    context_budget = max_prompt_tokens - fixed_tokens
    raw_prompt_tokens = fixed_tokens + len(context_ids)

    window_metadata: dict[str, object] = {}
    if len(context_ids) <= context_budget:
        kept_context_ids = context_ids
        truncation_mode = "none"
        context_dropped_tokens = 0
        kept_context_text = context
    else:
        if mode == "oracle_2k" or long_truncation == "error":
            raise ValueError(
                f"{mode} prompt needs {raw_prompt_tokens} tokens, budget is "
                f"{max_prompt_tokens}; use a larger physical_length or a "
                "smaller oracle scope"
            )
        if long_truncation == "assignment_windows":
            assignments = parse_assignments(context)
            if not assignments:
                raise ValueError("assignment_windows found no VAR assignments")
            window_chars = max(0, int(assignment_window_chars))
            trials: list[dict[str, int]] = []
            while True:
                kept_context_text, window_count = build_assignment_window_context(
                    context,
                    assignments,
                    window_chars=window_chars,
                )
                kept_context_ids = list(encode(kept_context_text))
                trials.append(
                    {
                        "window_chars": window_chars,
                        "tokens": len(kept_context_ids),
                        "windows": window_count,
                    }
                )
                if len(kept_context_ids) <= context_budget:
                    break
                if window_chars == 0:
                    raise ValueError(
                        "assignment-only context needs "
                        f"{len(kept_context_ids)} tokens, budget is "
                        f"{context_budget}"
                    )
                window_chars //= 2
            truncation_mode = "assignment_windows"
            window_metadata = {
                "assignment_window_chars": window_chars,
                "assignment_window_count": window_count,
                "assignment_window_trials": trials,
            }
        elif long_truncation == "head":
            kept_context_ids = context_ids[:context_budget]
            kept_context_text = tokenizer.decode(
                kept_context_ids,
                skip_special_tokens=False,
            )
            truncation_mode = "head"
        elif long_truncation == "tail":
            kept_context_ids = context_ids[-context_budget:]
            kept_context_text = tokenizer.decode(
                kept_context_ids,
                skip_special_tokens=False,
            )
            truncation_mode = "tail"
        else:
            raise ValueError(f"unsupported long_truncation={long_truncation}")
        context_dropped_tokens = len(context_ids) - len(kept_context_ids)

    gold_vars = extract_answer_vars(example)
    retained_vars = evidence_vars_in_context(kept_context_text, gold_vars)
    missing_vars = [
        variable for variable in gold_vars if variable not in set(retained_vars)
    ]
    prompt_ids = bos + prefix_ids + kept_context_ids + query_ids
    metadata: dict[str, object] = {
        **prompt_metadata,
        "mode": mode,
        "raw_context_tokens": len(context_ids),
        "raw_prompt_tokens": raw_prompt_tokens,
        "prompt_tokens_after_truncation": len(prompt_ids),
        "context_budget_tokens": context_budget,
        "context_kept_tokens": len(kept_context_ids),
        "context_dropped_tokens": context_dropped_tokens,
        "truncation_mode": truncation_mode,
        **window_metadata,
        "evidence_vars_retained": retained_vars,
        "evidence_missing_vars_after_prompt": missing_vars,
        "evidence_full": not missing_vars,
    }
    return prompt_ids, metadata


def set_scores(
    prediction_vars: Sequence[str],
    gold_vars: Sequence[str],
) -> dict[str, float | int]:
    prediction_set = set(prediction_vars)
    gold_set = set(gold_vars)
    true_positive = len(prediction_set & gold_set)
    false_positive = len(prediction_set - gold_set)
    false_negative = len(gold_set - prediction_set)
    precision = (
        true_positive / len(prediction_set) if prediction_set else 0.0
    )
    recall = true_positive / len(gold_set) if gold_set else 0.0
    if precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "accuracy": float(prediction_set == gold_set),
        "precision": precision,
        "recall": recall,
        "set_f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": len(prediction_set),
        "gold_count": len(gold_set),
    }


def score_prediction_text(
    prediction: str,
    gold_vars: Sequence[str],
    *,
    candidate_vars: Sequence[str] | None = None,
) -> dict[str, object]:
    allowed = None if candidate_vars is None else set(candidate_vars)
    prediction_vars = extract_variable_tokens(prediction, allowed=allowed)
    return {
        "prediction_vars": prediction_vars,
        **set_scores(prediction_vars, gold_vars),
    }
