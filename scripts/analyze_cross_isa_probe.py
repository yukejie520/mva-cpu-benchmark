"""Apply the pre-registered S2 execution-graph/latency decision rules.

The script consumes the node census produced by ``probe_qdq_operator.py`` and
the round summary produced by ``measure_interleaved.py``.  It deliberately
does not infer a speed claim from one median: a direction is reported only if
all required rounds lie on the same side of R = FP32 median / INT8 median.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _variant(row: dict[str, str]) -> str:
    value = (row.get("variant") or "").lower()
    if value in {"fp32", "fp32_baseline"}:
        return "FP32"
    if "qoperator" in value or "qop" in value:
        return "QOperator"
    if "qdq" in value:
        return "QDQ"
    return value


def _number(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in {"", None} else 0


def _direction(row: dict[str, str]) -> str:
    direction = row.get("speed_direction")
    if direction:
        return direction
    if row.get("ratio_all_gt1", "").lower() == "true":
        return "faster"
    if row.get("ratio_all_lt1", "").lower() == "true":
        return "slower"
    return "not_separated"


def classify(census_rows: list[dict[str, str]], summary_rows: list[dict[str, str]],
             required_rounds: int = 3) -> dict:
    census = {_variant(row): row for row in census_rows}
    summary = {_variant(row): row for row in summary_rows}
    decisions = {}

    qdq = census.get("QDQ")
    qop = census.get("QOperator")
    qdq_summary = summary.get("QDQ", {})
    qop_summary = summary.get("QOperator", {})

    if qdq is None:
        decisions["QDQ"] = {"outcome": "load_or_census_failure",
                             "reason": "no QDQ execution-graph row"}
    else:
        qdq_int = _number(qdq, "conv_int")
        qdq_direction = _direction(qdq_summary)
        qdq_conflict = qdq_int == 0 and qdq_direction == "faster"
        decisions["QDQ"] = {
            "conv_int": qdq_int,
            "conv_fp32": _number(qdq, "conv_fp32"),
            "conversion": _number(qdq, "conversion"),
            "speed_direction": qdq_direction,
            "conflict": qdq_conflict,
            "outcome": "fallback_observed_conflict" if qdq_conflict else
                       "fallback_observed",
            "reason": ("QDQ has no integer convolution, but its latency direction "
                       "is faster; report the mechanism/latency conflict explicitly"
                       if qdq_conflict else
                       "QDQ has no integer convolution; this supports the fallback "
                       "mechanism observation, while latency remains auxiliary"),
        }

    if qop is None:
        decisions["QOperator"] = {
            "outcome": "runtime_rejected_or_census_failure",
            "reason": "no QOperator execution-graph row; retain the load error log "
                      "to distinguish rejection from an incomplete probe",
        }
    else:
        qop_int = _number(qop, "conv_int")
        qop_direction = _direction(qop_summary)
        if qop_int == 0:
            outcome = "fallback_observed"
            reason = "QOperator loaded but produced no integer convolution"
        elif qop_direction == "faster":
            outcome = "integer_execution_acceleration_supported"
            reason = "integer convolution is present and all required R values exceed 1"
        elif qop_direction == "slower":
            outcome = "integer_execution_necessary_not_sufficient"
            reason = ("integer convolution is present but all required R values are "
                      "below 1; graph structure, scheduling, or memory layout also "
                      "controls speed")
        else:
            outcome = "node_level_only"
            reason = ("integer convolution is present but the three-round latency "
                      "directions do not separate")
        decisions["QOperator"] = {
            "conv_int": qop_int,
            "conv_fp32": _number(qop, "conv_fp32"),
            "conversion": _number(qop, "conversion"),
            "speed_direction": qop_direction,
            "outcome": outcome,
            "reason": reason,
        }

    for row in summary_rows:
        rounds = row.get("rounds")
        if rounds and int(float(rounds)) != required_rounds:
            raise ValueError(
                f"{row.get('variant')} has {rounds} rounds; expected {required_rounds}"
            )
    return {"required_rounds": required_rounds, "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--required-rounds", type=int, default=3)
    parser.add_argument("--out", default=None, help="optional JSON decision record")
    args = parser.parse_args()
    result = classify(_read(args.census), _read(args.summary), args.required_rounds)
    for label, decision in result["decisions"].items():
        print(f"{label}: {decision['outcome']} -- {decision['reason']}")
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(f"[json] {output}")


if __name__ == "__main__":
    main()
