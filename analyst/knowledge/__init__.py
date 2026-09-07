"""Loads the checker knowledge base and provides pattern matching helpers.

The knowledge base lives at `analyst/knowledge/checkers.yaml`. It is
loaded once at import time and consulted by `tools.build_analysis_context`
to attach a `_domain_context` block to every failure. That block is then
visible to every LLM agent in the pipeline.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import yaml


_YAML_PATH = Path(__file__).with_name("checkers.yaml")


def _load_knowledge():
    if not _YAML_PATH.exists():
        return {"families": [], "remark_patterns": []}
    with open(_YAML_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    families = data.get("families") or []
    remarks = data.get("remark_patterns") or []
    # Pre-compile regex patterns once
    for r in remarks:
        if r.get("is_regex"):
            try:
                r["_re"] = re.compile(r["pattern"], re.IGNORECASE)
            except re.error:
                r["_re"] = None
    return {"families": families, "remark_patterns": remarks}


_KB = _load_knowledge()


def match_family(test_case_name: str | None) -> dict | None:
    """Return the first family whose glob patterns match the test case name."""
    if not test_case_name:
        return None
    name = test_case_name.lower()
    for fam in _KB["families"]:
        for pattern in fam.get("match", []):
            if fnmatch.fnmatch(name, pattern.lower()):
                return {
                    "checker_family": fam.get("name"),
                    "verdict": fam.get("verdict"),
                    "retest_useful": fam.get("retest_useful", True),
                    "manual_action": fam.get("manual_action"),
                    "purpose": fam.get("purpose"),
                    "actual_false_means": fam.get("what_actual_false_means"),
                    "common_causes": fam.get("common_causes", []),
                    "affected_component_hint": fam.get("affected_component_hint"),
                    "typical_severity": fam.get("typical_severity"),
                }
    return None


def match_remark(remark: str | None) -> dict | None:
    """Return the first remark_pattern that matches the remark string."""
    if not remark:
        return None
    for r in _KB["remark_patterns"]:
        if r.get("is_regex"):
            rx = r.get("_re")
            if rx and rx.search(remark):
                return {
                    "matched_pattern": r["pattern"],
                    "meaning": r.get("meaning"),
                    "rules_out": r.get("rules_out"),
                }
        else:
            if r["pattern"].lower() in remark.lower():
                return {
                    "matched_pattern": r["pattern"],
                    "meaning": r.get("meaning"),
                    "rules_out": r.get("rules_out"),
                }
    return None


def knowledge_stats() -> dict:
    """Diagnostics for test_analyzer.py to print at startup."""
    return {
        "families": len(_KB["families"]),
        "remark_patterns": len(_KB["remark_patterns"]),
        "source": str(_YAML_PATH),
    }
