"""
Deterministic tools — pre-process raw DB data before sending to LLM agents.

These tools parse, group, and compute metrics on the raw action/verification
data so the LLM receives structured context rather than raw rows.
"""

import re
from collections import defaultdict


def parse_expected_value(expected_str):
    """
    Parse expected value string into structured dict.

    Examples:
        "StereoSineGenerator_1_frequency : 1000"
        -> {"param": "StereoSineGenerator_1_frequency", "value": 1000.0}

        "WAN-TX-1_channel_1_peak_level : -37 Upper Limit: -34 Lower Limit: -40"
        -> {"param": "WAN-TX-1_channel_1_peak_level", "target": -37.0,
            "upper_limit": -34.0, "lower_limit": -40.0}
    """
    if not expected_str:
        return {"raw": expected_str}

    result = {"raw": expected_str}

    # Try to extract param : value
    match = re.match(r"(.+?)\s*:\s*(.+)", expected_str)
    if not match:
        return result

    param = match.group(1).strip()
    rest = match.group(2).strip()
    result["param"] = param

    # Check for Upper Limit / Lower Limit
    upper_match = re.search(r"Upper Limit:\s*([-\d.]+)", rest)
    lower_match = re.search(r"Lower Limit:\s*([-\d.]+)", rest)

    if upper_match and lower_match:
        # This is a verification with bounds
        target_match = re.match(r"([-\d.]+)", rest)
        if target_match:
            result["target"] = float(target_match.group(1))
        result["upper_limit"] = float(upper_match.group(1))
        result["lower_limit"] = float(lower_match.group(1))
    else:
        # Simple value assignment
        try:
            result["value"] = float(rest)
        except ValueError:
            result["value"] = rest

    return result


def parse_actual_value(actual_str):
    """
    Parse actual value string.

    Example:
        "WAN-TX-1_channel_1_peak_level : -36.99650192"
        -> {"param": "WAN-TX-1_channel_1_peak_level", "value": -36.99650192}
    """
    if not actual_str:
        return {"raw": actual_str}

    match = re.match(r"(.+?)\s*:\s*(.+)", actual_str)
    if not match:
        return {"raw": actual_str}

    param = match.group(1).strip()
    val_str = match.group(2).strip()

    try:
        value = float(val_str)
    except ValueError:
        value = val_str

    return {"param": param, "value": value}


def group_actions_by_test_sequence(actions, tabs):
    """
    Group actions into logical test sequences.

    A sequence = consecutive Control Actions followed by their
    Control Verifications, grouped by tab.

    Returns list of sequences:
    [
        {
            "tab_name": "Tab 1",
            "tab_execution_id": 39578,
            "setup_actions": [...],
            "verifications": [...],
        },
        ...
    ]
    """
    # Build tab lookup
    tab_lookup = {t["TabExecutionID"]: t["TabName"] for t in tabs}

    # Group actions by tab
    by_tab = defaultdict(list)
    for action in actions:
        by_tab[action["TabExecutionID"]].append(action)

    sequences = []
    for tab_id, tab_actions in by_tab.items():
        # Sort by start time or ActionExecutionID
        tab_actions.sort(key=lambda a: a.get("ActionExecutionID", 0))

        current_setup = []
        current_verifications = []

        for action in tab_actions:
            action_name = action.get("ActionName", "")
            if "Verification" in action_name:
                current_verifications.append(action)
            else:
                # If we have verifications collected, flush the previous sequence
                if current_verifications:
                    sequences.append({
                        "tab_name": tab_lookup.get(tab_id, f"Tab_{tab_id}"),
                        "tab_execution_id": tab_id,
                        "setup_actions": current_setup,
                        "verifications": current_verifications,
                    })
                    current_setup = []
                    current_verifications = []
                current_setup.append(action)

        # Flush remaining
        if current_verifications or current_setup:
            sequences.append({
                "tab_name": tab_lookup.get(tab_id, f"Tab_{tab_id}"),
                "tab_execution_id": tab_id,
                "setup_actions": current_setup,
                "verifications": current_verifications,
            })

    return sequences


def extract_failures(actions, tabs=None, test_cases=None, test_plans=None):
    """
    Extract only failed actions/verifications with parsed values.

    Enriches each failure with the parent tab_name, test_case_name and
    test_plan_name (when the lookups are supplied) so downstream agents
    have the anchors they need to reason about the failure.
    """
    tab_lookup = {t.get("TabExecutionID"): t for t in (tabs or [])}
    case_lookup = {t.get("CaseExecutionID"): t for t in (test_cases or [])}
    plan_lookup = {t.get("PlanExecutionID"): t for t in (test_plans or [])}

    failures = []
    for action in actions:
        if action.get("Status") != "Fail":
            continue

        expected = parse_expected_value(action.get("ExpectedValues", ""))
        actual = parse_actual_value(action.get("ActualValues", ""))

        tab = tab_lookup.get(action.get("TabExecutionID")) or {}
        case = case_lookup.get(tab.get("CaseExecutionID")) or {}
        plan = plan_lookup.get(case.get("PlanExecutionID")) or {}

        failure = {
            "action_execution_id": action.get("ActionExecutionID"),
            "tab_execution_id": action.get("TabExecutionID"),
            "tab_name": tab.get("TabName"),
            "case_execution_id": case.get("CaseExecutionID"),
            "test_case_name": case.get("TestCaseName"),
            "test_plan_name": plan.get("TestPlanName"),
            "hardware": plan.get("Hardware"),
            "feature": plan.get("Feature"),
            "build": plan.get("Build") or case.get("Build"),
            "design": plan.get("DesignName"),
            "inventory": plan.get("Inventory"),
            "start_time": str(action.get("StartTime")) if action.get("StartTime") else None,
            "end_time": str(action.get("EndTime")) if action.get("EndTime") else None,
            "action_name": action.get("ActionName", ""),
            "status": "Fail",
            "raw_expected": action.get("ExpectedValues", ""),
            "raw_actual": action.get("ActualValues", ""),
            "expected": expected,
            "actual": actual,
            "remarks": action.get("Remarks", ""),
        }

        # Compute deviation for verification failures
        if "upper_limit" in expected and isinstance(actual.get("value"), (int, float)):
            actual_val = actual["value"]
            if actual_val > expected["upper_limit"]:
                failure["deviation"] = round(actual_val - expected["upper_limit"], 4)
                failure["deviation_type"] = "above_upper_limit"
            elif actual_val < expected["lower_limit"]:
                failure["deviation"] = round(expected["lower_limit"] - actual_val, 4)
                failure["deviation_type"] = "below_lower_limit"
        elif "value" in expected and isinstance(actual.get("value"), (int, float)):
            # Control action failure (expected vs actual mismatch)
            if isinstance(expected["value"], (int, float)):
                failure["deviation"] = round(actual["value"] - expected["value"], 4)
                failure["deviation_type"] = "value_mismatch"

        failures.append(failure)

    return failures


def correlate_failures(failures):
    """
    Find patterns across failures:
    - Same channel failing across multiple verifications
    - All channels failing (suggests upstream issue)
    - Same tab with multiple failures
    - Action failures causing downstream verification failures

    Returns correlation analysis dict.
    """
    if not failures:
        return {"patterns": [], "summary": "No failures to correlate."}

    patterns = []

    # Group by channel (extract channel from param name)
    channel_failures = defaultdict(list)
    tab_failures = defaultdict(list)
    no_signal_failures = []
    control_failures = []

    for f in failures:
        param = f.get("actual", {}).get("param", "")
        tab_id = f.get("tab_execution_id")
        tab_failures[tab_id].append(f)

        # Detect channel
        ch_match = re.search(r"channel_(\d+)", param)
        if ch_match:
            channel_failures[ch_match.group(1)].append(f)

        # Detect no-signal (-96 dB range)
        actual_val = f.get("actual", {}).get("value")
        if isinstance(actual_val, (int, float)) and actual_val < -90:
            no_signal_failures.append(f)

        # Detect control action failures
        action_name = f.get("action_name", "")
        if "Action" in action_name and "Verification" not in action_name:
            control_failures.append(f)

    # Pattern: No signal detected
    if no_signal_failures:
        patterns.append({
            "type": "no_signal",
            "description": f"{len(no_signal_failures)} verification(s) show no signal (~-96 dB)",
            "likely_cause": "Generator muted, disconnected, or routing failure",
            "affected_count": len(no_signal_failures),
        })

    # Pattern: Channel-specific
    if len(channel_failures) > 1:
        for ch, ch_fails in channel_failures.items():
            other_channels = {k: v for k, v in channel_failures.items() if k != ch}
            if len(ch_fails) > 0 and all(len(v) == 0 for v in other_channels.values()):
                patterns.append({
                    "type": "channel_specific",
                    "description": f"Only channel {ch} failing ({len(ch_fails)} failures)",
                    "likely_cause": f"Channel {ch} routing or gain issue",
                    "affected_count": len(ch_fails),
                })

    # Pattern: Control action failure causing cascade
    if control_failures:
        patterns.append({
            "type": "control_cascade",
            "description": f"{len(control_failures)} control action(s) failed — may cause downstream verification failures",
            "likely_cause": "Core control not responding or parameter not applied",
            "affected_count": len(control_failures),
        })

    # Pattern: All failures in same tab
    for tab_id, tab_fails in tab_failures.items():
        if len(tab_fails) >= 3:
            patterns.append({
                "type": "tab_concentrated",
                "description": f"Tab {tab_id} has {len(tab_fails)} failures concentrated",
                "likely_cause": "Systemic issue within this signal path/tab",
                "affected_count": len(tab_fails),
            })

    return {
        "patterns": patterns,
        "total_failures": len(failures),
        "no_signal_count": len(no_signal_failures),
        "control_failure_count": len(control_failures),
        "channels_affected": list(channel_failures.keys()),
        "tabs_affected": list(tab_failures.keys()),
    }


def build_analysis_context(execution_data):
    """
    Master tool: takes raw execution data from DB and produces the full
    structured context that gets sent to LLM agents.
    """
    actions = execution_data.get("actions", [])
    tabs = execution_data.get("tabs", [])
    test_cases = execution_data.get("test_cases", [])
    test_plans = execution_data.get("test_plans", [])
    plan_groups = execution_data.get("plan_groups", [])

    failures = extract_failures(actions, tabs=tabs, test_cases=test_cases, test_plans=test_plans)
    sequences = group_actions_by_test_sequence(actions, tabs)
    correlations = correlate_failures(failures)

    # Build test case summary
    case_summary = []
    for tc in test_cases:
        case_summary.append({
            "name": tc.get("TestCaseName"),
            "status": tc.get("Status"),
            "case_execution_id": tc.get("CaseExecutionID"),
            "plan_execution_id": tc.get("PlanExecutionID"),
            "build": tc.get("Build"),
        })

    # Build plan summary
    plan_summary = []
    for tp in test_plans:
        plan_summary.append({
            "name": tp.get("TestPlanName"),
            "status": tp.get("Status"),
            "total": tp.get("TotalTestCasesCount"),
            "passed": tp.get("TotalPassedTestCase"),
            "failed": tp.get("TotalFailedTestCase"),
            "design": tp.get("DesignName"),
            "hardware": tp.get("Hardware"),
            "feature": tp.get("Feature"),
            "build": tp.get("Build"),
            "inventory": tp.get("Inventory"),
        })

    plan_group_summary = []
    for pg in plan_groups:
        plan_group_summary.append({
            "name": pg.get("TestPlanGroupName"),
            "status": pg.get("Status"),
            "total_test_plans": pg.get("TotalTestPlansCount"),
            "passed": pg.get("TotalPassedTestPlans"),
            "failed": pg.get("TotalFailedTestPlans"),
            "start_time": str(pg.get("StartTime")) if pg.get("StartTime") else None,
            "end_time": str(pg.get("EndTime")) if pg.get("EndTime") else None,
        })

    total_cases_defined = sum((tp.get("TotalTestCasesCount") or 0) for tp in test_plans)
    total_passed = sum((tp.get("TotalPassedTestCase") or 0) for tp in test_plans)
    total_failed = sum((tp.get("TotalFailedTestCase") or 0) for tp in test_plans)
    total_incomplete = sum((tp.get("TotalIncompleteTestCase") or 0) for tp in test_plans)
    # Executed = the cases that actually ran (pass/fail/incomplete). The
    # DB's TotalTestCasesCount often includes non-selected cases from the
    # test-plan definition, so use the executed count for pass-rate math.
    total_executed = total_passed + total_failed + total_incomplete
    if total_executed == 0:
        total_executed = len(test_cases)
    pass_rate = round(total_passed / max(total_executed, 1) * 100, 1)

    starts = [pg.get("StartTime") for pg in plan_groups if pg.get("StartTime")]
    ends = [pg.get("EndTime") for pg in plan_groups if pg.get("EndTime")]
    run_start = min(starts) if starts else None
    run_end = max(ends) if ends else None
    duration_seconds = None
    if run_start and run_end:
        try:
            duration_seconds = int((run_end - run_start).total_seconds())
        except Exception:
            duration_seconds = None

    builds = sorted({(tp.get("Build") or "").strip() for tp in test_plans if tp.get("Build")})

    run_metadata = {
        "exec_id": execution_data.get("exec_id"),
        "test_run_name": (plan_groups[0].get("TestPlanGroupName") if plan_groups else None),
        "plan_group_names": [pg.get("TestPlanGroupName") for pg in plan_groups],
        "builds_under_test": builds,
        "start_time": str(run_start) if run_start else None,
        "end_time": str(run_end) if run_end else None,
        "duration_seconds": duration_seconds,
    }

    totals = {
        "total_test_cases": total_executed,
        "total_test_cases_defined": total_cases_defined,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_incomplete": total_incomplete,
        "pass_rate": pass_rate,
    }

    failed_case_names = sorted({
        tc.get("TestCaseName") for tc in test_cases
        if (tc.get("Status") or "").strip().lower() == "fail" and tc.get("TestCaseName")
    })

    return {
        "exec_id": execution_data.get("exec_id"),
        "run_metadata": run_metadata,
        "totals": totals,
        "plan_group_summary": plan_group_summary,
        "plan_summary": plan_summary,
        "case_summary": case_summary,
        "failed_case_names": failed_case_names,
        "test_sequences": sequences,
        "failures": failures,
        "correlations": correlations,
        "total_actions": len(actions),
        "total_failures": len(failures),
    }


# ── Failure Matrix (retest planner) ─────────────────────────────────────────

def _aggregate_case_status(statuses):
    """
    Roll up multiple occurrences of the same TestCaseName in one run
    into a single status label.

    Rules (in order):
      - any 'Incomplete' -> 'incomplete'
      - any 'Fail'       -> 'fail'
      - all 'Pass'       -> 'pass'
      - anything else    -> 'unknown'
    """
    if not statuses:
        return "unknown"
    norm = [(s or "").strip().lower() for s in statuses]
    if any(s == "incomplete" for s in norm):
        return "incomplete"
    if any(s == "fail" for s in norm):
        return "fail"
    if all(s == "pass" for s in norm):
        return "pass"
    return "unknown"


def build_failure_matrix(current_run, prior_runs):
    """
    Pivot per-testcase results across the current run + N prior runs.

    Args:
        current_run: dict with keys:
            - "run_meta": row from TestRunTable (ExecID, QsysBuildUnderTest,
              StartDateTime, ...)
            - "test_cases": list of TestCaseTable rows for this run
              (must have TestCaseName + Status)
        prior_runs: list of dicts with the same shape, ordered
            newest-first (i.e. prior_runs[0] is the most recent prior).

    Returns:
        {
            "runs": [                              # in chronological order:
                                                    # oldest prior ... newest prior, current
                {"exec_id", "build", "start", "label"},
                ...
            ],
            "matrix": {
                <TestCaseName>: {
                    "per_run": ["pass"|"fail"|"incomplete"|"missing", ...],
                                                    # aligned with "runs"
                    "current_status": "pass"|"fail"|"incomplete"|"missing",
                    "prior_statuses": [...],       # oldest -> newest prior
                    "classification": "new_regression"
                                    | "flaky"
                                    | "persistent"
                                    | "first_run"
                                    | "recovered"
                                    | "still_passing",
                }
            }
        }

    Classification rules (deterministic; agent may refine):
      current   priors                          -> label
      ---------------------------------------------------------
      fail      all pass                        -> new_regression
      fail      mixed pass/fail                 -> flaky
      fail      all fail                        -> persistent
      fail      no prior data                   -> first_run
      pass      any fail in priors              -> recovered
      pass      all pass                        -> still_passing
      pass      no prior data                   -> first_run
      incomplete/missing                        -> "incomplete" / "missing"
    """
    # Build ordered run list: oldest prior first, current last
    ordered = list(reversed(prior_runs)) + [current_run]

    runs_meta = []
    for idx, run in enumerate(ordered):
        meta = run.get("run_meta", {}) or {}
        is_current = idx == len(ordered) - 1
        runs_meta.append({
            "exec_id": meta.get("ExecID"),
            "build": meta.get("QsysBuildUnderTest"),
            "start": str(meta.get("StartDateTime")) if meta.get("StartDateTime") else None,
            "label": "current" if is_current else f"prior-{len(ordered) - 1 - idx}",
        })

    # Per-run: TestCaseName -> aggregated status
    per_run_status = []
    all_names = set()
    for run in ordered:
        by_name = {}
        for tc in run.get("test_cases", []) or []:
            name = tc.get("TestCaseName")
            if not name:
                continue
            by_name.setdefault(name, []).append(tc.get("Status"))
            all_names.add(name)
        per_run_status.append({n: _aggregate_case_status(s) for n, s in by_name.items()})

    matrix = {}
    for name in sorted(all_names):
        per_run = [rs.get(name, "missing") for rs in per_run_status]
        current_status = per_run[-1]
        prior_statuses = per_run[:-1]

        # Only priors that actually ran this testcase count for classification.
        prior_ran = [s for s in prior_statuses if s not in ("missing", "incomplete")]

        if current_status in ("missing", "incomplete"):
            classification = current_status
        elif current_status == "fail":
            if not prior_ran:
                classification = "first_run"
            elif all(s == "pass" for s in prior_ran):
                classification = "new_regression"
            elif all(s == "fail" for s in prior_ran):
                classification = "persistent"
            else:
                classification = "flaky"
        elif current_status == "pass":
            if not prior_ran:
                classification = "first_run"
            elif any(s == "fail" for s in prior_ran):
                classification = "recovered"
            else:
                classification = "still_passing"
        else:
            classification = "unknown"

        matrix[name] = {
            "per_run": per_run,
            "current_status": current_status,
            "prior_statuses": prior_statuses,
            "classification": classification,
        }

    return {"runs": runs_meta, "matrix": matrix}


# Tool registry for agent use
TOOL_REGISTRY = {
    "build_analysis_context": {
        "fn": build_analysis_context,
        "description": "Parses raw DB data into structured sequences, extracts failures, computes deviations and correlations.",
    },
    "extract_failures": {
        "fn": extract_failures,
        "description": "Extracts only failed actions with parsed expected/actual values and deviation.",
    },
    "correlate_failures": {
        "fn": correlate_failures,
        "description": "Finds patterns: channel-specific, no-signal, control cascades, tab concentration.",
    },
}
