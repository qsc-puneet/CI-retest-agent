"""Render the Director JSON summary into a Markdown report."""


def _fmt_duration(seconds):
    if seconds is None:
        return "?"
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _md_escape(value):
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ").strip()


def render_markdown_report(summary, exec_id=None):
    """Produce the human-readable Markdown report for the final Director JSON."""
    md = []
    rm = summary.get("run_metadata") or {}
    totals = summary.get("totals") or {}

    exec_id = exec_id or rm.get("exec_id") or "?"
    md.append(f"# CI Retest Analysis — ExecID {exec_id}")
    md.append("")

    md.append("## Ground Truth")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Test run name | {_md_escape(rm.get('test_run_name'))} |")
    builds = rm.get("builds_under_test") or []
    md.append(f"| Build(s) under test | {_md_escape(', '.join(builds) if builds else rm.get('build_under_test'))} |")
    md.append(f"| Start | {_md_escape(rm.get('start_time'))} |")
    md.append(f"| End | {_md_escape(rm.get('end_time'))} |")
    duration_str = rm.get("duration") or _fmt_duration(rm.get("duration_seconds"))
    md.append(f"| Duration | {_md_escape(duration_str)} |")
    total_exec = totals.get("total_test_cases")
    total_def = totals.get("total_test_cases_defined")
    incomplete = totals.get("total_incomplete") or 0
    exec_cell = _md_escape(total_exec)
    if total_def and total_exec and total_def != total_exec:
        exec_cell = f"{_md_escape(total_exec)} (of {_md_escape(total_def)} defined; {total_def - total_exec} not selected)"
    md.append(f"| Test cases executed | {exec_cell} |")
    md.append(f"| Passed | {_md_escape(totals.get('total_passed'))} |")
    md.append(f"| Failed | {_md_escape(totals.get('total_failed'))} |")
    if incomplete:
        md.append(f"| Incomplete | {_md_escape(incomplete)} |")
    pr = totals.get("pass_rate")
    pr_str = f"{pr}%" if isinstance(pr, (int, float)) else _md_escape(pr)
    md.append(f"| Pass rate | {pr_str} |")
    md.append("")

    md.append("## Verdict")
    md.append("")
    md.append(f"**Overall status:** `{summary.get('overall_status', '?')}`")
    md.append("")
    reasoning = summary.get("verdict_reasoning")
    if reasoning:
        md.append(reasoning)
        md.append("")
    exec_summary = summary.get("executive_summary")
    if exec_summary:
        md.append(f"> {exec_summary}")
        md.append("")

    failures = summary.get("failed_test_cases") or []
    if failures:
        md.append("## Failures")
        md.append("")
        for f in failures:
            name = f.get("test_case_name") or "?"
            md.append(f"### {name}")
            md.append("")
            md.append(f"- **Test plan:** {_md_escape(f.get('test_plan_name'))}")
            md.append(f"- **Hardware:** {_md_escape(f.get('hardware'))}")
            md.append(f"- **CaseExecutionID:** `{_md_escape(f.get('case_execution_id'))}`")
            md.append(f"- **Severity:** `{_md_escape(f.get('severity'))}`")
            md.append(f"- **Affected component:** {_md_escape(f.get('affected_component'))}")
            md.append("")

            actions = f.get("failed_actions") or []
            if actions:
                md.append("**Failed actions**")
                md.append("")
                md.append("| ActionExecutionID | Time | Tab | ActionName | Actual | Expected | Remarks |")
                md.append("|---|---|---|---|---|---|---|")
                for a in actions:
                    md.append(
                        "| "
                        + " | ".join(_md_escape(a.get(k)) for k in
                                     ("action_execution_id", "time", "tab_name",
                                      "action_name", "actual", "expected", "remarks"))
                        + " |"
                    )
                md.append("")

            interp = f.get("interpretation")
            if interp:
                md.append("**What it means**")
                md.append("")
                md.append(interp)
                md.append("")

            hypotheses = f.get("root_cause_hypotheses") or []
            if hypotheses:
                md.append("**Root-cause hypotheses**")
                md.append("")
                for h in hypotheses:
                    md.append(
                        f"- **{h.get('hypothesis', '?')}** — likelihood `{h.get('likelihood', '?')}`; "
                        f"evidence: {h.get('evidence', '?')}"
                    )
                md.append("")

            signals = f.get("corroborating_signals") or []
            if signals:
                md.append("**Corroborating signals**")
                md.append("")
                for s in signals:
                    md.append(f"- {s}")
                md.append("")

    actions = summary.get("recommended_actions") or []
    if actions:
        md.append("## Recommended actions")
        md.append("")
        for a in actions:
            md.append(
                f"- **[{a.get('priority', '?')}]** {a.get('action', '?')} — "
                f"*owner:* {a.get('owner', '?')}"
            )
        md.append("")

    guidance = summary.get("retest_guidance")
    if guidance:
        md.append("## Retest guidance")
        md.append("")
        md.append(guidance)
        md.append("")

    confidence = summary.get("confidence")
    if confidence is not None:
        md.append(f"_Confidence: {confidence}_")
        md.append("")

    return "\n".join(md)
