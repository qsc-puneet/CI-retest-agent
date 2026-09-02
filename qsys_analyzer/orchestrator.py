"""
Orchestrator — 3-phase Qsys test execution analysis pipeline.

Phase 1 — Independent analysis by Classifier, Pattern Analyst, Impact Assessor
Phase 2 — Critic reviews (2a) + agents revise (2b)
Phase 3 — Summary Director synthesizes final report
"""

from .agents import (
    PHASE1_AGENTS, CriticAgent, SummaryDirectorAgent,
)
from .db_client import fetch_full_execution, fetch_execution_summary
from .tools import build_analysis_context


class QsysAnalyzer:
    """Orchestrates the multi-agent Qsys failure analysis pipeline."""

    def __init__(self, client, model="qwen3:8b", enable_revise=False):
        self.client = client
        self.model = model
        # When False (default), skip Phase 2a/2b and go Phase 1 -> Director.
        self.enable_revise = enable_revise

    def _phase1_analyze(self, context):
        """Run all Phase 1 agents sequentially."""
        results = {}
        for agent_cls in PHASE1_AGENTS:
            agent = agent_cls(self.client, self.model)
            print(f"    [{agent.name}] Analyzing...")
            try:
                result = agent.analyze(context)
                results[agent.role] = result
                print(f"    [{agent.name}] Done — confidence: {result.get('confidence', '?')}")
            except Exception as exc:
                print(f"    [{agent.name}] FAILED: {exc}")
                results[agent.role] = {"error": str(exc)}
        return results

    def _phase2a_critique(self, context, phase1_results):
        """Critic reviews all Phase 1 analyses."""
        critic = CriticAgent(self.client, self.model)
        print(f"    [{critic.name}] Reviewing analyses...")
        try:
            result = critic.critique(context, phase1_results)
            print(f"    [{critic.name}] Done — confidence: {result.get('confidence', '?')}")
            return result
        except Exception as exc:
            print(f"    [{critic.name}] FAILED: {exc}")
            return {"error": str(exc)}

    def _phase2b_revise(self, context, phase1_results, critique):
        """Phase 1 agents revise after seeing critique."""
        revised = {}
        for agent_cls in PHASE1_AGENTS:
            agent = agent_cls(self.client, self.model)
            print(f"    [{agent.name}] Revising after critique...")

            revision_prompt = (
                "You previously analyzed this execution. Here is the critic's feedback:\n\n"
                f"```json\n{__import__('json').dumps(critique, indent=2, default=str)}\n```\n\n"
                "Your original analysis:\n"
                f"```json\n{__import__('json').dumps(phase1_results.get(agent.role, {}), indent=2, default=str)}\n```\n\n"
                "Revise your analysis if the critique identified valid corrections. "
                "Keep your assessment if you disagree with the critique and explain why.\n\n"
                f"Original execution data:\n```json\n{__import__('json').dumps(context, indent=2, default=str)}\n```\n\n"
                f"Output revised JSON:\n{__import__('json').dumps('See ANALYSIS_SCHEMA')}"
            )

            try:
                raw = agent._call_llm([
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": revision_prompt},
                ])
                result = agent._parse_json(raw)
                revised[agent.role] = result
                print(f"    [{agent.name}] Revised — confidence: {result.get('confidence', '?')}")
            except Exception as exc:
                print(f"    [{agent.name}] Revision failed: {exc}")
                # Fall back to original
                revised[agent.role] = phase1_results.get(agent.role, {})

        return revised

    def _phase3_synthesize(self, context, all_analyses):
        """Director produces final summary."""
        director = SummaryDirectorAgent(self.client, self.model)
        print(f"    [{director.name}] Synthesizing final report...")
        try:
            result = director.synthesize(context, all_analyses)
            print(f"    [{director.name}] Done")
            return result
        except Exception as exc:
            print(f"    [{director.name}] FAILED: {exc}")
            return {"error": str(exc)}

    def _enforce_ground_truth(self, summary, context):
        """
        Overwrite factual fields in the LLM summary from the deterministic
        `context`. The LLM is only trusted for narrative fields
        (interpretation, hypotheses, corroborating_signals,
        recommended_actions, retest_guidance, severity).
        """
        if not isinstance(summary, dict) or "error" in summary:
            return summary

        # Ground-truth run metadata and totals from context
        summary["run_metadata"] = context.get("run_metadata", {})
        summary["totals"] = context.get("totals", {})

        # Index failures for lookup by action_execution_id and case_execution_id
        failures = context.get("failures", []) or []
        f_by_action = {f.get("action_execution_id"): f for f in failures if f.get("action_execution_id") is not None}
        f_by_case = {}
        for f in failures:
            cid = f.get("case_execution_id")
            if cid is None:
                continue
            f_by_case.setdefault(cid, []).append(f)

        # Rebuild failed_test_cases keyed by case_execution_id, merging duplicates
        rebuilt = {}
        llm_cases = summary.get("failed_test_cases") or []
        for llm_case in llm_cases:
            # Find the true case_execution_id via the first failed action
            actions = llm_case.get("failed_actions") or []
            true_case_id = None
            for a in actions:
                aid = a.get("action_execution_id")
                if aid in f_by_action:
                    true_case_id = f_by_action[aid].get("case_execution_id")
                    break

            if true_case_id is None:
                # Nothing to map; keep as-is but skip factual override
                rebuilt.setdefault(id(llm_case), llm_case)
                continue

            anchor = (f_by_case.get(true_case_id) or [{}])[0]
            entry = rebuilt.get(true_case_id)
            if entry is None:
                entry = {
                    "test_case_name": anchor.get("test_case_name"),
                    "test_plan_name": anchor.get("test_plan_name"),
                    "hardware": anchor.get("hardware"),
                    "case_execution_id": true_case_id,
                    "failed_actions": [],
                    "interpretation": llm_case.get("interpretation", ""),
                    "root_cause_hypotheses": llm_case.get("root_cause_hypotheses", []),
                    "corroborating_signals": llm_case.get("corroborating_signals", []),
                    "severity": llm_case.get("severity", "medium"),
                    "affected_component": llm_case.get("affected_component"),
                }
                rebuilt[true_case_id] = entry
            else:
                # Merge narrative fields when the LLM emitted duplicate entries
                for narrative_key in ("interpretation",):
                    if not entry.get(narrative_key) and llm_case.get(narrative_key):
                        entry[narrative_key] = llm_case[narrative_key]
                for list_key in ("root_cause_hypotheses", "corroborating_signals"):
                    for item in llm_case.get(list_key) or []:
                        if item not in entry[list_key]:
                            entry[list_key].append(item)

            # Always fill failed_actions from ground truth (ALL failing actions
            # for this case), not just the subset the LLM enumerated
            entry["failed_actions"] = [
                {
                    "action_execution_id": f.get("action_execution_id"),
                    "tab_name": f.get("tab_name"),
                    "action_name": f.get("action_name"),
                    "time": f.get("start_time"),
                    "actual": f.get("raw_actual"),
                    "expected": f.get("raw_expected"),
                    "remarks": f.get("remarks"),
                }
                for f in f_by_case.get(true_case_id, [])
            ]

        # Also ensure EVERY failed case in context has an entry, even if the LLM missed it
        for cid, case_failures in f_by_case.items():
            if cid in rebuilt:
                continue
            anchor = case_failures[0]
            rebuilt[cid] = {
                "test_case_name": anchor.get("test_case_name"),
                "test_plan_name": anchor.get("test_plan_name"),
                "hardware": anchor.get("hardware"),
                "case_execution_id": cid,
                "failed_actions": [
                    {
                        "action_execution_id": f.get("action_execution_id"),
                        "tab_name": f.get("tab_name"),
                        "action_name": f.get("action_name"),
                        "time": f.get("start_time"),
                        "actual": f.get("raw_actual"),
                        "expected": f.get("raw_expected"),
                        "remarks": f.get("remarks"),
                    }
                    for f in case_failures
                ],
                "interpretation": "",
                "root_cause_hypotheses": [],
                "corroborating_signals": [],
                "severity": "medium",
                "affected_component": None,
            }

        summary["failed_test_cases"] = list(rebuilt.values())

        # Derive overall_status deterministically
        totals = summary.get("totals") or {}
        total_failed = totals.get("total_failed", 0) or 0
        total_passed = totals.get("total_passed", 0) or 0
        total_cases = totals.get("total_test_cases", 0) or 0
        severities = [(c.get("severity") or "").lower() for c in summary["failed_test_cases"]]
        if total_failed == 0:
            summary["overall_status"] = "pass"
        elif any(s == "critical" for s in severities):
            summary["overall_status"] = "fail"
        else:
            summary["overall_status"] = "unstable"

        # Deterministic verdict_reasoning derived from totals + failing cases
        failing_case_count = len(summary["failed_test_cases"])
        failing_names = [c.get("test_case_name") for c in summary["failed_test_cases"] if c.get("test_case_name")]
        if summary["overall_status"] == "pass":
            summary["verdict_reasoning"] = f"{total_passed}/{total_cases} test cases passed; no failures recorded."
        else:
            names_frag = f" ({', '.join(failing_names)})" if failing_names else ""
            summary["verdict_reasoning"] = (
                f"{total_passed}/{total_cases} test cases passed; "
                f"{failing_case_count} test case{'s' if failing_case_count != 1 else ''} failed"
                f"{names_frag}."
            )

        # Deterministic executive summary — replace the LLM's word-count guess
        summary["executive_summary"] = self._build_executive_summary(
            summary, context, total_passed, total_cases, failing_case_count
        )

        # Inject sibling checker signals into each failed test case
        case_summary = context.get("case_summary") or []
        for entry in summary["failed_test_cases"]:
            sibling_signals = self._find_sibling_signals(
                entry.get("test_case_name"), case_summary
            )
            if sibling_signals:
                # We have factual data; discard LLM guesses entirely
                entry["corroborating_signals"] = sibling_signals
            else:
                # No factual signals — keep only LLM signals that reference a
                # real test case name from case_summary
                known_names = {c.get("name") for c in case_summary if c.get("name")}
                existing = entry.get("corroborating_signals") or []
                entry["corroborating_signals"] = [
                    s for s in existing
                    if any(name and name in s for name in known_names)
                ]

        return summary

    @staticmethod
    def _build_executive_summary(summary, context, total_passed, total_cases, failing_case_count):
        """Compose a factually-grounded 1-2 sentence executive summary."""
        rm = context.get("run_metadata") or {}
        builds = rm.get("builds_under_test") or []
        build_str = ", ".join(builds) if builds else "the tested build"
        duration = rm.get("duration") or ""
        duration_frag = f" over {duration}" if duration else ""

        if summary.get("overall_status") == "pass":
            return f"All {total_passed} test cases passed on {build_str}{duration_frag}."

        total_failed_actions = sum(
            len(c.get("failed_actions") or []) for c in summary.get("failed_test_cases") or []
        )
        names = [
            c.get("test_case_name")
            for c in summary.get("failed_test_cases") or []
            if c.get("test_case_name")
        ]
        severities = {
            (c.get("severity") or "").lower()
            for c in summary.get("failed_test_cases") or []
        }
        components = [
            c.get("affected_component")
            for c in summary.get("failed_test_cases") or []
            if c.get("affected_component")
        ]
        sev_frag = f" ({'/'.join(sorted(s for s in severities if s))})" if severities else ""
        comp_frag = f" on {', '.join(sorted(set(components)))}" if components else ""
        names_frag = f": {', '.join(names)}" if names else ""

        return (
            f"{total_passed}/{total_cases} test cases passed on {build_str}{duration_frag}. "
            f"{failing_case_count} test case{'s' if failing_case_count != 1 else ''} failed{sev_frag}"
            f"{names_frag} with {total_failed_actions} failed action"
            f"{'s' if total_failed_actions != 1 else ''}{comp_frag}."
        )

    @staticmethod
    def _find_sibling_signals(failing_name, case_summary):
        """
        Find case_summary entries that share a distinctive suffix with the
        failing test case (e.g. `_Core-Crash_Checker`) and report their
        pass/fail status as factual corroborating signals.
        """
        if not failing_name or not case_summary:
            return []

        tokens = [t for t in failing_name.split("_") if t]
        if len(tokens) < 2:
            return []
        # Family suffix: last two underscore tokens (e.g. Core-Crash_Checker)
        family = "_".join(tokens[-2:])
        # Require the family to be reasonably distinctive
        if len(family) < 6:
            return []

        siblings = []
        for c in case_summary:
            name = c.get("name")
            if not name or name == failing_name:
                continue
            if name.endswith("_" + family) or name == family:
                siblings.append(c)

        if not siblings:
            return []

        passed = [s for s in siblings if (s.get("status") or "").lower() == "pass"]
        failed = [s for s in siblings if (s.get("status") or "").lower() == "fail"]

        signals = []
        if passed and not failed:
            names = ", ".join(f"`{s['name']}`" for s in passed)
            plural = "s" if len(passed) > 1 else ""
            signals.append(
                f"Sibling checker{plural} {names} passed in the same run — the "
                f"failure is device/context-specific, not systemic to the "
                f"`{family}` family."
            )
        elif failed and not passed:
            names = ", ".join(f"`{s['name']}`" for s in failed)
            plural = "s" if len(failed) > 1 else ""
            signals.append(
                f"Sibling checker{plural} {names} also failed in the same run — "
                f"the failure is systemic to the `{family}` family, not device-specific."
            )
        elif passed and failed:
            pass_names = ", ".join(f"`{s['name']}`" for s in passed)
            fail_names = ", ".join(f"`{s['name']}`" for s in failed)
            signals.append(
                f"Mixed sibling results in the `{family}` family: passed = {pass_names}; "
                f"failed = {fail_names}."
            )
        return signals

    def run(self, exec_id, failed_only=True):
        """
        Execute the full analysis pipeline for a given Execution ID.

        Args:
            exec_id: The ExecID to analyze
            failed_only: If True, only fetch failed test cases (faster)

        Returns:
            dict with final summary report
        """
        print("=" * 60)
        print("  QSYS TEST EXECUTION ANALYZER")
        print("=" * 60)

        # ── Fetch data from DB ──────────────────────────────────────────
        print(f"\n  Fetching execution data for ExecID: {exec_id}")
        execution_data = fetch_full_execution(exec_id, failed_only=failed_only)

        total_actions = len(execution_data.get("actions", []))
        total_cases = len(execution_data.get("test_cases", []))
        print(f"  Found {total_cases} test case(s), {total_actions} action(s)")

        if total_actions == 0:
            print("  No actions found. Nothing to analyze.")
            return {"overall_status": "pass", "executive_summary": "No failures found."}

        # ── Pre-process with deterministic tools ────────────────────────
        print("\n  Pre-processing data...")
        context = build_analysis_context(execution_data)
        print(f"  Extracted {context['total_failures']} failure(s) from {context['total_actions']} action(s)")

        if context["total_failures"] == 0:
            print("  All actions passed. No analysis needed.")
            return {"overall_status": "pass", "executive_summary": "All verifications passed."}

        # ── Phase 1: Independent Analysis ───────────────────────────────
        print(f"\n{'─' * 60}")
        print("  Phase 1 — Independent Agent Analysis")
        print("─" * 60)
        phase1_results = self._phase1_analyze(context)

        if self.enable_revise:
            # ── Phase 2a: Critic ──────────────────────────────────────
            print(f"\n{'─' * 60}")
            print("  Phase 2a — Critic Review")
            print("─" * 60)
            critique = self._phase2a_critique(context, phase1_results)

            # ── Phase 2b: Revision ───────────────────────────────────
            print(f"\n{'─' * 60}")
            print("  Phase 2b — Agent Revision")
            print("─" * 60)
            revised_results = self._phase2b_revise(context, phase1_results, critique)

            all_analyses = {
                "phase1": phase1_results,
                "critique": critique,
                "revised": revised_results,
            }
        else:
            print("\n  (PIPELINE_MODE=fast — skipping critic + revise pass)")
            all_analyses = {"phase1": phase1_results}

        # ── Phase 3: Final Summary ──────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("  Phase 3 — Summary Director")
        print("─" * 60)
        summary = self._phase3_synthesize(context, all_analyses)

        # Overwrite the LLM's factual fields with deterministic ground truth
        summary = self._enforce_ground_truth(summary, context)

        print(f"\n{'=' * 60}")
        print("  ANALYSIS COMPLETE")
        print("=" * 60)

        return summary
