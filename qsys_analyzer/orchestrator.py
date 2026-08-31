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

    def __init__(self, client, model="qwen3:8b"):
        self.client = client
        self.model = model

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
                f"```json\n{__import__('json').dumps(critique, indent=2)}\n```\n\n"
                "Your original analysis:\n"
                f"```json\n{__import__('json').dumps(phase1_results.get(agent.role, {}), indent=2)}\n```\n\n"
                "Revise your analysis if the critique identified valid corrections. "
                "Keep your assessment if you disagree with the critique and explain why.\n\n"
                f"Original execution data:\n```json\n{__import__('json').dumps(context, indent=2)}\n```\n\n"
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

        # ── Phase 2a: Critic ────────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("  Phase 2a — Critic Review")
        print("─" * 60)
        critique = self._phase2a_critique(context, phase1_results)

        # ── Phase 2b: Revision ──────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("  Phase 2b — Agent Revision")
        print("─" * 60)
        revised_results = self._phase2b_revise(context, phase1_results, critique)

        # ── Phase 3: Final Summary ──────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("  Phase 3 — Summary Director")
        print("─" * 60)
        all_analyses = {
            "phase1": phase1_results,
            "critique": critique,
            "revised": revised_results,
        }
        summary = self._phase3_synthesize(context, all_analyses)

        print(f"\n{'=' * 60}")
        print("  ANALYSIS COMPLETE")
        print("=" * 60)

        return summary
