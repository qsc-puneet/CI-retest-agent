"""
LLM Agents for Qsys test execution failure analysis.

Phase 1 — Independent analysis:
  - Failure Classifier: categorizes each failure
  - Pattern Analyst: finds correlations and root causes
  - Impact Assessor: ranks severity and release impact

Phase 2a — Critic validates the analyses
Phase 2b — Agents revise after seeing peer + critic input

Phase 3 — Summary Director produces final report
"""

import json
import textwrap


ANALYSIS_SCHEMA = """\
{
  "failures_analyzed": [
    {
      "action_execution_id": <int>,
      "category": "signal_path | gain_calibration | control_failure | timing | channel_specific | frequency_dependent | unknown",
      "severity": "critical | high | medium | low",
      "root_cause": "...",
      "evidence": ["...", "..."]
    }
  ],
  "summary": "...",
  "confidence": 0.0-1.0
}"""


SUMMARY_SCHEMA = """\
{
  "run_metadata": {
    "exec_id": "...",
    "test_run_name": "...",
    "builds_under_test": ["...", "..."],
    "start_time": "...",
    "end_time": "...",
    "duration": "e.g. 3h 14m"
  },
  "totals": {
    "total_test_cases": <int>,
    "total_passed": <int>,
    "total_failed": <int>,
    "pass_rate": "e.g. 99.4%"
  },
  "overall_status": "pass | unstable | fail",
  "verdict_reasoning": "one sentence explaining the status choice using totals.",
  "executive_summary": "2-3 sentence summary a lead engineer would read first.",
  "failed_test_cases": [
    {
      "test_case_name": "<TestCaseName exactly as it appears in case_summary>",
      "test_plan_name": "<parent TestPlanName>",
      "hardware": "<from plan_summary>",
      "case_execution_id": <int>,
      "failed_actions": [
        {
          "action_execution_id": <int>,
          "tab_name": "...",
          "action_name": "<ActionName exactly as recorded>",
          "time": "HH:MM:SS UTC or full timestamp",
          "actual": "<raw ActualValues>",
          "expected": "<raw ExpectedValues>",
          "remarks": "<raw Remarks — never paraphrase>"
        }
      ],
      "interpretation": "One short paragraph in plain English: what the failure means for the product, what the checker was actually testing, and why the observed actual/expected/remarks combination indicates that.",
      "root_cause_hypotheses": [
        {"hypothesis": "...", "likelihood": "high | medium | low", "evidence": "concrete data point supporting or refuting the hypothesis"}
      ],
      "corroborating_signals": [
        "Cross-check against other test cases in case_summary — e.g. sibling checker <TestCaseName> passed on the same build, so the failure is device-specific.",
        "Or: same script pattern failed in another plan too — infra bug."
      ],
      "severity": "critical | high | medium | low",
      "affected_component": "specific device / subsystem / script — not a generic 'core control team' label"
    }
  ],
  "recommended_actions": [
    {"action": "concrete next step (log to inspect, code path to check, retest command)", "priority": "immediate | next_build | backlog", "owner": "specific team / role"}
  ],
  "retest_guidance": "Which test cases to retest and which to skip, with one-line reason.",
  "confidence": 0.0-1.0
}"""


class BaseAgent:
    """Base class for all analysis agents."""

    name = "BaseAgent"
    role = "base"
    system_prompt = ""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _call_llm(self, messages, temperature=0.3):
        """Send messages to LLM and return response text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content

    def _parse_json(self, raw):
        """Extract the first balanced JSON object from an LLM response.

        Handles code fences, preamble/postamble prose, and multiple JSON
        blocks by scanning for the first `{...}` with matched braces,
        ignoring braces inside string literals.
        """
        text = raw.strip()
        # Strip common code-fence wrappers
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON found in {self.name} response")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i + 1])

        # Fallback: try the naive extraction if brace matching failed
        end = text.rfind("}") + 1
        if end > start:
            return json.loads(text[start:end])
        raise ValueError(f"No balanced JSON object in {self.name} response")


# ── Phase 1 Agents ──────────────────────────────────────────────────────────

class FailureClassifierAgent(BaseAgent):
    name = "Failure Classifier"
    role = "classifier"
    system_prompt = textwrap.dedent("""\
        You are a Qsys verification failure classifier for audio/DSP designs.

        Test structure:
        - Control Actions: configure parameters on signal generators/processors
          (frequency, level, mute, gain, routing)
        - Control Verifications: measure output at network TX/RX points and
          compare against upper/lower dB bounds

        Categories:
        - signal_path: no signal detected (-96 dB), routing disconnected
        - gain_calibration: level offset from bounds (actual within ~10 dB of expected)
        - control_failure: action parameter didn't apply (mute stuck, gain not set)
        - timing: measurement taken before signal stabilized
        - channel_specific: one channel fails while other passes
        - frequency_dependent: fails at certain frequencies only
        - unknown: cannot determine

        If a failure carries a `_domain_context` block, use its
        `checker_family` and `actual_false_means` to pick the category —
        do not override curated knowledge with a guess.

        Classify each failure. Respond ONLY with the JSON object.
    """)

    def analyze(self, context):
        prompt = (
            "Here is the test execution data with failures:\n\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
            f"Classify each failure. Output JSON:\n{ANALYSIS_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ])
        return self._parse_json(raw)


class PatternAnalystAgent(BaseAgent):
    name = "Pattern Analyst"
    role = "pattern_analyst"
    system_prompt = textwrap.dedent("""\
        You are a pattern analysis expert for Qsys audio DSP verification.

        Your job:
        - Identify correlated failures (same channel, same output port,
          same frequency, cascading from a control failure)
        - Determine if multiple failures share a single root cause
        - Distinguish systemic issues from isolated failures
        - Flag cascading failures (e.g., mute stuck → all downstream
          verifications show no signal)

        Consider the test sequences: actions before a verification are
        its preconditions. If a precondition action failed, downstream
        verifications are expected to fail.

        When a failure carries a `_domain_context` block, its
        `common_causes` list is the authoritative candidate set for
        root causes — pick from it rather than inventing new causes.

        Respond ONLY with the JSON object.
    """)

    def analyze(self, context):
        prompt = (
            "Here is the execution data with pre-computed correlations:\n\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
            "Identify failure patterns and root causes.\n"
            f"Output JSON:\n{ANALYSIS_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ])
        return self._parse_json(raw)


class ImpactAssessorAgent(BaseAgent):
    name = "Impact Assessor"
    role = "impact_assessor"
    system_prompt = textwrap.dedent("""\
        You are a test impact assessor for Qsys audio DSP designs.

        Your job:
        - Rank failure severity based on what the test verifies
        - Determine which failures would block a release vs. are acceptable
        - Consider: signal path broken (critical) vs. marginal level
          offset (medium) vs. cosmetic (low)
        - Assess whether passing tests provide confidence that the
          design is fundamentally working

        Severity guide:
        - critical: no signal, core unresponsive, data path broken
        - high: significant level deviation (>6 dB from bounds)
        - medium: moderate deviation (2-6 dB), single channel affected
        - low: marginal fail (<2 dB from bounds), likely measurement noise

        When a failure carries a `_domain_context` block, use its
        `typical_severity` as the default and only override if the
        specific evidence in this run justifies a different severity.

        Respond ONLY with the JSON object.
    """)

    def analyze(self, context):
        prompt = (
            "Here is the execution data:\n\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
            "Assess severity and release impact for each failure.\n"
            f"Output JSON:\n{ANALYSIS_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ])
        return self._parse_json(raw)


# ── Phase 2a: Critic ────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    name = "Critic"
    role = "critic"
    system_prompt = textwrap.dedent("""\
        You are a verification review critic for Qsys audio DSP testing.

        Review the analyses from other agents and:
        - Challenge incorrect classifications
        - Flag missed correlations or root causes
        - Identify false conclusions (e.g., labeling a cascade failure
          as multiple independent issues)
        - Check if severity ratings are appropriate
        - Verify that evidence supports the stated root cause

        Be constructive — correct errors, don't just criticize.
        Respond ONLY with the JSON object.
    """)

    def critique(self, context, agent_analyses):
        analyses_text = json.dumps(agent_analyses, indent=2, default=str)
        prompt = (
            "## Execution Data\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
            "## Agent Analyses to Review\n"
            f"```json\n{analyses_text}\n```\n\n"
            "Review these analyses. Correct any errors.\n"
            f"Output your corrected analysis as JSON:\n{ANALYSIS_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=0.2)
        return self._parse_json(raw)


# ── Phase 3: Summary Director ───────────────────────────────────────────────

class SummaryDirectorAgent(BaseAgent):
    name = "Summary Director"
    role = "director"
    system_prompt = textwrap.dedent("""\
        You are a senior test engineer producing the final execution
        report for a Qsys audio/DSP verification run. Match the depth
        and precision a lead engineer would expect: cite specific
        ActionExecutionIDs, TestCaseNames, actual/expected values and
        remarks verbatim.

        Deterministic rules — apply, do not override:

        1. `overall_status` is derived from `totals` and the severities
           you assign:
             - "fail"     if any failed test case has severity "critical"
             - "unstable" if there is at least one failure but none critical
             - "pass"     only when totals.total_failed == 0
           Set `verdict_reasoning` to explicitly reference the numbers,
           e.g. "173/174 passed; one critical sentinel failed on Core 110f".

        2. Names must be exact:
             - `test_case_name` MUST match a `name` in `case_summary`.
             - Never put an ActionName (e.g. "External Script Action")
               where a TestCaseName is expected.
             - `affected_component` should be a specific device,
               inventory item, or script — not a generic team label.

        3. For every failing test case (see `failed_case_names`), emit
           exactly one entry in `failed_test_cases` and include ALL of
           its failed actions in `failed_actions`, copying the raw
           `actual`, `expected` and `remarks` verbatim (do not paraphrase).

        4. `interpretation` must explain what the checker was actually
           verifying and why the observed data means it failed. If the
           action is an "External Script Action" whose remarks mention
           a specific token (e.g. "0 is not found in the script
           response"), explain what that token normally represents
           (e.g. "0 crashes found") and what its absence implies.

        5. `root_cause_hypotheses` — provide 1 to 3. For each, include
           `likelihood` and concrete `evidence`. Use the retry pattern
           (same action failed twice in a row with identical remarks)
           as evidence against transient network causes.

        6. `corroborating_signals` — cross-reference against
           `case_summary`. Example: if a sibling checker with a similar
           name (e.g. NV-side vs Core-side crash checker) PASSED on the
           same build, note that the failure is device-specific.
           If every similar checker failed, note that the failure is
           systemic.

        7. `recommended_actions` — each action must be concrete
           (a specific log to inspect, script path to check, or retest
           command). No filler like "investigate further".

        8. Confidence:
             - Cap at 0.7 when total_failures < 5.
             - Cap at 0.5 when total_failures < 3 and you had to rely
               on a single hypothesis.

        9. `_domain_context` — when a failed action carries a
           `_domain_context` block, treat it as CURATED GROUND TRUTH
           authored by the domain team:
             - `purpose` tells you what the checker actually verifies —
               use it verbatim material when writing `interpretation`.
             - `actual_false_means` explains what the observed
               False/failure signifies at the device level — do not
               contradict it.
             - `common_causes` is the candidate set for
               `root_cause_hypotheses`. Pick 1-3 that best fit the
               specific evidence in this run; do NOT invent new causes
               that contradict them.
             - `affected_component_hint` describes what `affected_component`
               must name (a device / subsystem / script — NEVER a team,
               org, or role such as "control team").
             - `typical_severity` is the default; only override if the
               concrete evidence justifies a different severity.
             - `remark_meaning` explains what the raw remark string
               indicates; use it to explain the failure in plain English.
             - `remark_rules_out` names failure modes the remark
               eliminates — do not propose a hypothesis it rules out.

        Respond ONLY with the JSON object matching the schema exactly.
        Never invent fields. Do not include commentary outside JSON.
    """)

    def synthesize(self, context, all_analyses):
        analyses_text = json.dumps(all_analyses, indent=2, default=str)
        prompt = (
            "## Execution Data\n"
            f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
            "## All Agent Analyses (post-critique revision)\n"
            f"```json\n{analyses_text}\n```\n\n"
            "Produce the final execution summary report.\n"
            f"Output JSON matching this schema exactly:\n{SUMMARY_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=0.2)
        return self._parse_json(raw)


PHASE1_AGENTS = [
    FailureClassifierAgent,
    PatternAnalystAgent,
    ImpactAssessorAgent,
]


# ═══════════════════════════════════════════════════════════════════════════
#  Retest Planner Agents
#
#  Consume:
#    - failure_matrix: pass/fail history of every testcase across
#      the current run + N prior CI runs (produced by tools.build_failure_matrix)
#    - failure_analysis: root-cause / cascade output from the existing
#      Phase 1-3 analyzer (optional)
#
#  Produce a retest plan that the deterministic ConfigMutator applies to
#  the QAT XML.
# ═══════════════════════════════════════════════════════════════════════════


REGRESSION_SCHEMA = """\
{
  "classifications": [
    {
      "test_case_name": "...",
      "label": "new_regression | flaky | persistent | first_run | cascade_downstream",
      "prior_pass_count": <int>,
      "prior_fail_count": <int>,
      "evidence": "one-line justification"
    }
  ],
  "summary": "...",
  "confidence": 0.0-1.0
}"""


BUILD_CORRELATION_SCHEMA = """\
{
  "suspect_build_windows": [
    {
      "first_bad_build": "...",
      "last_good_build": "... or null",
      "affected_test_cases": ["...", "..."],
      "reasoning": "..."
    }
  ],
  "notes": "...",
  "confidence": 0.0-1.0
}"""


RETEST_PLAN_SCHEMA = """\
{
  "retest_test_cases": ["<TestCaseName>", "..."],
  "skip_persistent": ["<TestCaseName>", "..."],
  "preserved_preconditions": ["<TestCaseName>", "..."],
  "rationale_per_test_case": {
    "<TestCaseName>": "why this decision was made"
  },
  "warnings": ["..."],
  "confidence": 0.0-1.0
}"""


class RegressionClassifierAgent(BaseAgent):
    name = "Regression Classifier"
    role = "regression_classifier"
    system_prompt = textwrap.dedent("""\
        You classify failing Qsys testcases based on their pass/fail
        history across recent CI runs.

        Labels (use exactly one per testcase):
          - new_regression:   currently failing; passed in ALL prior runs
                              where it was executed.
          - flaky:            currently failing; prior runs are mixed
                              (both passes and fails).
          - persistent:       currently failing; failed in ALL prior runs
                              where it was executed.
          - first_run:        currently failing; no prior data (missing
                              or incomplete in every prior run).
          - cascade_downstream: currently failing but the failure analysis
                              indicates it is a downstream effect of another
                              failing testcase (e.g. firmware upgrade broke,
                              so verification tests can't run correctly).

        Only classify testcases that are currently failing. Use the
        deterministic classification already computed in the matrix
        (`classification` field) as a strong prior — only override it
        when the failure_analysis provides clear cascade evidence.

        Respond ONLY with the JSON object.
    """)

    def analyze(self, failure_matrix, failure_analysis=None):
        payload = {
            "failure_matrix": failure_matrix,
            "failure_analysis": failure_analysis or {},
        }
        prompt = (
            "Classify every currently-failing testcase.\n\n"
            f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n\n"
            f"Output JSON:\n{REGRESSION_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=0.2)
        return self._parse_json(raw)


class BuildCorrelationAgent(BaseAgent):
    name = "Build Correlation"
    role = "build_correlation"
    system_prompt = textwrap.dedent("""\
        You identify which Q-Sys build introduced a regression, based on
        per-run build labels and pass/fail history.

        For each cluster of tests that started failing together, report:
          - first_bad_build:  the earliest build where the failure appears
          - last_good_build:  the most recent build where those tests passed
                              (null if no such build in the window)
          - affected_test_cases

        Only report a suspect window when there is a clear
        pass -> fail transition in the history. If everything was
        already failing in the oldest available run, mark
        `last_good_build` as null.

        Respond ONLY with the JSON object.
    """)

    def analyze(self, failure_matrix):
        prompt = (
            "Identify build windows where regressions first appeared.\n\n"
            f"```json\n{json.dumps(failure_matrix, indent=2, default=str)}\n```\n\n"
            f"Output JSON:\n{BUILD_CORRELATION_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=0.2)
        return self._parse_json(raw)


class RetestPlannerAgent(BaseAgent):
    name = "Retest Planner"
    role = "retest_planner"
    system_prompt = textwrap.dedent("""\
        You produce a retest plan for a Qsys QAT run: the exact list of
        TestCase names that should be re-executed.

        Decision rules:
          - new_regression   -> RETEST
          - flaky            -> RETEST (mention flakiness in rationale)
          - first_run        -> RETEST
          - cascade_downstream -> RETEST only if the upstream cause is
                                  also being retested; otherwise skip
                                  and note in warnings.
          - persistent       -> SKIP (put in skip_persistent); retesting
                                won't reveal new info. Recommend a bug
                                filing in the rationale.
          - recovered / still_passing / passing testcases -> do NOT retest.

        Always keep the following preconditions checked, even if they
        passed — regardless of label. Include their names in
        `preserved_preconditions`:
          - Any testcase whose name contains "Firmware", "firmware",
            "upgrade", "Upgrade" (firmware upgrade steps).
          - Any testcase whose name contains "Status check",
            "statuscheck", "Environment Setup", "camera_position",
            "camera exposure" (environment preconditions).
          - Any testcase whose name contains "Crash_Checker",
            "background_monitoring" (monitoring / sentinel tests).
          - Any testcase whose name contains "QRCM" (control API check).

        `retest_test_cases` MUST include every retested failure name AND
        every preserved precondition name. This is the single source of
        truth handed to the deterministic mutator.

        Respond ONLY with the JSON object. Only reference testcase names
        that appear in the failure_matrix.
    """)

    def plan(self, failure_matrix, regression_result, build_correlation, failure_analysis=None):
        payload = {
            "failure_matrix": failure_matrix,
            "regression_classification": regression_result,
            "build_correlation": build_correlation,
            "failure_analysis": failure_analysis or {},
        }
        prompt = (
            "Produce the retest plan.\n\n"
            f"```json\n{json.dumps(payload, indent=2, default=str)}\n```\n\n"
            f"Output JSON:\n{RETEST_PLAN_SCHEMA}"
        )
        raw = self._call_llm([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=0.2)
        return self._parse_json(raw)


RETEST_AGENTS = [
    RegressionClassifierAgent,
    BuildCorrelationAgent,
    RetestPlannerAgent,
]
