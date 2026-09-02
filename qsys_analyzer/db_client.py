"""
DB Client — Fetches Qsys test execution data from MSSQL.

Hierarchy:
  ExecID → TestPlanGroup → TestPlan → TestCase → Tab → Action/Verification

Database: [QE_QAT_Server_Runner]
Tables:
  - TestPlanGroupTable  (top-level test suites)
  - TestPlanTable       (test plans within a group)
  - TestCaseTable       (test cases within a plan)
  - TabTable            (tabs within a test case)
  - ActionTable         (actions/verifications within a tab)
"""

import os
import pymssql


def _get_connection():
    """Create a connection to MSSQL using env variables."""
    return pymssql.connect(
        server=os.environ["MSSQL_HOST"],
        port=int(os.environ.get("MSSQL_PORT", "1433")),
        user=os.environ["MSSQL_USER"],
        password=os.environ["MSSQL_PASSWORD"],
        database=os.environ.get("MSSQL_DATABASE", "QE_QAT_Server_Runner"),
    )


def _query(sql, params=None):
    """Execute a query and return rows as list of dicts."""
    conn = _get_connection()
    cursor = conn.cursor(as_dict=True)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


# ── Level 1: Test Plan Groups (from ExecID) ────────────────────────────────

def fetch_plan_groups(exec_id):
    """Fetch all test plan groups for a given Execution ID."""
    sql = """
        SELECT PlanGroupExecutionID, ExecID, TestPlanGroupName,
               TestPlanGroupType, Status, TotalTestPlansCount,
               SelectedTestPlans, ExecutedTestPlans,
               TotalPassedTestPlans, TotalFailedTestPlans,
               TotalIncompleteTestPlans, LoopIterationCount,
               DesignDeployOption, StartTime, EndTime, Remarks
        FROM [dbo].[TestPlanGroupTable]
        WHERE ExecID = %s
    """
    return _query(sql, (str(exec_id),))


# ── Level 2: Test Plans (from PlanGroupExecutionIDs) ────────────────────────

def fetch_test_plans(plan_group_ids):
    """Fetch all test plans for given PlanGroupExecutionIDs."""
    if not plan_group_ids:
        return []
    placeholders = ",".join(["%s"] * len(plan_group_ids))
    sql = f"""
        SELECT PlanExecutionID, PlanGroupExecutionID, TestPlanName,
               Status, Hardware, Feature, Component, Collection,
               Details, TotalTestCasesCount, TestCaseSelected,
               TestCaseExecuted, TotalPassedTestCase, TotalFailedTestCase,
               TotalIncompleteTestCase, StartTime, EndTime,
               Build, DesignName, Inventory, Remarks,
               TestPlanLoopIterations
        FROM [dbo].[TestPlanTable]
        WHERE PlanGroupExecutionID IN ({placeholders})
    """
    return _query(sql, tuple(str(i) for i in plan_group_ids))


# ── Level 3: Test Cases (from PlanExecutionIDs) ────────────────────────────

def fetch_test_cases(plan_execution_ids, status_filter=None):
    """Fetch test cases. Optionally filter by status ('Fail', 'Pass', etc.)."""
    if not plan_execution_ids:
        return []
    placeholders = ",".join(["%s"] * len(plan_execution_ids))
    sql = f"""
        SELECT CaseExecutionID, PlanExecutionID, TestCaseName,
               Status, Labels, Details, StartTime, EndTime,
               Build, Remarks
        FROM [dbo].[TestCaseTable]
        WHERE PlanExecutionID IN ({placeholders})
    """
    params = tuple(str(i) for i in plan_execution_ids)
    if status_filter:
        sql += " AND Status = %s"
        params += (status_filter,)
    return _query(sql, params)


# ── Level 4: Tabs (from CaseExecutionIDs) ──────────────────────────────────

def fetch_tabs(case_execution_ids):
    """Fetch all tabs for given CaseExecutionIDs."""
    if not case_execution_ids:
        return []
    placeholders = ",".join(["%s"] * len(case_execution_ids))
    sql = f"""
        SELECT TabExecutionID, CaseExecutionID, TabName,
               Status, StartTime, EndTime, Remarks
        FROM [dbo].[TabTable]
        WHERE CaseExecutionID IN ({placeholders})
    """
    return _query(sql, tuple(str(i) for i in case_execution_ids))


# ── Level 5: Actions & Verifications (from TabExecutionIDs) ────────────────

def fetch_actions(tab_execution_ids):
    """Fetch all actions/verifications for given TabExecutionIDs."""
    if not tab_execution_ids:
        return []
    placeholders = ",".join(["%s"] * len(tab_execution_ids))
    sql = f"""
        SELECT ActionExecutionID, TabExecutionID, Status,
               StartTime, EndTime, ActionName, Remarks,
               ActualValues, ExpectedValues
        FROM [dbo].[ActionTable]
        WHERE TabExecutionID IN ({placeholders})
    """
    return _query(sql, tuple(str(i) for i in tab_execution_ids))


# ── High-Level: Full execution fetch (all levels) ──────────────────────────

def fetch_full_execution(exec_id, failed_only=False):
    """
    Traverse the full hierarchy for an ExecID and return structured data.

    If failed_only=True, only fetches test cases with Status='Fail'
    and their associated tabs/actions.

    Returns:
        {
            "exec_id": str,
            "plan_groups": [...],
            "test_plans": [...],
            "test_cases": [...],
            "tabs": [...],
            "actions": [...]
        }
    """
    # Level 1
    plan_groups = fetch_plan_groups(exec_id)
    pg_ids = [pg["PlanGroupExecutionID"] for pg in plan_groups]

    # Level 2
    test_plans = fetch_test_plans(pg_ids)
    tp_ids = [tp["PlanExecutionID"] for tp in test_plans]

    # Level 3 — always fetch all test cases so cross-referencing passing vs failing is possible.
    test_cases = fetch_test_cases(tp_ids)
    if failed_only:
        drill_cases = [tc for tc in test_cases if (tc.get("Status") or "").strip().lower() == "fail"]
    else:
        drill_cases = test_cases
    tc_ids = [tc["CaseExecutionID"] for tc in drill_cases]

    # Level 4
    tabs = fetch_tabs(tc_ids)
    tab_ids = [t["TabExecutionID"] for t in tabs]

    # Level 5
    actions = fetch_actions(tab_ids)

    return {
        "exec_id": str(exec_id),
        "plan_groups": plan_groups,
        "test_plans": test_plans,
        "test_cases": test_cases,
        "tabs": tabs,
        "actions": actions,
    }


def fetch_execution_summary(exec_id):
    """Fetch just the top-level summary (plan groups + plans) without drilling into actions."""
    plan_groups = fetch_plan_groups(exec_id)
    pg_ids = [pg["PlanGroupExecutionID"] for pg in plan_groups]
    test_plans = fetch_test_plans(pg_ids)

    total_cases = sum(tp.get("TotalTestCasesCount", 0) or 0 for tp in test_plans)
    total_passed = sum(tp.get("TotalPassedTestCase", 0) or 0 for tp in test_plans)
    total_failed = sum(tp.get("TotalFailedTestCase", 0) or 0 for tp in test_plans)

    return {
        "exec_id": str(exec_id),
        "plan_groups": plan_groups,
        "test_plans": test_plans,
        "total_test_cases": total_cases,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "pass_rate": round(total_passed / max(total_cases, 1) * 100, 1),
    }


# ── TestRunTable: run-level metadata & history ─────────────────────────────

_RERUN_SUFFIX = "_Rerun"


def strip_rerun_suffix(test_run_name):
    """Return the base TestRunName (drops a trailing '_Rerun', case-sensitive)."""
    if test_run_name and test_run_name.endswith(_RERUN_SUFFIX):
        return test_run_name[: -len(_RERUN_SUFFIX)]
    return test_run_name


def fetch_test_run(exec_id):
    """
    Fetch the TestRunTable row for a given ExecID.

    Returns a dict with the columns needed by the retest planner, or None.
    """
    sql = """
        SELECT ExecID, TestRunName, CI_CDRun, status,
               RunTime, StartDateTime, TotalIncompleteTestCase,
               TotalFailedTestCase, QsysBuildUnderTest
        FROM [dbo].[TestRunTable]
        WHERE ExecID = %s
    """
    rows = _query(sql, (str(exec_id),))
    return rows[0] if rows else None


def fetch_relevant_prior_runs(
    base_test_run_name,
    before_datetime,
    exclude_exec_id,
    n=3,
    min_runtime_seconds=7200,
    require_ci_cd=True,
):
    """
    Find up to `n` most recent complete, relevant prior runs.

    A run is "relevant" if:
      - TestRunName == base_test_run_name (i.e. same suite, not a _Rerun)
      - StartDateTime is strictly before `before_datetime`
      - status = 'Completed'
      - TotalIncompleteTestCase = 0
      - RunTime >= `min_runtime_seconds` (default 2h — filters out
        ad-hoc partial runs)
      - CI_CDRun = 1 when `require_ci_cd` (default True — scheduled runs only)
      - ExecID != `exclude_exec_id`

    Fetches a small pool ordered by StartDateTime DESC and returns the
    first `n`. Returns fewer than `n` if not enough qualify.
    """
    sql = """
        SELECT TOP 20
               ExecID, TestRunName, CI_CDRun, status,
               RunTime, StartDateTime, TotalIncompleteTestCase,
               TotalFailedTestCase, QsysBuildUnderTest
        FROM [dbo].[TestRunTable]
        WHERE TestRunName = %s
          AND ExecID <> %s
          AND StartDateTime < %s
          AND status = %s
          AND TotalIncompleteTestCase = 0
          AND RunTime >= %s
    """
    params = [
        base_test_run_name,
        str(exclude_exec_id),
        before_datetime,
        "Completed",
        int(min_runtime_seconds),
    ]
    if require_ci_cd:
        sql += " AND CI_CDRun = 1"
    sql += " ORDER BY StartDateTime DESC"

    rows = _query(sql, tuple(params))
    return rows[: int(n)]
