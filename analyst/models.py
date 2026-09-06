"""Data models for the Qsys test execution analyzer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureCategory(Enum):
    SIGNAL_PATH = "signal_path"          # routing / no signal
    GAIN_CALIBRATION = "gain_calibration" # level offset from expected
    CONTROL_FAILURE = "control_failure"   # action didn't apply (e.g. mute stuck)
    TIMING = "timing"                     # measurement before stabilization
    CHANNEL_SPECIFIC = "channel_specific" # one channel fails, other passes
    FREQUENCY_DEPENDENT = "frequency_dependent"  # fails at some freq only
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """A single failed action or verification."""
    action_execution_id: int
    tab_execution_id: int
    tab_name: str
    action_name: str
    expected_values: str
    actual_values: str
    remarks: Optional[str]
    test_case_name: str
    test_plan_name: str


@dataclass
class FailureAnalysis:
    """Analysis result for a group of related failures."""
    category: FailureCategory
    severity: Severity
    affected_signal_path: str
    root_cause: str
    evidence: list[str]
    affected_actions: list[int]  # ActionExecutionIDs
    recommendation: str


@dataclass
class ExecutionSummary:
    """Final summary of a complete test execution analysis."""
    exec_id: str
    total_test_cases: int
    total_passed: int
    total_failed: int
    pass_rate: float
    failure_analyses: list[FailureAnalysis]
    overall_assessment: str
    key_findings: list[str]
    recommended_actions: list[str]
    affected_components: list[str]
