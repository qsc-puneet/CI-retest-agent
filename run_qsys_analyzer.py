"""
Qsys Test Execution Analyzer — Entry Point

Usage:
    python run_qsys_analyzer.py <ExecID>
    python run_qsys_analyzer.py 287
    python run_qsys_analyzer.py 287 --all   (analyze all test cases, not just failures)
"""

import json
import os
import sys


def main():
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Parse args
    if len(sys.argv) < 2:
        print("Usage: python run_qsys_analyzer.py <ExecID> [--all]")
        print("  ExecID: The execution ID to analyze")
        print("  --all:  Analyze all test cases (default: only failed)")
        sys.exit(1)

    exec_id = sys.argv[1]
    failed_only = "--all" not in sys.argv

    # Validate env
    required_vars = ["MSSQL_HOST", "MSSQL_USER", "MSSQL_PASSWORD", "OPENAI_API_KEY", "LLM_MODEL"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Add them to .env or export them.")
        sys.exit(1)

    model = os.environ["LLM_MODEL"]
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY")
    # Model family slug (e.g. "llama3.1", "qwen2.5") tags outputs so old and new
    # model reports can coexist for comparison
    model_slug = model.split(":")[0].replace("/", "-")
    default_json = f"qsys_analysis_output_{model_slug}.json"
    output_path = os.environ.get("OUTPUT_JSON_PATH", default_json)
    report_path = os.environ.get("OUTPUT_REPORT_PATH") or (
        os.path.splitext(output_path)[0] + ".md"
    )
    # PIPELINE_MODE: 'fast' (default) skips the critic/revise pass; 'full' runs it.
    pipeline_mode = os.environ.get("PIPELINE_MODE", "fast").lower()
    # Per-request timeout in seconds; slow 7B models on remote hosts can stall.
    try:
        llm_timeout = float(os.environ.get("LLM_TIMEOUT", "600"))
    except ValueError:
        llm_timeout = 600.0

    # Create LLM client
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=llm_timeout)

    # Run analysis
    from qsys_analyzer.orchestrator import QsysAnalyzer
    from qsys_analyzer.report import render_markdown_report
    analyzer = QsysAnalyzer(client=client, model=model, enable_revise=(pipeline_mode == "full"))
    summary = analyzer.run(exec_id, failed_only=failed_only)

    # Print summary
    print("\n")
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
    else:
        status = summary.get("overall_status", "unknown")
        print(f"  Status:  {status.upper()}")
        print(f"  Summary: {summary.get('executive_summary', 'N/A')}")

        if summary.get("key_findings"):
            print("\n  Key Findings:")
            for finding in summary["key_findings"]:
                print(f"    - {finding}")

        if summary.get("failure_groups"):
            print("\n  Failure Groups:")
            for group in summary["failure_groups"]:
                print(f"    [{group.get('severity', '?').upper()}] {group.get('group_name', 'N/A')}")
                print(f"      Root cause: {group.get('root_cause', 'N/A')}")
                print(f"      Action: {group.get('recommendation', 'N/A')}")

        if summary.get("recommended_actions"):
            print("\n  Recommended Actions:")
            for action in summary["recommended_actions"]:
                print(f"    [{action.get('priority', '?')}] {action.get('action', 'N/A')}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full report written to: {output_path}")

    try:
        report_md = render_markdown_report(summary, exec_id=exec_id)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"  Markdown report written to: {report_path}")
    except Exception as exc:
        print(f"  WARN: could not render Markdown report: {exc}")


if __name__ == "__main__":
    main()
