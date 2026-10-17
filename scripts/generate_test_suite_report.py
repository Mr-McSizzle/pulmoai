#!/usr/bin/env python3
"""Generate Final Test Suite Report Section from Unittest Log.

Reads reports/logs/unittest_9c.txt, computes per-module test counts,
counts '... ok' occurrences, and formats the test inventory markdown table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "reports" / "logs" / "unittest_9c.txt"


def parse_unittest_log(log_path: Path) -> dict:
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}", file=sys.stderr)
        return {}

    test_pattern = re.compile(r"^([a-zA-Z0-9_]+)\s+\(([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", re.MULTILINE)
    module_tests: dict[str, list[dict[str, str]]] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Exact grep counts
    grep_inline_ok = len(re.findall(r" \.\.\. ok", content))
    grep_total_ok = len(re.findall(r"(\.\.\. ok|^ok$)", content, re.MULTILINE))

    for m in test_pattern.finditer(content):
        test_name, mod_name, cls_name = m.groups()
        if mod_name not in module_tests:
            module_tests[mod_name] = []
        module_tests[mod_name].append({
            "test_name": test_name,
            "class_name": cls_name,
        })

    return {
        "grep_inline_ok": grep_inline_ok,
        "grep_total_ok": grep_total_ok,
        "module_tests": module_tests,
        "total_parsed_tests": sum(len(t) for t in module_tests.values()),
    }


def generate_markdown_report(data: dict) -> str:
    lines = []
    lines.append("### Final Test Suite Inventory (Pass 9c)")
    lines.append("")
    lines.append(f"- **Total Passing Tests (Single-line `grep -c ' ... ok'`):** {data['grep_inline_ok']}")
    lines.append(f"- **Total Passing Tests (All ok completions `grep -c -E '(\\.\\.\\. ok|^ok$)'`):** {data['grep_total_ok']}")
    lines.append(f"- **Total Parsed Executed Tests:** {data['total_parsed_tests']}")
    lines.append(f"- **Total Modules:** {len(data['module_tests'])}")
    lines.append("")
    lines.append("| Module Name | Test Count | Test Classes Included |")
    lines.append("| :--- | :--- | :--- |")

    for mod_name in sorted(data["module_tests"].keys()):
        tests = data["module_tests"][mod_name]
        classes = sorted(list(set(t["class_name"] for t in tests)))
        lines.append(f"| `{mod_name}` | {len(tests)} | {', '.join(classes)} |")

    lines.append("")
    lines.append("#### Per-Module Detailed Test List")
    lines.append("")
    for mod_name in sorted(data["module_tests"].keys()):
        tests = data["module_tests"][mod_name]
        lines.append(f"**`{mod_name}` ({len(tests)} tests):**")
        for t in tests:
            lines.append(f"- `{t['test_name']}` (`{t['class_name']}`)")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    data = parse_unittest_log(LOG_PATH)
    if not data:
        return
    print(f"Parsed {data['total_parsed_tests']} tests across {len(data['module_tests'])} modules.")
    print(f"Single-line '... ok' count: {data['grep_inline_ok']}")
    print(f"Total '... ok' / '^ok$' count: {data['grep_total_ok']}")
    print("\nPer-module test counts derived from log:")
    for mod in sorted(data["module_tests"].keys()):
        print(f"  - {mod}: {len(data['module_tests'][mod])} tests")

    md_table = generate_markdown_report(data)
    out_path = PROJECT_ROOT / "reports" / "final_test_suite_inventory.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"\nWrote formatted test suite markdown to {out_path}")


if __name__ == "__main__":
    main()
