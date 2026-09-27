#!/usr/bin/env python3
"""Preflight an ONNX model against the published RV1106 RKNN op table."""

from __future__ import annotations

import argparse
import json
import onnx

from rv1106_compat import legalize_rv1106


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", help="write the simplified/legalized ONNX model")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="fail on warnings and unknown ops too")
    args = parser.parse_args()
    simplified, findings = legalize_rv1106(onnx.load(args.model))
    if args.output:
        onnx.save(simplified, args.output)
    if args.json:
        print(json.dumps([finding.json() for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"{finding.level}: {finding.node} ({finding.op_type}): {finding.message}")
        if not findings:
            print("ok: no RV1106 compatibility findings")
    blocking = {"error", "unknown"} | ({"warning"} if args.strict else set())
    return 1 if any(finding.level in blocking for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
