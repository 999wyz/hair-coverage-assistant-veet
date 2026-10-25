"""Regression checks for relaxed photo-comparison gating."""
import ast
from pathlib import Path
import numpy as np

tree = ast.parse(Path(__file__).with_name("app.py").read_text(encoding="utf-8"))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "comparison_decision")
scope = {"np": np}
exec(compile(ast.Module(body=[node], type_ignores=[]), "comparison", "exec"), scope)
decide = scope["comparison_decision"]


def result(status="可分析", brightness=140, fraction=0.5, coverage=8, valid=True):
    return {
        "quality": {"状态": status, "亮度": brightness},
        "skin_fraction": fraction, "coverage": coverage,
        "analysis_region": np.full((20, 20), 255 if valid else 0, np.uint8),
    }


cases = [
    ("similar photos", result(), result(coverage=2), "可比较"),
    ("lighting difference", result(brightness=60), result(brightness=220), "部分可比较"),
    ("different framing", result(fraction=0.6), result(fraction=0.1), "部分可比较"),
    ("partial quality", result("部分可分析"), result(), "部分可比较"),
    ("quality warning but valid region", result("建议重拍"), result(), "部分可比较"),
    ("no skin region", result(valid=False), result(), "建议重拍"),
    ("invalid coverage", result(coverage=float("nan")), result(), "建议重拍"),
    ("zero baseline", result(coverage=0), result(coverage=2), "可比较"),
]
for name, before, after, expected in cases:
    actual, _ = decide(before, after)
    assert actual == expected, (name, actual, expected)
    print(f"PASS {name}: {actual}")
