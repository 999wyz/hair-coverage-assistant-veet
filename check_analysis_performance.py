"""Regression checks against the previously deployed detector and CPU bounds."""
import ast
import logging
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image


project = Path(__file__).resolve().parent


def load_functions(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    scope = {"cv2": cv2, "np": np, "logging": logging, "time": time, "Image": Image}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "analysis", "exec"), scope)
    return scope


reference = subprocess.check_output(
    ["git", "show", "5f87eb3:app.py"], cwd=project, text=True, encoding="utf-8"
)
current = (project / "app.py").read_text(encoding="utf-8")
old = load_functions("from __future__ import annotations\n" + reference)
new = load_functions("from __future__ import annotations\n" + current)

# Thousands of small components trigger the former repeated full-image scans.
image = np.full((768, 768, 3), 180, np.uint8)
for y in range(12, 750, 14):
    for x in range(12, 750, 14):
        cv2.line(image, (x, y), (x + 6, y + 4), (90, 90, 90), 2)
skin = np.full(image.shape[:2], 255, np.uint8)
started = time.perf_counter()
expected = old["visible_hair_mask"](image, skin)
old_time = time.perf_counter() - started
started = time.perf_counter()
actual = new["visible_hair_mask"](image, skin)
new_time = time.perf_counter() - started
assert np.array_equal(expected, actual), "Hair mask changed"
print(f"PASS identical hair mask: previous={old_time:.3f}s current={new_time:.3f}s")

# Observe actual GraphCut call dimensions, not just the output size.
grabcut = cv2.grabCut
shapes = []


def observed_grabcut(image, *args, **kwargs):
    shapes.append(image.shape[:2])
    return grabcut(image, *args, **kwargs)


cv2.grabCut = observed_grabcut
try:
    image = np.full((1536, 2048, 3), (80, 85, 90), np.uint8)
    cv2.ellipse(image, (1024, 768), (540, 740), 0, 0, 360, (130, 160, 190), -1)
    started = time.perf_counter()
    mask = new["central_foreground_mask"](image)
    elapsed = time.perf_counter() - started
    assert shapes and all(max(shape) <= 512 for shape in shapes)
    assert mask.shape == image.shape[:2]
    assert mask[768, 1024] == 255 and mask[768, 0] == 0
    print(f"PASS 2048px foreground: GraphCut input={shapes}, elapsed={elapsed:.3f}s")
finally:
    cv2.grabCut = grabcut

for dimension in (48, 96):
    image = np.full((dimension, dimension, 3), 180, np.uint8)
    skin = np.full(image.shape[:2], 255, np.uint8)
    assert np.array_equal(old["visible_hair_mask"](image, skin), new["visible_hair_mask"](image, skin))
    assert not np.any(new["visible_hair_mask"](image, np.zeros_like(skin)))
print("PASS smooth skin and empty analysis region")
