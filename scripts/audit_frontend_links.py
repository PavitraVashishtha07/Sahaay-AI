import os
import re

pub_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
deleted_patterns = [
    "index.html",
    "chat.html",
    "onboarding.html",
    "profile-settings.html",
    "recommendation.html",
    "states.html",
    "stitch",
]

all_clean = True
for fname in sorted(os.listdir(pub_dir)):
    if not fname.endswith(".html"):
        continue
    filepath = os.path.join(pub_dir, fname)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_idx, line in enumerate(lines, 1):
        for pattern in deleted_patterns:
            if re.search(r'(?:href|onclick|src)=[\'\"][^\'\"]*' + re.escape(pattern), line, re.IGNORECASE):
                print(f"ISSUE in {fname}:{line_idx} -> {pattern} referenced: {line.strip()}")
                all_clean = False

if all_clean:
    print("ALL CLEAN: No dead links or references to retired files in frontend/public/*.html")
