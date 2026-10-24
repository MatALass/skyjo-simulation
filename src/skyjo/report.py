"""Affichage console et export CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_COLUMNS = [
    ("strategy", "stratégie", "<", 34, "{}"),
    ("win_rate", "victoires", ">", 10, "{:.1%}"),
    ("win_ci", "IC 95 %", ">", 15, "{}"),
    ("win_index", "indice", ">", 7, "{:.2f}"),
    ("avg_round_score", "pts/manche", ">", 11, "{:.2f}"),
    ("round_score_ci95", "±", ">", 6, "{:.2f}"),
    ("round_win_rate", "manches gagnées", ">", 16, "{:.1%}"),
    ("finish_rate", "finit", ">", 7, "{:.1%}"),
    ("doubled_rate", "doublé", ">", 7, "{:.1%}"),
]


def format_table(rows: list[dict[str, Any]], title: str) -> str:
    header = "".join(f"{label:{align}{w}}" for _, label, align, w, _ in _COLUMNS)
    lines = [title, header, "-" * len(header)]
    for r in rows:
        r = {**r, "win_ci": f"[{r['win_rate_ci_low']:.1%}–{r['win_rate_ci_high']:.1%}]"}
        lines.append("".join(f"{fmt.format(r[key]):{align}{w}}" for key, _, align, w, fmt in _COLUMNS))
    return "\n".join(lines)


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
