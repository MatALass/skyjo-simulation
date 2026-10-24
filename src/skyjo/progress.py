"""Barre de progression (sans dépendance) et exécution parallèle de tâches avec suivi."""

from __future__ import annotations

import multiprocessing
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TextIO, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


class Progress:
    """Barre de progression sur stderr.

    En terminal : une ligne réécrite sur place (au plus toutes les `min_interval` secondes).
    Hors terminal (sortie redirigée vers un fichier) : une ligne toutes les `log_interval` secondes.
    """

    WIDTH = 30

    def __init__(
        self,
        total: int,
        label: str = "",
        groups: int = 0,
        unit: str = "parties",
        groups_unit: str = "tables",
        stream: TextIO | None = None,
        min_interval: float = 0.5,
        log_interval: float = 30.0,
    ) -> None:
        self.total = max(total, 1)
        self.label = label
        self.groups = groups
        self.unit = unit
        self.groups_unit = groups_unit
        self.stream = stream if stream is not None else sys.stderr
        self.tty = getattr(self.stream, "isatty", lambda: False)()
        self.interval = min_interval if self.tty else log_interval
        self.done = 0
        self.groups_done = 0
        self.start = time.monotonic()
        self._last_render = float("-inf")
        self._last_state: tuple[int, int] | None = None
        self._last_len = 0

    def __enter__(self) -> Progress:
        self.render(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(interrupted=exc_type is not None)

    def update(self, n: int) -> None:
        self.done += n
        self.render()

    def group_done(self) -> None:
        self.groups_done += 1
        self.render(force=True)

    def line(self, compact: bool = False) -> str:
        elapsed = time.monotonic() - self.start
        frac = min(self.done / self.total, 1.0)
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        width = self.WIDTH // 2 if compact else self.WIDTH
        filled = int(frac * width)
        bar = "#" * filled + "-" * (width - filled)
        parts = [f"{self.label} [{bar}] {frac:6.1%}", f"{self.done}/{self.total}"]
        if self.groups > 1:
            parts.append(f"{self.groups_done}/{self.groups} {self.groups_unit}")
        if not compact:
            parts.append(f"{rate:.0f} {self.unit}/s")
            parts.append(f"écoulé {_fmt_duration(elapsed)}")
        parts.append(f"reste ~{_fmt_duration(eta)}")
        return " | ".join(parts).strip()

    def render(self, force: bool = False) -> None:
        now = time.monotonic()
        state = (self.done, self.groups_done if self.groups > 1 else 0)
        if state == self._last_state and not (self.tty and force):
            return  # rien de neuf : pas de ligne dupliquée dans un fichier
        if not force and now - self._last_render < self.interval and self.done < self.total:
            return
        self._last_render, self._last_state = now, state
        text = self.line()
        if self.tty:
            # Une ligne plus large que le terminal passe à la ligne et "\r" ne l'efface plus.
            width = shutil.get_terminal_size((120, 20)).columns - 1
            text = self.line(compact=len(text) > width)[:width]
            pad = " " * max(self._last_len - len(text), 0)  # efface la fin de la ligne précédente
            self.stream.write("\r" + text + pad)
            self._last_len = len(text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def close(self, interrupted: bool = False) -> None:
        self.render(force=True)
        if self.tty:
            self.stream.write("\n")
        if interrupted:
            self.stream.write("Interrompu : résultats partiels non sauvegardés.\n")
        self.stream.flush()


def execute(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    jobs: int = 1,
    progress: Progress | None = None,
    size: Callable[[T], int] = lambda _: 1,
    group: Callable[[T], Hashable] = lambda _: 0,
) -> list[R]:
    """Exécute `worker` sur chaque tâche (en parallèle si jobs > 1) ; résultats dans l'ordre des tâches.

    Les tâches sont soumises en entrelaçant les groupes (1er paquet de chaque table, puis 2e, ...) :
    toutes les tables avancent ensemble, ce qui rend l'estimation du temps restant représentative
    et évite qu'une table lente se retrouve seule à la fin.
    """
    group_ids: dict[Hashable, int] = {}
    rank_in_group: list[int] = []
    seen: Counter[Hashable] = Counter()
    for t in tasks:
        g = group(t)
        group_ids.setdefault(g, len(group_ids))
        rank_in_group.append(seen[g])
        seen[g] += 1
    remaining = Counter(seen)
    order = sorted(range(len(tasks)), key=lambda i: (rank_in_group[i], group_ids[group(tasks[i])]))
    results: list[R | None] = [None] * len(tasks)

    def on_done(i: int, result: R) -> None:
        results[i] = result
        g = group(tasks[i])
        remaining[g] -= 1
        if progress is not None:
            progress.update(size(tasks[i]))
            if remaining[g] == 0:
                progress.group_done()

    if jobs <= 1 or len(tasks) <= 1:
        for i in order:
            on_done(i, worker(tasks[i]))
    else:
        # « spawn » partout (défaut de Windows) : un fork d'un processus qui a déjà des threads (pandas, pyarrow,
        # scikit-learn) peut se bloquer, et le comportement devient identique sur tous les systèmes.
        ex = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"))
        try:
            futures = {ex.submit(worker, tasks[i]): i for i in order}
            for fut in as_completed(futures):
                on_done(futures[fut], fut.result())
        except BaseException:  # Ctrl+C compris : on n'attend pas les tâches en file
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
    return results  # type: ignore[return-value]
