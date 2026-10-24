"""Plateau d'un joueur (grille 3x4) et vue en lecture seule pour les stratégies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

ROWS = 3
COLS = 4

# Position = (colonne, ligne). Les indices restent stables même après suppression d'une colonne.
Pos = tuple[int, int]


@dataclass
class Slot:
    value: int
    visible: bool = False


class Board:
    def __init__(self, cards: list[int]) -> None:
        if len(cards) != ROWS * COLS:
            raise ValueError(f"Un plateau demande {ROWS * COLS} cartes, reçu {len(cards)}.")
        # None = colonne supprimée.
        self._columns: list[list[Slot] | None] = [
            [Slot(cards[c * ROWS + r]) for r in range(ROWS)] for c in range(COLS)
        ]

    @classmethod
    def from_columns(cls, columns: list[list[Slot] | None]) -> Board:
        """Plateau arbitraire : 4 colonnes de 3 Slot, None = colonne déjà retirée."""
        if len(columns) != COLS or any(c is not None and len(c) != ROWS for c in columns):
            raise ValueError(f"Il faut {COLS} colonnes de {ROWS} cartes (ou None).")
        board = cls.__new__(cls)
        board._columns = [None if c is None else [Slot(s.value, s.visible) for s in c] for c in columns]
        return board

    def copy(self) -> Board:
        return Board.from_columns(self._columns)

    # ---------- lecture ----------
    def slot(self, pos: Pos) -> Slot:
        col, row = pos
        if not (0 <= col < COLS and 0 <= row < ROWS):
            raise ValueError(f"Position hors plateau : {pos}")
        column = self._columns[col]
        if column is None:
            raise ValueError(f"La colonne {col} a été supprimée.")
        return column[row]

    def column(self, col: int) -> list[Slot] | None:
        return self._columns[col]

    def column_indices(self) -> list[int]:
        return [c for c, col in enumerate(self._columns) if col is not None]

    def cells(self) -> list[list[tuple[int, bool]] | None]:
        """Instantané complet (valeurs cachées comprises) : 4 colonnes de (valeur, visible), None = retirée."""
        return [None if col is None else [(s.value, s.visible) for s in col] for col in self._columns]

    def has(self, pos: Pos) -> bool:
        col, row = pos
        return 0 <= col < COLS and 0 <= row < ROWS and self._columns[col] is not None

    def positions(self) -> list[Pos]:
        return [(c, r) for c, column in enumerate(self._columns) if column is not None for r in range(ROWS)]

    def hidden_positions(self) -> list[Pos]:
        return [p for p in self.positions() if not self.slot(p).visible]

    def visible_positions(self) -> list[Pos]:
        return [p for p in self.positions() if self.slot(p).visible]

    def hidden_count(self) -> int:
        return len(self.hidden_positions())

    def all_visible(self) -> bool:
        return self.hidden_count() == 0

    def visible_score(self) -> int:
        return sum(self.slot(p).value for p in self.visible_positions())

    def total_score(self) -> int:
        """Somme de toutes les cartes restantes (visibles ou non)."""
        return sum(self.slot(p).value for p in self.positions())

    # ---------- actions ----------
    def reveal(self, pos: Pos) -> int:
        slot = self.slot(pos)
        if slot.visible:
            raise ValueError(f"La carte {pos} est déjà visible.")
        slot.visible = True
        return slot.value

    def reveal_all(self) -> None:
        for p in self.hidden_positions():
            self.slot(p).visible = True

    def replace(self, pos: Pos, card: int) -> int:
        """Pose `card` face visible en `pos` et renvoie l'ancienne carte."""
        slot = self.slot(pos)
        old = slot.value
        slot.value = card
        slot.visible = True
        return old

    def remove_completed_columns(self, only: Iterable[int] | None = None) -> list[int]:
        """Supprime les colonnes de 3 cartes visibles identiques ; renvoie les cartes retirées."""
        cols = range(COLS) if only is None else only
        removed: list[int] = []
        for c in cols:
            column = self._columns[c]
            if column is None:
                continue
            if all(s.visible for s in column) and len({s.value for s in column}) == 1:
                removed.extend(s.value for s in column)
                self._columns[c] = None
        return removed

    def __repr__(self) -> str:
        lines = []
        for r in range(ROWS):
            cells = []
            for column in self._columns:
                if column is None:
                    cells.append("  .")
                else:
                    s = column[r]
                    cells.append(f"{s.value:>3}" if s.visible else "  ?")
            lines.append(" ".join(cells))
        return "\n".join(lines)


class BoardView:
    """Vue en lecture seule : une stratégie ne peut jamais lire une carte cachée."""

    __slots__ = ("_b",)

    def __init__(self, board: Board) -> None:
        self._b = board

    def positions(self) -> list[Pos]:
        return self._b.positions()

    def hidden_positions(self) -> list[Pos]:
        return self._b.hidden_positions()

    def visible_positions(self) -> list[Pos]:
        return self._b.visible_positions()

    def hidden_count(self) -> int:
        return self._b.hidden_count()

    def visible_score(self) -> int:
        return self._b.visible_score()

    def is_visible(self, pos: Pos) -> bool:
        return self._b.slot(pos).visible

    def value(self, pos: Pos) -> int:
        slot = self._b.slot(pos)
        if not slot.visible:
            raise PermissionError(f"La carte {pos} est cachée.")
        return slot.value

    def peek(self, pos: Pos) -> int | None:
        """Valeur si visible, None sinon."""
        slot = self._b.slot(pos)
        return slot.value if slot.visible else None

    def column_others(self, pos: Pos) -> list[int | None]:
        """Les deux autres cartes de la colonne de `pos` (None = cachée)."""
        col, row = pos
        column = self._b.column(col)
        if column is None:
            raise ValueError(f"La colonne {col} a été supprimée.")
        return [s.value if s.visible else None for r, s in enumerate(column) if r != row]

    def columns_left(self) -> list[int]:
        return sorted({c for c, _ in self._b.positions()})

    def __repr__(self) -> str:
        return repr(self._b)
