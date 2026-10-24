"""Paquet de cartes Skyjo, pioches et croyance probabiliste sur les cartes inconnues."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping

# Composition officielle : 150 cartes.
CARD_DISTRIBUTION: dict[int, int] = {-2: 5, -1: 10, 0: 15, **{v: 10 for v in range(1, 13)}}
CARD_VALUES: tuple[int, ...] = tuple(sorted(CARD_DISTRIBUTION))
DECK_SIZE = sum(CARD_DISTRIBUTION.values())
EXPECTED_CARD_VALUE = sum(v * n for v, n in CARD_DISTRIBUTION.items()) / DECK_SIZE  # ~5.07


def build_deck() -> list[int]:
    """Retourne les 150 cartes du jeu, non mélangées."""
    return [value for value, count in CARD_DISTRIBUTION.items() for _ in range(count)]


class CardBelief:
    """Distribution des cartes encore inconnues (pioche + cartes cachées de tous les plateaux).

    Une carte est "connue" si elle est visible sur un plateau, dans la défausse ou en main.
    La défausse est publique : un joueur attentif peut la mémoriser entièrement.
    """

    __slots__ = ("_counts", "_total", "_expected", "_variance")

    def __init__(self, counts: Mapping[int, int]) -> None:
        if any(c < 0 for c in counts.values()):
            raise ValueError(f"Comptage négatif : {dict(counts)}")
        self._counts = {v: c for v, c in counts.items() if c > 0}
        self._total = sum(self._counts.values())
        if self._total:
            self._expected = sum(v * c for v, c in self._counts.items()) / self._total
            second = sum(v * v * c for v, c in self._counts.items()) / self._total
            self._variance = max(second - self._expected**2, 0.0)
        else:
            self._expected, self._variance = EXPECTED_CARD_VALUE, 0.0

    def variance(self) -> float:
        return self._variance

    @property
    def total(self) -> int:
        return self._total

    def count(self, value: int) -> int:
        return self._counts.get(value, 0)

    def expected(self) -> float:
        return self._expected

    def prob(self, value: int) -> float:
        return self._counts.get(value, 0) / self._total if self._total else 0.0

    def items(self) -> Iterator[tuple[int, float]]:
        """(valeur, probabilité) pour chaque valeur encore possible."""
        for v, c in self._counts.items():
            yield v, c / self._total


NAIVE_BELIEF = CardBelief(CARD_DISTRIBUTION)


class Piles:
    """Pioche face cachée + défausse face visible."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.draw_pile: list[int] = build_deck()
        self._rng.shuffle(self.draw_pile)
        self.discard_pile: list[int] = []

    @property
    def top_discard(self) -> int | None:
        return self.discard_pile[-1] if self.discard_pile else None

    def draw(self) -> int:
        if not self.draw_pile:
            self._reshuffle()
        return self.draw_pile.pop()

    def take_discard(self) -> int:
        if not self.discard_pile:
            raise RuntimeError("La défausse est vide.")
        return self.discard_pile.pop()

    def discard(self, card: int) -> None:
        self.discard_pile.append(card)

    def _reshuffle(self) -> None:
        """Pioche vide : on remélange la défausse en gardant sa carte du dessus."""
        if len(self.discard_pile) <= 1:
            raise RuntimeError("Plus aucune carte disponible.")
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile
        self._rng.shuffle(self.draw_pile)
        self.discard_pile = [top]
