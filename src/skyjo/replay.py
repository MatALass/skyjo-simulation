"""Replay : enregistre une partie complète, coup par coup, avec le raisonnement de chaque joueur.

Chaque étape contient l'état complet de la table (cartes cachées comprises, pour l'option "voir les cartes
cachées" du lecteur), ce qui vient de changer, une phrase de journal et les raisons chiffrées du choix.
Le résultat est exporté en JSON puis injecté dans une page HTML autonome (viewer/template.html).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from skyjo import __version__
from skyjo.board import Pos
from skyjo.game import Round, play_game
from skyjo.rules import DEFAULT_RULES, Rules
from skyjo.strategies import Strategy, make_strategy

LABELS = {
    "prudent": "Prudent",
    "fonceur": "Fonceur",
    "collectionneur": "Collectionneur",
    "opportuniste": "Opportuniste",
    "stratege": "Stratège",
    "expert": "Expert",
    "greedy": "Glouton",
    "random": "Hasard",
}


# ---------------------------------------------------------------------------------------------- mise en forme
def num(x: float, digits: int = 1) -> str:
    """Nombre à la française : 4,1 ; entiers sans décimale."""
    if abs(x - round(x)) < 1e-9:
        text = str(abs(int(round(x))))
    else:
        text = f"{abs(x):.{digits}f}".replace(".", ",")
    return ("−" if x < 0 and text.strip("0,") else "") + text


def signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + num(abs(x))


def cell(pos: Pos) -> str:
    c, r = pos
    return f"col. {c + 1}, ligne {r + 1}"


def card_name(v: int) -> str:
    return f"un {v}"


def target_text(parts: dict[str, Any]) -> str:
    """'le 12 (col. 3, ligne 2)' ou 'une carte cachée (col. 1, ligne 1)'."""
    where = cell(parts["pos"])
    if parts.get("hidden"):
        return f"une carte cachée ({where})"
    return f"le {parts['current']} ({where})"


def _profile_label(strategy: Strategy) -> str:
    base = strategy.name.split("[")[0]
    label = LABELS.get(base, base.capitalize())
    return label + (" (réglé)" if "[" in strategy.name else "")


def breakdown(parts: dict[str, Any], card: int) -> list[str]:
    """Décomposition lisible de la valeur d'un coup."""
    out: list[str] = []
    if parts.get("kind") == "place":
        cur = parts["cur_value"]
        if parts.get("completes"):
            out.append(f"colonne de trois {card} complétée : la colonne disparaît ({signed(parts['gain'])})")
        elif parts.get("hidden"):
            out.append(f"carte cachée estimée à {num(cur)} remplacée par {card_name(card)} ({signed(parts['gain'])})")
        else:
            out.append(f"{parts['current']} remplacé par {card_name(card)} ({signed(parts['gain'])})")
    if "pair" in parts:
        out.append(f"forme une paire dans la colonne ({signed(parts['pair'])})")
    if "break_pair" in parts:
        out.append(f"casse une paire ({signed(parts['break_pair'])})")
    if "column_chance" in parts:
        out.append(f"chance de compléter la paire de {parts['column_value']} ({signed(parts['column_chance'])})")
    if "tempo" in parts:
        out.append(f"bonus pour avancer en retournant une carte ({signed(parts['tempo'])})")
    if "deny" in parts:
        out.append(f"carte utile au joueur suivant ({signed(parts['deny'])})")
    if "finish" in parts:
        out.append(finish_text(parts["finish_detail"], parts["finish"]))
    return out


def finish_text(d: dict[str, Any], value: float) -> str:
    me, opp = d["my_final"], d["opp_best"]
    if d["policy"] == "always":
        return "ce coup termine sa grille : il finit sans regarder les scores adverses (aucune pénalité anticipée)"
    if d["policy"] == "probabilistic":
        return (f"ce coup termine sa grille : score estimé {num(me)} contre {num(opp)} pour le meilleur adversaire, "
                f"{num(100 * d['p_lowest'], 0)} % de chances d'être strictement le plus bas ({signed(value)})")
    if value < 0:
        return (f"ce coup terminerait sa grille avec ~{num(me)} contre ~{num(opp)} pour le meilleur adversaire ; "
                f"il exige {num(d['margin'])} points d'avance, donc risque de doublement ({signed(value)})")
    return f"ce coup termine sa grille avec une avance jugée suffisante (~{num(me)} contre ~{num(opp)})"


def ctx_text(ctx: dict[str, Any]) -> str:
    counted = f", comptage des {ctx['unknown']} cartes inconnues" if ctx["counting"] else ""
    place = ""
    if abs(ctx["hidden_place"] - ctx["hidden_value"]) > 1e-9:
        place = f" (il la compte {num(ctx['hidden_place'])} quand il envisage de la remplacer)"
    return (f"Il estime une carte cachée à {num(ctx['hidden_value'], 2)}{counted}{place}. "
            f"Son score final estimé : {num(ctx['my_est'])} ; meilleur adversaire : {num(ctx['opp_best'])}.")


def option_label(o: dict[str, Any], card: int) -> str:
    if o["kind"] == "reveal":
        return f"Défausser le {card}, retourner {cell(o['pos'])}"
    target = "cachée" if o["current"] is None else f"le {o['current']}"
    return f"Poser sur {target} ({cell(o['pos'])})"


# ---------------------------------------------------------------------------------------------- enregistreur
@dataclass
class Recorder:
    strategies: list[Strategy]
    labels: list[str]
    steps: list[dict[str, Any]] = field(default_factory=list)
    round_no: int = 0
    totals: list[int] = field(default_factory=list)
    round_scores: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.totals = [0] * len(self.strategies)

    # ------------------------------------------------------------------ état
    def _state(self, rnd: Round) -> dict[str, Any]:
        belief = rnd._belief()  # information publique : paquet - cartes visibles - défausse
        e = belief.expected()
        boards = []
        estimates = []
        for b in rnd.boards:
            boards.append([None if col is None else [[v, int(vis)] for v, vis in col] for col in b.cells()])
            estimates.append(round(b.visible_score() + b.hidden_count() * e, 1))
        return {
            "boards": boards,
            "draw": len(rnd.piles.draw_pile),
            "discard": rnd.piles.top_discard,
            "discardCount": len(rnd.piles.discard_pile),
            "expected": round(e, 2),
            "unknown": {str(v): belief.count(v) for v in range(-2, 13)},
            "visible": [b.visible_score() for b in rnd.boards],
            "hidden": [b.hidden_count() for b in rnd.boards],
            "estimates": estimates,
            "totals": list(self.totals),
        }

    def _add(self, rnd: Round, kind: str, title: str, details: list[str], player: int | None = None,
             hl: dict[str, Any] | None = None, options: list[dict[str, Any]] | None = None,
             turn: int | None = None) -> None:
        self.steps.append({
            "kind": kind, "round": self.round_no, "turn": turn, "player": player, "title": title,
            "details": details, "options": options or [], "hl": hl or {}, "state": self._state(rnd),
        })

    # ------------------------------------------------------------------ événements
    def __call__(self, event: str, rnd: Round, payload: dict[str, Any]) -> None:
        getattr(self, f"_on_{event}")(rnd, payload)

    def _on_deal(self, rnd: Round, _: dict[str, Any]) -> None:
        self.round_no += 1
        n = len(self.strategies)
        details = [f"Chaque joueur reçoit 12 cartes face cachée, en 3 lignes de 4 colonnes ({12 * n} cartes).",
                   f"Une carte est retournée pour lancer la défausse : {rnd.piles.top_discard}.",
                   f"La pioche contient {len(rnd.piles.draw_pile)} cartes."]
        if self.round_no == 1:
            details.append("Objectif : avoir le plus petit total. La partie s'arrête quand quelqu'un atteint 100.")
        self._add(rnd, "deal", f"Manche {self.round_no} : distribution", details)

    def _on_initial(self, rnd: Round, p: dict[str, Any]) -> None:
        i = p["player"]
        who = self.labels[i]
        board = rnd.boards[i]
        vals = [board.slot(pos).value for pos in p["positions"]]
        a, b = p["positions"]
        exp = p.get("explanation") or {}
        details: list[str] = []
        if exp.get("same_column"):
            details.append(f"Le {who} retourne deux cartes de la même colonne : il vise une colonne identique "
                           "(trois cartes égales disparaissent) dès le départ.")
        elif exp.get("step") == "initial":
            details.append("Deux cartes prises dans deux colonnes différentes, au hasard : tant qu'aucune carte "
                           "n'est connue, toutes les positions se valent ; changer de colonne garde plus "
                           "d'options pour former des paires.")
        else:
            details.append("Deux cartes au hasard.")
        if a[0] == b[0] and vals[0] == vals[1]:
            details.append(f"Coup de chance : une paire de {vals[0]} dans la colonne {a[0] + 1}.")
        details.append(f"Total visible : {sum(vals)}. Le plus gros total visible commencera la manche 1.")
        self._add(rnd, "initial", f"{who} retourne {vals[0]} ({cell(a)}) et {vals[1]} ({cell(b)})", details,
                  player=i, hl={"player": i, "cells": [[*a, "revealed"], [*b, "revealed"]]})

    def _on_start(self, rnd: Round, p: dict[str, Any]) -> None:
        first = p["first"]
        who = self.labels[first]
        totals = ", ".join(f"{self.labels[i]} {t}" for i, t in enumerate(p["totals"]))
        if p["forced"]:
            title = f"{who} commence : il a terminé la manche précédente"
            details = ["Règle : le joueur qui a fini la manche précédente joue en premier.",
                       f"Totaux visibles après le retournement : {totals}."]
        else:
            title = f"{who} commence : plus gros total visible ({p['totals'][first]})"
            details = [f"Totaux visibles : {totals}.", "En cas d'égalité, le premier joueur est tiré au sort."]
        self._add(rnd, "start", title, details, player=first, hl={"player": first})

    def _on_turn(self, rnd: Round, p: dict[str, Any]) -> None:
        i, info = p["player"], p["info"]
        who = self.labels[i]
        card, pos = info["card"], tuple(info["pos"])
        old = info["target_value"]
        if info["source"] == "discard":
            target = "une carte cachée" if not info["target_was_visible"] else f"le {old}"
            title = f"{who} prend le {card} de la défausse et le pose sur {target} ({cell(pos)})"
            if not info["target_was_visible"]:
                title += f" : c'était {card_name(old)}"
        elif info["action"] == "reveal":
            title = f"{who} pioche {card_name(card)}, le défausse et retourne {cell(pos)} : {card_name(old)}"
        else:
            if info["target_was_visible"]:
                title = f"{who} pioche {card_name(card)} et le pose à la place du {old} ({cell(pos)})"
            else:
                title = (f"{who} pioche {card_name(card)} et le pose sur une carte cachée ({cell(pos)}) : "
                         f"c'était {card_name(old)}")
        if p["is_last_turn"]:
            title = "Dernier tour · " + title

        details: list[str] = []
        options: list[dict[str, Any]] = []
        first, second = p["explanations"]
        if first:
            details += self._explain(first)[0]
        if second:
            d, options = self._explain(second)
            if first and first.get("step") == "discard_choice" and first.get("take"):
                d = d[1 + len(breakdown(second["chosen"], second["card"])):]  # déjà détaillé ci-dessus
            details += d
        ctx = next((e["ctx"] for e in p["explanations"] if e and "ctx" in e), None)
        if ctx:
            details.append(ctx_text(ctx))

        removed_cols = []
        if info["removed"]:
            v = info["removed"][0]
            removed_cols = [pos[0]]
            title += f". Colonne de trois {v} : retirée !"
            details.append(f"Trois {v} visibles dans la même colonne : la colonne quitte le jeu "
                           f"({signed(-3 * v)} sur son score).")
        if p["finished_round"]:
            title += f". {who} a retourné toute sa grille : dernier tour pour les autres"
            details.append("Chaque autre joueur joue encore une fois. Si le finisseur n'a pas strictement le plus "
                           "petit score, son score (s'il est positif) est doublé.")

        self._add(rnd, "turn", title, details, player=i, turn=p["turn"], options=options, hl={
            "player": i, "cells": [[*pos, "new" if info["action"] != "reveal" else "revealed"]],
            "source": info["source"], "card": card, "old": old, "action": info["action"],
            "removedCols": removed_cols,
        })

    def _explain(self, exp: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        step = exp.get("step")
        if step == "random":
            return ["Il joue au hasard."], []
        prm = exp.get("params", {})
        if step == "discard_choice":
            card = exp["card"]
            if exp["blocked"]:
                return [f"Défausse : {card}. Il ne pose jamais plus de {prm['hidden_max']} sur une carte cachée et "
                        "il n'a que des cartes cachées : il pioche."], []
            best = exp["best"]
            lines = [f"Défausse : {card}. Meilleur usage pour lui : {target_text(best)}, "
                     f"valeur {num(best['total'])}."]
            lines += [f"• {t}" for t in breakdown(best, card)]
            if exp["draw_ev"] is not None:
                verdict = "il la prend" if exp["take"] else "il préfère piocher"
                lines.append(f"Valeur moyenne d'une pioche (calculée carte par carte) : {num(exp['draw_ev'])} : "
                             f"{verdict}.")
            else:
                thr = prm["discard_min_gain"]
                verdict = "il la prend" if exp["take"] else "il préfère piocher"
                lines.append(f"Il ne prend la défausse que si elle rapporte au moins {num(thr)} : {verdict}.")
            return lines, []
        if step == "place_discard":
            card, ch = exp["card"], exp["chosen"]
            lines = [f"Il pose le {card} sur {target_text(ch)} (valeur {num(ch['total'])}) :"]
            lines += [f"• {t}" for t in breakdown(ch, card)]
            return lines, self._options(exp["options"], card, ch)
        if step == "drawn":
            card = exp["card"]
            lines = [f"Il a pioché {card_name(card)}."]
            pl, rv = exp["place"], exp["reveal"]
            chosen = pl if exp["replace"] else rv
            if pl is not None and not exp["forced"]:
                lines.append(f"Meilleur placement : {target_text(pl)}, valeur {num(pl['total'])}.")
                lines += [f"• {t}" for t in breakdown(pl, card)]
            elif prm.get("hidden_max") is not None and card > prm["hidden_max"]:
                lines.append(f"Il ne pose jamais plus de {prm['hidden_max']} sur une carte cachée.")
            if rv is not None:
                lines.append(f"Défausser et retourner {cell(rv['pos'])} : valeur {num(rv['total'])}.")
                lines += [f"• {t}" for t in breakdown(rv, card)]
            if exp["forced"]:
                lines.append("Plus de carte cachée à retourner : il doit poser la carte.")
            elif pl is not None and rv is not None:
                a, b = (pl, rv) if exp["replace"] else (rv, pl)
                action = "il pose la carte" if exp["replace"] else "il défausse et retourne"
                lines.append(f"Conclusion : {action} ({num(a['total'])} contre {num(b['total'])}).")
            opts = self._options(exp["options"], card, chosen)
            if any(o["chosen"] and o["count"] > 1 for o in opts):
                lines.append("Plusieurs cases donnent la même valeur : la case est tirée au hasard parmi elles.")
            return lines, opts
        return [], []

    @staticmethod
    def _options(opts: Sequence[dict[str, Any]], card: int, chosen: dict[str, Any] | None,
                 limit: int = 5) -> list[dict[str, Any]]:
        """Options chiffrées, en regroupant les coups équivalents (même type, même carte visée, même valeur)."""
        chosen_key = None
        if chosen is not None:
            chosen_key = (chosen["kind"], chosen.get("current") if not chosen.get("hidden") else None,
                          round(chosen["total"], 2))
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for o in opts:
            key = (o["kind"], o["current"], round(o["value"], 2))
            g = groups.setdefault(key, {"option": o, "count": 0})
            g["count"] += 1
        out = []
        for key, g in groups.items():
            o, n = g["option"], g["count"]
            if n == 1:
                label = option_label(o, card)
            elif o["kind"] == "reveal":
                label = f"Défausser le {card}, retourner une carte cachée"
            elif o["current"] is None:
                label = "Poser sur une carte cachée"
            else:
                label = f"Poser sur un {o['current']}"
            out.append({"label": label, "value": round(o["value"], 2), "count": n, "chosen": key == chosen_key})
        top = out[:limit]
        if chosen_key is not None and not any(o["chosen"] for o in top):
            top += [o for o in out if o["chosen"]]
        return top

    def _on_round_end(self, rnd: Round, p: dict[str, Any]) -> None:
        res = p["result"]
        raw, scores = res.raw_scores, res.scores
        details = ["Toutes les cartes encore cachées sont retournées."]
        for i, cols in enumerate(p["final_removed"]):
            for c in cols:
                details.append(f"{self.labels[i]} : colonne {c + 1} identique, retirée au retournement final.")
        for i in range(len(raw)):
            details.append(f"{self.labels[i]} : {raw[i]} points" + (f", doublé à {scores[i]}"
                                                                     if scores[i] != raw[i] else "") + ".")
        if res.stalled:
            details.append("Manche arrêtée (trop longue) : pas de doublement.")
        elif res.finisher is not None:
            f = res.finisher
            others = min(s for j, s in enumerate(raw) if j != f)
            who = self.labels[f]
            if res.finisher_doubled:
                details.append(f"{who} a terminé la manche mais n'a pas strictement le plus petit score "
                               f"({raw[f]} contre {others}) : son score est doublé.")
            elif raw[f] <= 0:
                details.append(f"{who} a terminé avec un score de {raw[f]} : un score nul ou négatif n'est "
                               "jamais doublé.")
            else:
                details.append(f"{who} a terminé avec strictement le plus petit score : pas de pénalité.")
        self.round_scores.append(list(scores))
        self.totals = [t + s for t, s in zip(self.totals, scores, strict=True)]
        details.append("Totaux : " + ", ".join(f"{self.labels[i]} {t}" for i, t in enumerate(self.totals)) + ".")
        hl_cells = [[i, [[c, r, "final"] for c, r in cells]] for i, cells in enumerate(p["final_revealed"])]
        self._add(rnd, "round_end", f"Fin de la manche {self.round_no}", details, hl={
            "finalCells": hl_cells, "finisher": res.finisher, "doubled": res.finisher_doubled,
            "roundScores": scores, "rawScores": raw,
        })

    def _on_game_end(self, rnd: Round, p: dict[str, Any]) -> None:
        res = p["result"]
        winners = " et ".join(self.labels[w] for w in res.winners)
        details = [f"{self.labels[i]} : {t} points" for i, t in
                   sorted(enumerate(res.totals), key=lambda x: x[1])]
        details.insert(0, f"Un joueur a atteint {max(res.totals)} points (≥ 100) : la partie s'arrête "
                          "et le plus petit total gagne.")
        self._add(rnd, "game_end", f"Victoire : {winners}", details, hl={"winners": res.winners})


# ---------------------------------------------------------------------------------------------- API
def _labels(strategies: Sequence[Strategy]) -> list[str]:
    base = [_profile_label(s) for s in strategies]
    out = []
    for i, label in enumerate(base):
        out.append(f"{label} {base[:i + 1].count(label)}" if base.count(label) > 1 else label)
    return out


def record_game(specs: Sequence[str], seed: int | None = None, rules: Rules = DEFAULT_RULES) -> dict[str, Any]:
    strategies = [make_strategy(s) for s in specs]
    for s in strategies:
        s.explain = True
    labels = _labels(strategies)
    rec = Recorder(strategies, labels)
    rng = random.Random(seed)
    play_game(strategies, rng, rules, observer=rec)
    return {
        "meta": {
            "version": __version__,
            "seed": seed,
            "players": [{"label": labels[i], "spec": specs[i], "description": s.description}
                        for i, s in enumerate(strategies)],
            "rounds": rec.round_no,
            "roundScores": rec.round_scores,
        },
        "steps": rec.steps,
    }


def _template() -> str:
    return resources.files("skyjo.viewer").joinpath("template.html").read_text(encoding="utf-8")


def render_html(replay: dict[str, Any], standalone: bool = True) -> str:
    """Page du lecteur avec la partie embarquée. `standalone` ajoute le squelette HTML complet."""
    data = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = _template().replace("/*__REPLAY_DATA__*/null", data)
    if not standalone:
        return page
    return ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")


def write_replay(specs: Sequence[str], out: str | Path, seed: int | None = None, rules: Rules = DEFAULT_RULES,
                 standalone: bool = True, json_out: str | Path | None = None) -> dict[str, Any]:
    replay = record_game(specs, seed, rules)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(replay, standalone), encoding="utf-8")
    if json_out:
        Path(json_out).write_text(json.dumps(replay, ensure_ascii=False, indent=1), encoding="utf-8")
    return replay
