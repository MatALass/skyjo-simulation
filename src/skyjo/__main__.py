"""CLI.

python -m skyjo list
python -m skyjo match --strategies prudent,fonceur,collectionneur,opportuniste --games 5000 --seed 1
python -m skyjo tournament --pool prudent,fonceur,collectionneur,opportuniste,stratege --players 2 3 4 --games 2000
python -m skyjo sweep --base stratege --param finish_bonus --values 0 3 6 \
    --opponents prudent,collectionneur,opportuniste
python -m skyjo analyze results/decisions.csv
python -m skyjo optimize --base stratege --games 2000
python -m skyjo replay --strategies expert,prudent,fonceur,collectionneur --seed 19
python -m skyjo coach --me "5 ? ? 3 / ? 12 ? ? / 0 ? ? ?" --opp "? 4 ? ? / ? ? ? 1 / ? ? 2 ?" --discard 7 --drawn 9
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from skyjo.progress import Progress
from skyjo.report import format_table, write_csv
from skyjo.rules import Rules
from skyjo.simulate import run_match
from skyjo.strategies import LAMBDA_PLAYERS, available
from skyjo.sweep import run_sweep
from skyjo.tournament import run_tournament, tables_for

DEFAULT_POOL = ("prudent", "collectionneur", "opportuniste", "stratege", "expert")


def _rules(args: argparse.Namespace) -> Rules:
    return Rules(columns_on_reveal=not args.no_columns_on_reveal,
                 columns_at_final_reveal=not args.no_columns_at_final_reveal)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--games", type=int, default=2000, help="Parties par table.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Processus parallèles.")
    p.add_argument("--out", type=Path, default=Path("results"), help="Dossier de sortie des CSV.")
    p.add_argument("--quiet", action="store_true", help="Pas de barre de progression.")
    p.add_argument("--no-columns-on-reveal", action="store_true",
                   help="Variante : colonne retirée seulement si complétée par un échange.")
    p.add_argument("--no-columns-at-final-reveal", action="store_true",
                   help="Variante : pas de colonne retirée au retournement final.")


def _progress(args: argparse.Namespace, total: int, label: str, groups: int = 0,
              groups_unit: str = "tables") -> Progress | None:
    if args.quiet:
        return None
    return Progress(total, label=label, groups=groups, groups_unit=groups_unit)


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def cmd_list(_: argparse.Namespace) -> None:
    for name, desc in available().items():
        tag = " (joueur lambda)" if name in LAMBDA_PLAYERS else ""
        print(f"{name}{tag}\n    {desc}\n")


def cmd_match(args: argparse.Namespace) -> None:
    specs = [s.strip() for s in args.strategies.split(",")]
    t0 = time.perf_counter()
    pr = _progress(args, args.games, f"{len(specs)} joueurs")
    with pr or _Null():
        res = run_match(specs, args.games, args.seed, args.jobs, args.log_games, _rules(args), pr)
    rows = res.summaries()
    print(format_table(rows, f"{len(specs)} joueurs — {args.games} parties ({time.perf_counter() - t0:.0f}s)"))
    print(f"\nManches bloquées : {res.stalled_rounds}/{res.total_rounds}")
    print(f"→ {write_csv(rows, args.out / 'match_summary.csv')}")
    if res.decisions:
        print(f"→ {write_csv(res.decisions, args.out / 'decisions.csv')} ({len(res.decisions)} décisions)")


def cmd_tournament(args: argparse.Namespace) -> None:
    pool = [s.strip() for s in args.pool.split(",")]
    t0 = time.perf_counter()
    n_tables = sum(len(tables_for(pool, n)) for n in args.players)
    print(f"Tournoi : {n_tables} tables x {args.games} parties sur {args.jobs} processus", flush=True)
    pr = _progress(args, n_tables * args.games, "tournoi", groups=n_tables)
    with pr or _Null():
        res = run_tournament(pool, args.players, args.games, args.seed, args.jobs, _rules(args), pr)
    rows = res.ranking()
    for n in args.players:
        print(format_table([r for r in rows if r["n_players"] == n], f"\n=== {n} joueurs ==="))
    stalled = sum(m.stalled_rounds for m in res.matches)
    total = sum(m.total_rounds for m in res.matches)
    print(f"\n{len(res.matches)} tables, {time.perf_counter() - t0:.0f}s, manches bloquées : {stalled}/{total}")
    print(f"→ {write_csv(rows, args.out / 'tournament_ranking.csv')}")
    print(f"→ {write_csv(res.match_rows(), args.out / 'tournament_matches.csv')}")


def cmd_sweep(args: argparse.Namespace) -> None:
    opponents = [s.strip() for s in args.opponents.split(",")]
    t0 = time.perf_counter()
    pr = _progress(args, len(args.values) * args.games, "sweep", groups=len(args.values), groups_unit="valeurs")
    with pr or _Null():
        rows = run_sweep(args.base, args.param, args.values, opponents, args.games, args.seed, args.jobs,
                         _rules(args), pr)
    title = (f"{args.base} : balayage de {args.param} contre {', '.join(opponents)} "
             f"— {args.games} parties par valeur ({time.perf_counter() - t0:.0f}s)")
    print(format_table(rows, title))
    base = re.sub(r"[^\w.=-]+", "_", args.base)  # ':' interdit dans un nom de fichier Windows
    print(f"→ {write_csv(rows, args.out / f'sweep_{base}_{args.param}.csv')}")


def cmd_optimize(args: argparse.Namespace) -> None:
    from skyjo.optimize import DEFAULT_FIELDS, Optimizer, fields_for_players

    if args.fields:
        fields = [[x.strip() for x in f.split(",")] for f in args.fields]
    elif args.players:
        fields = fields_for_players(args.players)
    else:
        fields = DEFAULT_FIELDS
    grid = {}
    for item in args.grid or []:
        k, v = item.split("=", 1)
        grid[k] = [x.strip() for x in v.split(",")]

    def factory(total: int, label: str, groups: int) -> Progress | None:
        return None if args.quiet else Progress(total, label=label, groups=groups, groups_unit="évals")

    print(f"Optimisation de {args.base} sur {len(fields)} terrains : " + " | ".join(",".join(f) for f in fields))
    opt = Optimizer(args.base, fields, args.games, args.seed if args.seed is not None else 1, args.jobs,
                    _rules(args), factory)
    t0 = time.perf_counter()
    res = opt.run(params=args.params, grid=grid, max_passes=args.passes, min_gain=args.min_gain)

    print(f"\nTerminé en {time.perf_counter() - t0:.0f}s.")
    print("Changements retenus :" if res.steps else "Aucun changement retenu.")
    for step in res.steps:
        print(f"  {step}")
    print(f"\nMeilleur profil : {res.best_spec}")
    print("\nValidation sur une graine neuve (indice de victoire, 1.00 = hasard) :")
    for ev in res.validation:
        detail = "  ".join(f"{'+'.join(f)}={r['win_index']:.2f}" for f, r in zip(fields, ev.per_field, strict=True))
        print(f"  {ev.score:.3f}  {ev.spec}\n         {detail}")
    base = re.sub(r"[^\w.=-]+", "_", args.base)
    print(f"→ {write_csv(res.history, args.out / f'optimize_{base}.csv')}")
    (args.out / f"optimize_{base}_best.txt").write_text(res.best_spec + "\n", encoding="utf-8")


def cmd_coach(args: argparse.Namespace) -> None:
    from skyjo.coach import Situation, coach, format_report, parse_grid

    situation = Situation(
        me=parse_grid(args.me),
        opponents=tuple(parse_grid(g) for g in args.opp),
        top_discard=args.discard,
        drawn=args.drawn,
        seen=tuple(int(x) for x in (args.seen or "").split()),
    )
    pr = None if args.quiet else Progress(args.samples, label="coach", unit="tirages")
    with pr or _Null():
        results = coach(situation, args.samples, seed=args.seed or 0, policy=args.policy,
                        opp_policy=args.opp_policy, jobs=args.jobs, rules=_rules(args), progress=pr)
    report = format_report(situation, results)
    print(report)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")
        print(f"\n→ {args.save}")


def cmd_replay(args: argparse.Namespace) -> None:
    from skyjo.replay import write_replay

    specs = [s.strip() for s in args.strategies.split(",")]
    out = args.out
    replay = write_replay(specs, out, seed=args.seed, rules=_rules(args), json_out=args.json)
    meta = replay["meta"]
    print(f"Partie enregistrée : {len(replay['steps'])} étapes, {meta['rounds']} manches, graine {meta['seed']}.")
    totals = [0] * len(specs)
    for scores in meta["roundScores"]:
        totals = [t + s for t, s in zip(totals, scores, strict=True)]
    print("Totaux : " + ", ".join(f"{p['label']} {t}" for p, t in zip(meta["players"], totals, strict=True)))
    print(f"→ {out}  (ouvrez ce fichier dans votre navigateur)")


def cmd_analyze(args: argparse.Namespace) -> None:
    from skyjo.analysis import analyze

    text = analyze(args.decisions, args.strategy, args.min_n)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\n→ {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="skyjo", description="Simulateur de parties de Skyjo.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Lister les stratégies.").set_defaults(func=cmd_list)

    p = sub.add_parser("match", help="Une table fixe de stratégies.")
    p.add_argument("--strategies", default=",".join(LAMBDA_PLAYERS),
                   help="Une stratégie par joueur, ex : prudent,fonceur,stratege:finish_margin=0")
    p.add_argument("--log-games", type=int, default=0,
                   help="Enregistrer chaque décision des N premières parties (decisions.csv).")
    _common(p)
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("tournament", help="Toutes les tables possibles d'un pool, par nombre de joueurs.")
    p.add_argument("--pool", default=",".join(DEFAULT_POOL),
                   help="Stratégies du tournoi (ajouter fonceur pour l'inclure).")
    p.add_argument("--players", type=int, nargs="+", default=[2, 3, 4])
    _common(p)
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("sweep", help="Faire varier un paramètre d'une stratégie face à des adversaires fixes.")
    p.add_argument("--base", required=True, help="Stratégie de base, ex : prudent")
    p.add_argument("--param", required=True, help="Paramètre du profil, ex : hidden_max")
    p.add_argument("--values", nargs="+", required=True, help="Valeurs testées, ex : 2 3 4 5")
    p.add_argument("--opponents", required=True, help="Adversaires, ex : fonceur,collectionneur,opportuniste")
    _common(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("optimize", help="Optimiser les paramètres d'un profil (descente de coordonnées).")
    p.add_argument("--base", default="stratege", help="Profil de départ, ex : stratege ou prudent:reveal_bonus=0")
    p.add_argument("--fields", nargs="+", default=None,
                   help="Terrains d'évaluation, une liste d'adversaires par terrain "
                        "(ex : prudent,collectionneur,opportuniste prudent)")
    p.add_argument("--players", type=int, nargs="+", default=None,
                   help="Optimiser pour ce(s) nombre(s) de joueurs : terrains = toutes les tables de joueurs "
                        "lambda de cette taille (ignoré si --fields est donné).")
    p.add_argument("--params", nargs="+", default=None, help="Paramètres à optimiser (défaut : tous les pertinents).")
    p.add_argument("--grid", nargs="+", default=None, help="Valeurs à tester, ex : hidden_max=2,3,4 pair_bonus=0,1")
    p.add_argument("--passes", type=int, default=3, help="Nombre maximal de passes.")
    p.add_argument("--min-gain", type=float, default=0.03, help="Gain minimal d'indice pour changer une valeur.")
    _common(p)
    p.set_defaults(func=cmd_optimize, games=2000)

    p = sub.add_parser("coach", help="Meilleur coup dans une situation donnée (simulation de la fin de manche).")
    p.add_argument("--me", required=True,
                   help="Mon plateau : 3 lignes séparées par '/', '?' = cachée, 'x' = colonne retirée")
    p.add_argument("--opp", required=True, action="append",
                   help="Plateau d'un adversaire (répéter --opp, dans l'ordre de jeu après moi)")
    p.add_argument("--discard", type=int, default=None, help="Carte du dessus de la défausse.")
    p.add_argument("--drawn", type=int, default=None, help="Carte que je viens de piocher (sinon : début de tour).")
    p.add_argument("--seen", default=None, help="Autres cartes connues hors jeu, ex : \"12 12 5\"")
    p.add_argument("--samples", type=int, default=400, help="Nombre de tirages simulés.")
    p.add_argument("--policy", default="expert", help="Stratégie supposée pour moi ensuite.")
    p.add_argument("--opp-policy", default=None, help="Stratégie supposée des adversaires (défaut : --policy).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--save", type=Path, default=None, help="Enregistrer le rapport dans un fichier.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-columns-on-reveal", action="store_true")
    p.add_argument("--no-columns-at-final-reveal", action="store_true")
    p.set_defaults(func=cmd_coach)

    p = sub.add_parser("replay", help="Enregistrer une partie et la rejouer pas à pas dans le navigateur.")
    p.add_argument("--strategies", default="expert,prudent,fonceur,collectionneur",
                   help="Une stratégie par joueur (2 à 8), dans l'ordre des sièges.")
    p.add_argument("--seed", type=int, default=None, help="Graine (même graine = même partie).")
    p.add_argument("--out", type=Path, default=Path("results/replay.html"), help="Page HTML produite.")
    p.add_argument("--json", type=Path, default=None, help="Exporter aussi les données brutes en JSON.")
    p.add_argument("--no-columns-on-reveal", action="store_true")
    p.add_argument("--no-columns-at-final-reveal", action="store_true")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("analyze", help="Analyser un journal de décisions.")
    p.add_argument("decisions", type=Path)
    p.add_argument("--strategy", default=None, help="Restreindre à une stratégie.")
    p.add_argument("--min-n", type=int, default=30, help="Effectif minimal affiché par cellule.")
    p.add_argument("--out", type=Path, default=None, help="Fichier texte de sortie.")
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
