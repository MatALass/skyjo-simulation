"""Analyse du journal de décisions : quel coup est associé au meilleur score de manche, selon la situation.

Attention : ce sont des corrélations. Un coup choisi surtout par une bonne stratégie paraîtra bon
même s'il ne l'est pas en soi. Comparer les coups *à stratégie égale* (filtre --strategy) limite ce biais.
"""

from __future__ import annotations

from pathlib import Path


def _cell(mean_df, count_df, min_n: int):
    """Fusionne moyenne et effectif : '12.3 (456)', vide si effectif < min_n."""
    out = mean_df.copy().astype(object)
    for c in mean_df.columns:
        for i in mean_df.index:
            n = count_df.loc[i, c] if c in count_df.columns and i in count_df.index else 0
            m = mean_df.loc[i, c]
            out.loc[i, c] = f"{m:.1f} ({int(n)})" if n >= min_n and m == m else ""
    return out


def analyze(path: str | Path, strategy: str | None = None, min_n: int = 30) -> str:
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise SystemExit("L'analyse demande pandas : pip install -e '.[analysis]'") from e

    df = pd.read_csv(path)
    if strategy:
        df = df[df["strategy"] == strategy]
        if df.empty:
            raise SystemExit(f"Aucune décision pour la stratégie '{strategy}'.")

    df["phase"] = pd.cut(df["hidden_before"], bins=[-1, 3, 7, 12],
                         labels=["3_fin (<=3 cachées)", "2_milieu (4-7)", "1_début (>=8)"])
    df["menace"] = pd.cut(df["opp_min_hidden"], bins=[-1, 1, 3, 12],
                          labels=["adv. à 0-1 carte", "adv. à 2-3 cartes", "adv. loin (4+)"])
    df["position"] = pd.cut(df["lead_before"], bins=[-1e9, -5, 5, 1e9],
                            labels=["derrière (>5)", "au coude-à-coude", "devant (>5)"])

    out: list[str] = []
    title = f"Analyse de {Path(path).name}" + (f" — stratégie {strategy}" if strategy else " — toutes stratégies")
    out += [title, f"{len(df)} décisions, {df['game'].nunique()} parties", ""]
    out.append("Cellules : score moyen de manche du joueur (effectif). Plus bas = mieux. Vide si < "
               f"{min_n} décisions.\n")

    def pivot(data, index, columns, value="round_score"):
        mean = data.pivot_table(index=index, columns=columns, values=value, aggfunc="mean", observed=True)
        count = data.pivot_table(index=index, columns=columns, values=value, aggfunc="count", observed=True)
        return _cell(mean, count, min_n).to_string()

    draws = df[df["source"] == "draw"]
    out += ["1) Carte piochée → que faire ? (par phase de la manche)", ""]
    for phase, sub in sorted(draws.groupby("phase", observed=True), key=lambda x: str(x[0])):
        out += [f"   Phase {phase}", pivot(sub, "card", "action"), ""]

    out += ["2) Carte sur la défausse → la prendre ou piocher ?", ""]
    d = df[df["top_discard_before"].notna()].copy()
    d["choix"] = d["source"].map({"discard": "prend", "draw": "pioche"})
    d["top_discard_before"] = d["top_discard_before"].astype(int)
    out += [pivot(d, "top_discard_before", "choix"), ""]

    out += ["3) Selon la menace adverse (carte piochée, tous moments confondus)", ""]
    draws = draws.copy()
    draws["carte"] = pd.cut(draws["card"], bins=[-3, 2, 6, 12],
                            labels=["basse (-2..2)", "moyenne (3..6)", "haute (7..12)"])
    for threat, sub in draws.groupby("menace", observed=True):
        out += [f"   {threat}", pivot(sub, "carte", "action"), ""]

    out += ["4) Finir la manche : quand est-ce rentable ?", ""]
    fin = df[df["finished_round"]]
    if len(fin):
        g = fin.groupby("position", observed=True).agg(
            n=("round_score", "size"),
            doublé=("finisher_doubled", "mean"),
            manche_gagnée=("won_round", "mean"),
            score_moyen=("round_score", "mean"),
        )
        out += [g.round(3).to_string(), ""]

    out += ["5) Profil de jeu par stratégie (répartition des actions)", ""]
    mix = pd.crosstab(df["strategy"], df["source"] + "/" + df["action"], normalize="index")
    out += [(mix * 100).round(1).to_string(), ""]
    return "\n".join(out)
