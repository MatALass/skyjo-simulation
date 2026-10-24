"""Personnalités de joueurs. Les 4 premières imitent des joueurs "lambda" ; le stratège sert de référence."""

from __future__ import annotations

from skyjo.strategies.heuristic import Profile

HIDDEN_DISCOUNT = 0.0  # 2.0 testé : moins bon (sweep, 300 parties, 4 joueurs)

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile(
            name="prudent",
            description="Le Prudent : ne pose que des cartes ≤ 2 sur ses cartes cachées, prend rarement la "
            "défausse, préfère échanger que retourner, ne finit que s'il est nettement devant.",
            hidden_max=2,
            discard_min_gain=4.0,
            reveal_bonus=-1.0,
            finish_policy="if_lowest",
            finish_margin=5.0,
        ),
        Profile(
            name="fonceur",
            description="Le Fonceur : veut finir vite. Retourne dès qu'il peut, accepte des cartes moyennes "
            "(≤ 6) sur ses cachées, termine la manche sans regarder les scores adverses.",
            hidden_max=6,
            discard_min_gain=2.0,
            reveal_bonus=1.5,
            finish_policy="always",
        ),
        Profile(
            name="collectionneur",
            description="Le Collectionneur : obsédé par les colonnes. Retourne 2 cartes de la même colonne, "
            "valorise fortement les paires, ne finit que s'il pense être devant.",
            hidden_max=5,
            discard_min_gain=2.0,
            pair_bonus=4.0,
            finish_policy="if_lowest",
            initial_same_column=True,
        ),
        Profile(
            name="opportuniste",
            description="L'Opportuniste : regarde les adversaires. Accélère quand il mène, ralentit quand il est "
            "derrière, évite de défausser ce qui arrange le joueur suivant.",
            hidden_max=4,
            discard_min_gain=3.0,
            tempo_weight=3.0,
            deny_weight=0.5,
            finish_policy="if_lowest",
        ),
        Profile(
            name="stratege",
            description="Le Stratège (référence) : compte les cartes, compare la défausse à l'espérance exacte "
            "de la pioche, évite de servir le joueur suivant, ne finit que si "
            "la probabilité d'être le plus bas le justifie.",
            card_counting=True,
            draw_lookahead=True,
            hidden_max=None,
            hidden_discount=HIDDEN_DISCOUNT,
            pair_bonus=1.0,
            tempo_weight=0.0,  # 3.0 à l'origine : 29 % -> 38 % de victoires à 0 (sweep, 3000 parties)
            deny_weight=0.5,
            finish_policy="probabilistic",
            finish_bonus=3.0,
        ),
        Profile(
            name="expert",
            description="L'Expert (issu de skyjo optimize, puis complété) : carte cachée estimée à ~4, défausse "
            "seulement si elle rapporte >= 5, vise fortement les paires, retourne volontiers en gardant une "
            "inconnue par colonne, ne finit qu'avec 10 points d'avance estimée. Ne compte pas les cartes.",
            hidden_max=None,
            hidden_discount=1.0,      # carte cachée ~ 5.07 - 1 : on pose jusqu'à ~4 dessus
            discard_min_gain=5.0,
            pair_bonus=4.0,
            reveal_bonus=1.0,
            finish_policy="if_lowest",
            finish_margin=10.0,
            # v0.5 : 45,2 % -> 53,1 % de victoires à 4 joueurs (2 400 parties, graine neuve) ; contrôle
            # reveal_bonus=3 sans répartition : 46,2 % -> le gain vient bien de la répartition.
            reveal_spread=4.0,
        ),
        Profile(
            name="greedy",
            description="Glouton simple (ancienne version) : gain immédiat, carte ≤ 4 sur une cachée.",
            hidden_max=4,
            discard_min_gain=3.0,
        ),
    )
}

LAMBDA_PLAYERS: tuple[str, ...] = ("prudent", "fonceur", "collectionneur", "opportuniste")
