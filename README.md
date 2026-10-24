# skyjo-sim

Simulateur Monte Carlo de Skyjo : comparer des styles de jeu selon le nombre de joueurs (2 à 8)
et comprendre, par les probabilités, quels coups sont les meilleurs dans chaque situation.

```bash
pip install -e ".[dev]"
pytest
skyjo list                       # stratégies disponibles
```

## Commandes

| Commande | Question à laquelle elle répond |
|---|---|
| `skyjo tournament --players 2 3 4 --games 2000` | Quel style gagne le plus, à 2, 3, 4 joueurs ? (toutes les tables possibles du pool) |
| `skyjo match --strategies prudent,fonceur,collectionneur,opportuniste --games 5000 --log-games 1000` | Une table précise + journal de chaque décision |
| `skyjo analyze results/decisions.csv [--strategy stratege]` | Quel coup est associé au meilleur score selon la carte, la phase, la menace adverse |
| `skyjo sweep --base prudent --param hidden_max --values 1 2 3 4 --opponents fonceur,collectionneur,opportuniste` | Quel réglage d'un paramètre est le meilleur |
| `skyjo optimize --base stratege --games 2000` | Meilleur réglage de *tous* les paramètres d'un profil, validé sur une graine neuve |
| `skyjo replay --strategies expert,prudent,fonceur,collectionneur --seed 19` | Voir une partie tour par tour dans le navigateur, avec la raison de chaque coup |
| `skyjo coach --me "5 ? ? 3 / ? 12 ? ? / 0 ? ? ?" --opp "? 4 ? ? / ? ? ? 1 / ? ? 2 ?" --discard 7 --drawn 9` | Meilleur coup dans une situation précise |

Options communes : `--seed`, `--jobs` (processus parallèles), `--out` (dossier des CSV),
`--no-columns-on-reveal`, `--no-columns-at-final-reveal` (variantes de règles).

## Replay

`skyjo replay` joue une partie complète et écrit `results/replay.html`, une page autonome à ouvrir dans le
navigateur (aucun serveur). Boutons ou flèches ← → pour avancer, espace pour la lecture automatique,
« Voir les cartes cachées » pour suivre ce que les joueurs ignorent. Pour chaque coup : la phrase du journal,
la décomposition chiffrée de la décision (gain, paire, bonus, risque de doublement), les options comparées,
et ce que la table sait (cartes encore inconnues, espérance d'une carte cachée).
Les explications sont calculées sans toucher aux décisions : une partie rejouée est identique à une partie
simulée normalement avec la même graine (testé).

## Coach

Plateaux : 3 lignes séparées par `/`, 4 cartes par ligne. `?` = carte cachée, `x` = colonne retirée.
Un `--opp` par adversaire, dans l'ordre de jeu après vous. Sans `--drawn` : décision de début de tour
(prendre la défausse et où la poser, ou piocher). Avec `--drawn` : où poser la carte piochée, ou laquelle retourner.
Un adversaire sans aucun `?` est considéré comme ayant fini : c'est votre dernier tour.

Pour chaque tirage des cartes inconnues, chaque coup est joué puis la fin de manche est simulée
(`--policy`, par défaut `expert`). Les coups sont comparés sur les mêmes tirages : l'écart mesure l'effet
du coup. Les coups nettement moins bons sont écartés en cours de route.

## Optimisation

Descente de coordonnées : chaque paramètre est balayé à tour de rôle, la meilleure valeur est gardée si elle
améliore l'objectif d'au moins `--min-gain`, jusqu'à ce qu'une passe ne change plus rien.
Objectif = indice de victoire moyen sur plusieurs terrains (`--fields`, par défaut : une table à 4 sérieuse,
une table à 4 avec un fonceur, un duel contre le prudent ; `--players 3` = toutes les tables à 3 de joueurs lambda).
Le profil final est revalidé sur une graine neuve :
si le gain disparaît, c'était du bruit.

## Personnalités

| Nom | Style |
|---|---|
| `prudent` | Seulement des cartes ≤ 2 sur ses cachées, prend rarement la défausse, ne finit que s'il est nettement devant |
| `fonceur` | Retourne vite, accepte des cartes ≤ 6, finit sans regarder les adversaires |
| `collectionneur` | Vise les colonnes, valorise les paires |
| `opportuniste` | Accélère s'il mène, ralentit s'il est derrière, évite de servir le joueur suivant |
| `stratege` | Référence : comptage des cartes, espérance exacte de la pioche, fin de manche probabiliste (sans lecture du rythme depuis la v0.3) |
| `expert` | Résultat de `skyjo optimize` : carte cachée estimée à ~4, défausse seulement si gain ≥ 5, paires très valorisées, finit seulement avec 10 pts d'avance estimée. Ne compte pas les cartes |
| `greedy`, `random` | Références basses |

Chaque personnalité est un `Profile` (`strategies/personalities.py`). Tout paramètre se surcharge en ligne :
`stratege:finish_bonus=0:deny_weight=1`.

## Statistiques

- **win_rate** + IC 95 % (Wilson) ; **win_index** = win_rate × nb joueurs (1.0 = niveau du hasard).
  À comparer entre tailles de table plutôt que le taux brut.
- **avg_round_score** ± IC 95 % (approximatif : les manches d'une partie sont corrélées).
- **finish_rate**, **doubled_rate** (part des fins de manche sanctionnées).
- Les sièges sont tirés au sort à chaque partie : pas de biais de position.

## Règles (livret officiel)

- 150 cartes : 5 × −2, 10 × −1, 15 × 0, 10 × chaque valeur de 1 à 12.
- Manche 1 : le plus gros total des 2 cartes retournées commence (égalité : tirage au sort, non précisé par le livret).
  Manches suivantes : le joueur qui a terminé la précédente commence.
- Colonne de 3 cartes identiques visibles : défaussée (après la carte échangée), y compris au retournement final.
- Le finisseur double son score s'il n'est pas strictement le plus bas, uniquement si ce score est positif.
- Fin de partie quand un joueur atteint 100 en fin de manche ; le plus petit total gagne.

## Architecture

```
src/skyjo/
  cards.py        paquet, pioches, CardBelief (distribution des cartes inconnues)
  board.py        plateau 3×4 + BoardView (lecture seule, cartes cachées illisibles)
  rules.py        variantes de règles
  game.py         moteur de manche / partie + journal de décisions
  strategies/     interface, moteur heuristique paramétrable, personnalités
  simulate.py     N parties, agrégation par stratégie, parallélisme
  tournament.py   toutes les tables d'un pool
  sweep.py        balayage d'un paramètre
  analysis.py     analyse du journal (pandas)
  optimize.py     optimisation multi-paramètres
  coach.py        meilleur coup dans une situation (Monte Carlo)
  progress.py     barre de progression, exécution parallèle
  replay.py       enregistrement d'une partie expliquée + export HTML
  viewer/         lecteur HTML (template.html)
```
