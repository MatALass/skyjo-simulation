# Modèles

Les modèles entraînés ne sont pas versionnés (fichiers binaires régénérables). Pour reproduire :

```bash
skyjo ml dataset --games 20000 --seed 1        # → data/ml/dataset.parquet
skyjo ml train                                 # → models/value.joblib + models/value.json (métriques)
skyjo ml eval --games 2000 --seed 2            # le modèle contre l'Expert
```

`models/value.json` (métriques, importances, hyperparamètres) est écrit à côté du modèle.
