# Preuve 15 — campagne de charge et dimensionnement

Campagne du 18 novembre 2026 contre le champion réel (v1), servi depuis le
registre MLflow de la pile Compose. Machine : Apple M5, 10 cœurs, 16 Go ; API et
Locust (4 processus) sur le même hôte. Paliers de 30 s, `OMP_NUM_THREADS=1`.
Protocole, biais et recommandation : `loadtest/PROTOCOLE.md`.

Commandes :

```bash
AVEC_PERSISTANCE=1 bash scripts/loadtest_campaign.sh config/base.yaml 30s
uv run python scripts/collect_loadtest_results.py loadtest/results
uv run python scripts/bench_compute.py
```

## Tableau (relu depuis `loadtest/results/sw_*_stats.csv`)

| Scénario | Workers | Concurrence | Requêtes | Erreurs | RPS | Lignes/s | moy. (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Soutenable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PredictionLot100 | 1 | 1 | 5093 | 0 | 169.6 | 16957 | 6 | 5 | 8 | 15 | oui |
| PredictionLot100 | 1 | 2 | 4630 | 0 | 154.2 | 15417 | 13 | 11 | 25 | 41 | oui |
| PredictionLot100 | 1 | 4 | 4555 | 0 | 151.7 | 15167 | 26 | 25 | 31 | 58 | oui |
| PredictionLot100 | 2 | 2 | 10251 | 0 | 341.4 | 34140 | 6 | 6 | 6 | 8 | oui |
| PredictionLot100 | 2 | 4 | 9371 | 0 | 312.1 | 31210 | 13 | 7 | 23 | 29 | oui |
| PredictionLot100 | 2 | 8 | 7577 | 0 | 252.2 | 25218 | 32 | 30 | 39 | 66 | oui |
| PredictionLot100 | 4 | 2 | 9290 | 0 | 309.4 | 30942 | 6 | 6 | 8 | 11 | oui |
| PredictionLot100 | 4 | 4 | 8832 | 0 | 294.2 | 29417 | 13 | 7 | 25 | 30 | oui |
| PredictionLot100 | 4 | 8 | 6780 | 0 | 225.6 | 22561 | 35 | 24 | 74 | 150 | oui |
| PredictionLot100 | 4 | 16 | 6386 | 0 | 210.6 | 21058 | 75 | 64 | 160 | 330 | oui |
| PredictionLot100 | 8 | 4 | 11748 | 0 | 391.1 | 39110 | 10 | 9 | 15 | 31 | oui |
| PredictionLot100 | 8 | 8 | 8409 | 0 | 280.0 | 27997 | 28 | 29 | 41 | 73 | oui |
| PredictionLot100 | 8 | 16 | 15386 | 0 | 512.0 | 51198 | 31 | 16 | 90 | 130 | oui |
| PredictionLot1000 | 1 | 1 | 1599 | 0 | 53.3 | 53265 | 18 | 16 | 21 | 45 | oui |
| PredictionLot1000 | 1 | 2 | 1636 | 0 | 54.5 | 54493 | 35 | 31 | 62 | 79 | oui |
| PredictionLot1000 | 4 | 1 | 1625 | 0 | 54.1 | 54123 | 17 | 16 | 19 | 45 | oui |
| PredictionLot1000 | 4 | 2 | 2842 | 0 | 94.6 | 94624 | 20 | 18 | 23 | 53 | oui |
| PredictionLot1000 | 4 | 4 | 4087 | 0 | 136.0 | 136017 | 28 | 25 | 39 | 72 | oui |
| PredictionLot100Persist | 2 | 2 | 2680 | 0 | 89.3 | 8934 | 22 | 22 | 26 | 31 | oui |
| PredictionLot100Persist | 2 | 4 | 3003 | 0 | 100.1 | 10011 | 40 | 38 | 51 | 77 | oui |
| PredictionLot100Persist | 2 | 8 | 3012 | 0 | 100.5 | 10047 | 79 | 77 | 120 | 160 | oui |
| PredictionUnitaire | 1 | 1 | 8024 | 0 | 267.2 | 267 | 4 | 4 | 4 | 5 | oui |
| PredictionUnitaire | 1 | 2 | 7337 | 0 | 244.4 | 244 | 8 | 8 | 10 | 12 | oui |
| PredictionUnitaire | 1 | 4 | 6731 | 0 | 224.4 | 224 | 18 | 17 | 24 | 30 | oui |
| PredictionUnitaire | 1 | 8 | 6147 | 0 | 204.7 | 205 | 39 | 38 | 55 | 67 | oui |
| PredictionUnitaire | 1 | 16 | 6105 | 0 | 203.6 | 204 | 78 | 75 | 110 | 140 | oui |
| PredictionUnitaire | 2 | 2 | 13912 | 0 | 463.4 | 463 | 4 | 4 | 5 | 6 | oui |
| PredictionUnitaire | 2 | 4 | 11871 | 0 | 395.4 | 395 | 10 | 10 | 12 | 15 | oui |
| PredictionUnitaire | 2 | 8 | 7863 | 0 | 261.8 | 262 | 30 | 27 | 55 | 100 | oui |
| PredictionUnitaire | 2 | 16 | 7691 | 0 | 256.2 | 256 | 62 | 58 | 100 | 130 | oui |
| PredictionUnitaire | 4 | 4 | 8963 | 0 | 298.5 | 298 | 13 | 11 | 27 | 41 | oui |
| PredictionUnitaire | 4 | 8 | 13897 | 0 | 462.7 | 463 | 17 | 10 | 36 | 46 | oui |
| PredictionUnitaire | 4 | 16 | 10590 | 0 | 351.2 | 351 | 45 | 42 | 74 | 130 | oui |
| PredictionUnitaire | 4 | 32 | 8436 | 0 | 280.8 | 281 | 114 | 100 | 200 | 330 | non |
| PredictionUnitaire | 8 | 8 | 15381 | 0 | 512.2 | 512 | 16 | 14 | 20 | 31 | oui |
| PredictionUnitaire | 8 | 16 | 23176 | 0 | 771.6 | 772 | 21 | 20 | 40 | 51 | oui |
| PredictionUnitaire | 8 | 32 | 19974 | 0 | 664.7 | 665 | 48 | 29 | 130 | 170 | oui |
| PredictionUnitaire | 8 | 64 | 17946 | 0 | 597.1 | 597 | 107 | 100 | 210 | 260 | non |
| TraficRealiste | 4 | 8 | 771 | 0 | 25.7 | 26 | 12 | 11 | 20 | 32 | oui |
| TraficRealiste | 4 | 16 | 1549 | 0 | 51.6 | 52 | 8 | 7 | 14 | 30 | oui |
| TraficRealiste | 4 | 32 | 3088 | 0 | 102.8 | 103 | 7 | 6 | 14 | 54 | oui |

Meilleur débit soutenable (p95 < 200 ms) : 136017 lignes/s (PredictionLot1000, 4 worker(s), 4 utilisateur(s) simultané(s))
Soit environ 489 660 833 disques scorés par heure de calcul.

Lecture de la dernière ligne : le meilleur débit soutenable est obtenu par lots
de 1 000 ; la recommandation retient pourtant 2 workers et des lots de 100,
pour la latence (voir le protocole, section 9).

## Coût de calcul hors HTTP (`scripts/bench_compute.py`)

modèle : models:/hdd-rul-hgb@champion | cœurs : 10 | répétitions : 30

| Taille du lot | 1 thread : ms/appel | 1 thread : lignes/s | 10 threads : ms/appel | 10 threads : lignes/s |
|---|---|---|---|---|
| 1 | 1.5 | 646 | 5.9 | 168 |
| 10 | 1.6 | 6 303 | 13.8 | 726 |
| 100 | 2.2 | 45 519 | 14.0 | 7 164 |
| 1 000 | 8.9 | 111 987 | 16.3 | 61 277 |
| 10 000 | 70.4 | 142 077 | 28.6 | 350 252 |

## Avant / après le réglage OpenMP (prédiction unitaire)

| Configuration | OpenMP par défaut | `OMP_NUM_THREADS=1` |
|---|---|---|
| 1 worker, 1 utilisateur | 79 req/s, p95 24 ms | 267 req/s, p95 4 ms |
| 2 workers, 2 utilisateurs | 38 req/s, p95 63 ms | 463 req/s, p95 5 ms |
| 4 workers, 8 utilisateurs | 30 req/s, p95 310 ms | 463 req/s, p95 36 ms |
| 8 workers, 8 utilisateurs | 28 req/s, p95 320 ms | 512 req/s, p95 20 ms |

Paliers « avant » : `loadtest/results/2026-11-18_avant_omp/`. Campagne du
5 novembre (jeu synthétique, autre machine) : `loadtest/results/2026-11-05_synthetique/`.

## Recommandation

**2 workers uvicorn, `OMP_NUM_THREADS=1`, 2 vCPU, 1 Go de mémoire** ; scoring
quotidien du parc dans le runner (~30 s pour 345 000 disques, traces
comprises) ; lots de 100 pour les appels par lot. Prochain facteur limitant :
l'écriture des traces (~10 000 lignes/s).
