# Protocole de charge et dimensionnement

## 1. Ce qu'on cherche à savoir

Une seule question, posée en termes d'exploitation :

> **Combien de workers faut-il pour scorer un parc de N disques une fois par
> jour, en restant sous 200 ms de latence p95 et sous 1 % d'erreurs ?**

Et une question corollaire, qui change complètement la réponse :

> **Faut-il scorer disque par disque, ou par lot ?**

## 2. Critère de capacité soutenable

Une configuration est déclarée soutenable si, sur la durée du palier :

| Critère | Seuil |
|---|---|
| Latence p95 | **< 200 ms** |
| Taux d'erreur | **< 1 %** |
| Erreurs 5xx | **0** |

Le p95 est retenu plutôt que la moyenne : une moyenne basse masque des
requêtes très lentes, et c'est la queue de distribution qui dégrade
l'exploitation.

## 3. Scénarios

Définis dans `loadtest/locustfile.py`, sélectionnables par `--class-picker` :

| Classe | Requête | Cas d'usage représenté |
|---|---|---|
| `PredictionUnitaire` | `POST /predict`, 1 disque | Alerte en ligne sur un disque précis. |
| `PredictionLot100` | `POST /predict/batch`, 100 disques | Scoring d'un rack. |
| `PredictionLot1000` | `POST /predict/batch`, 1 000 disques | Scoring d'un fragment de parc. |
| `TraficRealiste` | 80 % unitaire, 18 % lot 100, 2 % lot 1 000, plus `/health` et `/model` | Mélange attendu en production. |

Les observations sont tirées aléatoirement avec des niveaux de dégradation
variés, pour que le modèle ne travaille pas toujours sur la même branche.

Par défaut, `persist: false` : on mesure la capacité de calcul de l'API, et la
campagne ne pollue pas les traces exploitées par Grafana. Une série dédiée
(`LOADTEST_PERSIST=1`, traces activées) mesure le coût de l'écriture en base.

## 4. Configurations comparées

| Variable | Valeurs |
|---|---|
| Workers uvicorn | 1, 2, 4, 8 |
| Taille de lot | 1, 100, 1 000 |
| Concurrence (utilisateurs Locust) | 1 à 64 selon le scénario |
| Threads OpenMP par prédiction | 1 (`OMP_NUM_THREADS=1`), comparé au défaut (tous les cœurs) |
| Persistance des prédictions | sans, puis avec (lots de 100, 2 workers) |

Un **balayage de concurrence** est fait à configuration fixée, plutôt qu'un seul
point de mesure : c'est la seule façon d'obtenir une courbe latence/débit, donc
de savoir où se situe le point de saturation.

## 5. Mode opératoire

La campagne complète est scriptée, donc reproductible :

```bash
bash scripts/loadtest_campaign.sh config/base.yaml 30s       # ou : mise run loadtest:campaign
AVEC_PERSISTANCE=1 bash scripts/loadtest_campaign.sh config/base.yaml 30s   # + coût des traces
uv run python scripts/collect_loadtest_results.py loadtest/results   # ou : mise run loadtest:report
uv run python scripts/bench_compute.py                       # coût de calcul hors HTTP
```

Le script démarre sa propre API sur le port **8011** (celle de la pile Compose
occupe 8001), avec le champion du registre. Précautions appliquées :

- **`OMP_NUM_THREADS=1`** pour l'API, comme en production (section 8) ;
- **mise en chauffe de chaque worker** : attente proportionnelle au nombre de
  workers, puis un **palier de rodage écarté, à la concurrence maximale de la
  série**, pour que tous les workers aient chargé leur modèle avant la mesure ;
- **Locust sur 4 processus** (`--processes 4`) : un seul processus plafonne vers
  500 à 600 req/s et mesurerait le client ;
- paliers de 30 s, un CSV par palier ; aucune autre charge pendant la mesure
  (`caffeinate` pour empêcher la mise en veille).

## 6. Biais de mesure à déclarer

- **Même machine pour Locust et l'API** : Apple M5, 10 cœurs, 16 Go, avec la
  pile Compose (PostgreSQL, MLflow, Prefect, Grafana) en fonctionnement. Au-delà
  de 4 workers, l'API et les 4 processus Locust se disputent les cœurs : les
  débits à 8 workers sont des **minorants**.
- **Une seule exécution par point**. D'une exécution à l'autre, un même palier
  varie jusqu'à ~40 % (8 workers, 8 utilisateurs : 936 puis 512 req/s). Il faut
  lire les **tendances** de la courbe, pas un point isolé.
- **Incidents écartés** : un rejeu lancé en parallèle (16:13-16:20) et une mise
  en veille de la machine ont faussé 13 paliers. Ils ont été détectés (débits
  incohérents, durée réelle de 215 à 1 050 s au lieu de 30 s) puis **mesurés à
  nouveau** ; seuls les paliers refaits figurent dans le tableau.
- Les observations sont synthétiques (tirées dans `locustfile.py`), avec des
  niveaux de dégradation variés pour ne pas toujours parcourir les mêmes branches
  des arbres.

## 7. Coût de calcul mesuré directement

`scripts/bench_compute.py` : champion v1 (HistGradientBoosting, 300 itérations,
profondeur 8), lignes réelles de la couche bronze, médiane de 30 appels.

| Taille du lot | 1 thread : ms/appel | 1 thread : lignes/s | 10 threads : ms/appel | 10 threads : lignes/s |
|---|---|---|---|---|
| 1 | 1,5 | 646 | 5,9 | 168 |
| 10 | 1,6 | 6 303 | 13,8 | 726 |
| 100 | 2,2 | 45 519 | 14,0 | 7 164 |
| 1 000 | 8,9 | 111 987 | 16,3 | 61 277 |
| 10 000 | 70,4 | 142 077 | 28,6 | 350 252 |

Lecture :

- **le coût est presque fixe jusqu'à 100 lignes** : prédire 100 disques coûte
  à peine plus qu'en prédire un. Scorer par lots est donc l'essentiel du gain ;
- **le parallélisme interne du modèle est contre-productif pour l'API** : avec
  10 threads, une seule ligne prend 4 fois plus longtemps (démarrage des
  threads). Il ne devient rentable qu'au-delà de ~10 000 lignes, c'est-à-dire
  pour le scoring quotidien du runner (345 000 lignes d'un coup), qui garde
  donc le réglage par défaut.

## 8. Résultats

Campagne du 18 novembre 2026, champion v1 servi depuis le registre MLflow de la
pile Compose. Tableau généré par `scripts/collect_loadtest_results.py` ;
« soutenable » = p95 < 200 ms et moins de 1 % d'erreurs.

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

### Le résultat qui a tout changé : un thread OpenMP par prédiction

Première campagne du jour, **réglage par défaut** (OpenMP sur tous les cœurs ;
paliers conservés dans `loadtest/results/2026-11-18_avant_omp/`), comparée à la
campagne finale :

| Unitaire | OpenMP par défaut | `OMP_NUM_THREADS=1` | Gain |
|---|---|---|---|
| 1 worker, 1 utilisateur | 79 req/s, p95 24 ms | 267 req/s, p95 4 ms | ×3,4 |
| 1 worker, 4 utilisateurs | 32 req/s, p95 140 ms | 224 req/s, p95 24 ms | ×7 |
| 2 workers, 2 utilisateurs | 38 req/s, p95 63 ms | 463 req/s, p95 5 ms | ×12 |
| 4 workers, 8 utilisateurs | 30 req/s, p95 310 ms | 463 req/s, p95 36 ms | ×16 |
| 8 workers, 8 utilisateurs | 28 req/s, p95 320 ms | 512 req/s, p95 20 ms | ×18 |

Avec le réglage par défaut, le débit **plafonnait vers 30 req/s quel que soit
le nombre de workers** et **baissait** dès 2 requêtes simultanées : chaque
prédiction lançait une équipe de 10 threads, et deux prédictions concurrentes
(dans le même worker ou dans deux workers) se disputaient les mêmes cœurs. La
campagne du 5 novembre montrait déjà ce motif (61 → 49 → 35 req/s), sans qu'il
ait été diagnostiqué. Correctif : `OMP_NUM_THREADS=1` sur le service `api`
(`compose.yml`) et dans la campagne ; la concurrence vient alors des
workers uvicorn, et les workers redeviennent utiles.

### Lecture

**Prédiction unitaire.** ~250 req/s par worker avec une latence de quelques
millisecondes ; **463 req/s à 2 workers** (p95 5 ms) ; jusqu'à **772 req/s à 8
workers** (p95 40 ms). Au-delà de 2 workers, la progression ralentit : les
workers, Locust et la pile Compose se partagent les 10 cœurs de l'hôte.

**Lots de 100.** **17 000 lignes/s sur 1 worker, 34 000 sur 2** (p95 6 ms),
~39 000 à 51 000 à 8 workers. Le lot de 100 est le meilleur compromis : débit
×70 par rapport à l'unitaire, et latence inchangée.

**Lots de 1 000.** 54 000 lignes/s sur 1 worker, **136 000 lignes/s sur 4
workers** (p95 39 ms) : le meilleur débit mesuré. Chaque requête porte alors
plusieurs Mo de JSON, ce qui rend ce mode moins adapté aux appels interactifs.

**Coût de la persistance.** Lots de 100, 2 workers, chaque prédiction écrite
dans `tracing.predictions` : le débit tombe de 34 000 à **~10 000 lignes/s** et
**plafonne** à ce niveau quelle que soit la concurrence. L'écriture en base
(`executemany`) est le vrai goulot dès qu'on trace tout. C'est cohérent avec le
scoring quotidien du runner : ~345 000 lignes traitées en ~30 s, écriture
comprise.

**Trafic réaliste** (80 % unitaire, 18 % lot 100, 2 % lot 1 000, temps de
réflexion de 0,1 à 0,5 s) : 103 req/s pour 32 utilisateurs avec un p95 de
14 ms, très loin de la saturation.

**Ressources** (somme des processus uvicorn, après chauffe) :

| Workers | Mémoire totale | Par worker |
|---|---|---|
| 1 | 308 Mo | 308 Mo (processus unique) |
| 2 | 609 Mo | ~195 Mo + superviseur ~220 Mo |
| 4 | 1 024 Mo | ~200 Mo + superviseur |
| 8 | 2 152 Mo | ~240 Mo + superviseur |

Le modèle sérialisé pèse 1,1 Mo : l'essentiel de la mémoire est l'interpréteur
et la pile scientifique (pandas, numpy, scikit-learn).

**Aucune erreur** sur les 41 paliers retenus (328 244 requêtes) : la
saturation se traduit par de la latence, pas par des erreurs.

## 9. Recommandation

Hypothèse de travail : **parc de 300 000 disques, scoré une fois par jour**,
plus quelques milliers d'appels unitaires pour les alertes à la demande.

| Mode de scoring | Débit mesuré | Temps pour 300 000 disques |
|---|---|---|
| Runner `daily_batch` (en processus, écriture des traces comprise) | ~11 500 lignes/s | **~30 s** |
| API, lots de 100, 2 workers, sans traces | 34 000 lignes/s | **~9 s** |
| API, lots de 1 000, 4 workers, sans traces | 136 000 lignes/s | **~2 s** |
| API, unitaire, 2 workers | 463 req/s | ~11 min |
| *(avant correction OpenMP)* API, unitaire, 4 workers | 30 req/s | *~2 h 45* |

**Recommandation : 2 workers uvicorn avec `OMP_NUM_THREADS=1`, 2 vCPU et 1 Go
de mémoire, et le scoring du parc dans le runner.**

1. **Le réglage des threads compte plus que le nombre de workers.** Sans
   `OMP_NUM_THREADS=1`, ajouter des workers ne sert à rien (~30 req/s de 1 à 8
   workers). Avec, 2 workers servent ~460 req/s unitaires avec un p95 de 5 ms.
   C'est le premier réglage à vérifier sur tout service qui sert un modèle à
   base d'arbres de scikit-learn.
2. **Le scoring quotidien ne passe pas par l'API.** Le flow `daily_batch` score
   le lot en processus (avec tous les cœurs, rentables à cette taille) en ~30 s
   par jour, écriture des traces comprise. L'API sert les appels en ligne et les
   lots ponctuels.
3. **Deux workers suffisent** pour l'usage en ligne prévu : marge ×100 sur
   « quelques milliers d'appels par jour », et un worker reste libre si l'autre
   traite un lot. Mémoire mesurée : ~610 Mo.
4. **Taille de lot recommandée : 100.** Débit ×70 par rapport à l'unitaire,
   p95 inférieur à 25 ms ; 1 000 maximise le débit mais alourdit chaque requête.
5. **Le facteur limitant suivant est l'écriture des traces** (~10 000 lignes/s
   avec `executemany`). Pour un parc dix fois plus grand, passer à une insertion
   par `COPY` avant d'ajouter des workers.

### Marge de sécurité

Le seuil (p95 < 200 ms, moins de 1 % d'erreurs) est tenu avec une large marge
dans la configuration recommandée : 5 à 12 ms de p95 mesurés pour 2 à 4
utilisateurs simultanés, contre 200 ms de budget. Un parc **dix fois plus
grand** (3 millions de disques) demanderait ~5 min de scoring quotidien dans le
runner, écriture des traces comprise.
