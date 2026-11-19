# Synthèse OPS — décisions, preuves, dimensionnement, limites

Document de synthèse demandé par le sujet : **pourquoi** ces choix techniques,
**ce qui a réellement tourné**, **quelles ressources** il faut, et **ce qui ne
marche pas encore**.

---

## 1. Décisions techniques et justifications

### 1.1 Formulation du problème

| Décision | Alternative écartée | Pourquoi |
|---|---|---|
| **Régression de la RUL** plafonnée à 90 jours | Classification « panne dans 30 jours ? » | La régression donne une information exploitable (un délai, donc un ordre de priorité de remplacement) et se dégrade en classification quand on veut une alerte, alors que l'inverse est impossible. Le plafonnement rend la cible bornée et interprétable : `90` = « pas de panne en vue ». |
| **Features de l'état du jour seulement** | Deltas SMART sur 7 jours | Une seule ligne Drive Stats suffit, donc l'API peut répondre **sans historique du disque**. Les deltas seraient plus informatifs mais imposeraient de connaître le passé, et donc de restreindre le service à la prédiction par lot. Identifié comme évolution v2. |
| **Valeurs SMART `raw` uniquement** | Valeurs `normalized` | Les valeurs `normalized` sont propres à chaque constructeur : une même valeur n'a pas le même sens chez Seagate et chez Toshiba. Une feature `vendor` est ajoutée pour que le modèle puisse différencier les sémantiques. |
| **Échantillon de disques sains par hachage du numéro de série** | Tirage aléatoire avec graine | Le hachage ne demande **aucune liste à stocker** et donne le même échantillon à chaque exécution, dans n'importe quel langage. `hashlib.md5` est préféré au `hash()` intégré, randomisé par processus. |
| **Label des sains conditionné à l'observation de l'horizon** | Étiqueter tous les sains à `H` | Sans cette condition, un disque dont on perd la trace après 10 jours serait étiqueté « 90 jours à vivre » sans aucune preuve : c'est un biais de censure, et il tire le modèle vers des RUL trop optimistes. |

### 1.2 Pile technique

| Décision | Pourquoi |
|---|---|
| **uv + `uv.lock`** | Installation reproductible à l'identique, et résolution rapide. |
| **DuckDB pour l'ingestion** | ~35 Go de CSV décompressés. DuckDB les lit en streaming, et `union_by_name = true` absorbe les variations de schéma entre trimestres sans code spécifique. Rien ne passe intégralement en mémoire pandas. |
| **`HistGradientBoostingRegressor`** | Gère **nativement les valeurs manquantes**, très présentes dans les attributs SMART (colonnes absentes d'un trimestre, disques qui ne remontent pas un attribut), insensible à l'échelle, et s'entraîne en quelques secondes. La performance n'étant pas notée, la robustesse et la rapidité comptent davantage. Perte `absolute_error`, cohérente avec la MAE suivie en production. |
| **Prétraitement embarqué dans le `Pipeline` sklearn** | Un seul artefact déployable : l'API applique exactement les mêmes transformations que l'entraînement. C'est ce qui évite le décalage entraînement/service, et c'est aussi ce qui fait que le prétraitement est **versionné et tracé** avec le modèle. |
| **`OneHotEncoder(max_categories=K, handle_unknown="infrequent_if_exist")`** | Implémente exactement la règle « top-K modèles + modalité autre », sans code maison : les modèles rares sont regroupés à l'apprentissage, et un modèle inconnu à la prédiction tombe dans cette même modalité au lieu de faire échouer l'appel. |
| **Configuration Pydantic validée au chargement** | Une clé inconnue ou une valeur hors bornes fait échouer le **démarrage**, pas le run — après vingt minutes d'ingestion. |
| **cloudpickle plutôt que le format skops** | Le pipeline embarque `derive_features`, une fonction du paquet, que skops refuse de recharger sans liste blanche explicite. |

### 1.3 Qualité des données

Le contrat distingue deux sévérités, et ce choix est le cœur de la fiabilité :

- **bloquant** — le lot est refusé, le flow s'arrête, une alerte part, aucun
  modèle n'est publié, l'API continue de servir le champion ;
- **non bloquant** — les lignes fautives partent en quarantaine, le lot continue,
  et le compteur remonte dans Grafana.

Rejeter une journée entière parce que Backblaze a publié trois
`capacity_bytes = -1` serait disproportionné et finirait par rendre le contrôle
inutilisable (on le désactiverait). Mais laisser passer silencieusement 50 % de
lignes douteuses serait pire. D'où le **seuil** : au-delà de 5 % de quarantaine,
la règle non bloquante redevient bloquante.

Trois expectations sont **personnalisées**, parce qu'elles encodent une
connaissance du jeu que la bibliothèque standard n'a pas :

| Expectation | Ce qu'elle attrape |
|---|---|
| `ExpectColumnValuesToBeValidDriveSerial` | Champ tronqué ou colonnes décalées à l'analyse du CSV. |
| `ExpectColumnValuesToNotBeKnownFailedDrive` | Un disque déclaré en panne qui réapparaît : soit `failure` est erroné, soit un lot déjà traité est rejoué. Dans les deux cas, le label RUL serait faux. **Avertissement et quarantaine, pas blocage** : sur les données réelles, un disque (`2BJD2XYD`) émet encore après sa panne déclarée, et bloquer refuserait tout Q2. Une résurrection massive dépasse le plafond de quarantaine (5 %) et bloque. |
| `ExpectDriveFailureRateToBeBelow` | Anomalie de volume : un taux de pannes très supérieur à 0,1 ‰ signale un fichier corrompu, pas une vague de pannes. |

### 1.4 Orchestration

| Décision | Pourquoi |
|---|---|
| **Pas de sous-flow** | Un lot repris doit apparaître dans **un seul** flow run, avec tout son historique. Avec des sous-flows, une reprise donne deux runs sans lien dans l'interface. |
| **Entraînements en tâches concurrentes** | Baseline et candidat sont indépendants : `.submit()` les lance en parallèle. |
| **Runs MLflow frères, reliés par `training.group_id`** | L'état « run actif » de MLflow est global au processus : des runs imbriqués lancés depuis plusieurs threads se mélangeraient, et un artefact finirait dans le mauvais run. |
| **Retries 2/4/8/16 s sur MLflow et PostgreSQL, relances HTTP courtes** | Ce sont des indisponibilités transitoires typiques. Le client MLflow est limité à 2 relances (`MLFLOW_HTTP_REQUEST_*`) : sinon il bloquait ~4 min par tentative et masquait les retries Prefect. Un entraînement lancé pendant une panne échoue proprement en 10 s, avec une alerte. |
| **Un run à la fois par deployment** | La scrutation part toutes les 30 s ; un lot réel prend plus longtemps. Sans limite, plusieurs runs chargeaient le même lot jusqu'à l'OOM. Un tick en trop est annulé, le lot passe au suivant. |
| **`as_of` par défaut = dernier lot reçu** | Un entraînement planifié ne doit pas apprendre sur l'avenir du rejeu : il se place à la date du dernier lot traité. |
| **Cache sur `INPUTS + TASK_SOURCE`** | Le code de la tâche fait partie de la clé : modifier la tâche invalide le cache, ce qu'une clé construite seulement sur les entrées ne ferait pas. |
| **Hooks `on_failure`** | Toute exécution échouée alerte et écrit son statut en base : aucun échec silencieux. |
| **Lot refusé laissé dans `data/landing/`** | La reprise après correction est triviale, et l'état du dépôt dit ce qui reste à traiter. |

### 1.5 Exploitation

| Décision | Pourquoi |
|---|---|
| **Version du modèle dans chaque réponse** | Sans cela, une prédiction surprenante remontée par l'exploitation n'est pas rattachable à une version, et le diagnostic est impossible. |
| **Traces en base non bloquantes** | Si PostgreSQL tombe, l'API continue de répondre : une trace perdue ne doit pas faire perdre une prédiction. `/health` signale la dégradation. |
| **`/health` à trois états** | `ok`, `degraded` (base de traces ou registre MLflow injoignable, prédictions toujours servies), `unavailable` → `503` (pas de modèle). Un unique booléen « up/down » ne permettrait pas de distinguer « dégradé mais utile » de « inutilisable ». |
| **`POST /admin/reload`** | Prendre en compte une promotion ou un rollback sans redémarrer l'API — c'est ce qui rend le retour arrière observable en direct. |
| **Modèle de disque inconnu : avertissement, pas erreur** | Le parc évolue en permanence. Refuser la requête serait un faux négatif de disponibilité. |
| **Règle de promotion sur la même validation** | Le champion est rechargé et réévalué sur la validation du challenger. Comparer à ses métriques stockées mélangeait qualité du modèle et difficulté du jeu (+3,55 % affiché, −3,4 % réel). |
| **`OMP_NUM_THREADS=1` pour l'API** | La concurrence vient des workers uvicorn, pas des threads du modèle : ×17 de débit à 4 workers (section 4). |
| **Dashboard Grafana généré par script** | Un JSON de dashboard écrit à la main est illisible en revue et impossible à maintenir. `scripts/build_dashboard.py` le décrit en Python ; le JSON est un artefact reproductible. |

---

## 2. Preuves d'intégration

Tout ce qui suit a tourné le 18 novembre 2026 sur la **pile Compose et les
données Backblaze réelles** (Q4 2025 à Q2 2026, 273 fichiers journaliers d'environ
345 000 lignes). Chaque ligne renvoie à une trace dans `preuves/`.

| Étape | Ce qui a été vérifié | Preuve |
|---|---|---|
| Installation reproductible | Clone vierge, `mise run setup`, `lint`, `test` : 114 tests au vert, aucun fichier modifié. | `01_setup.txt` |
| Ingestion | Archives en cache revérifiées par sha256 ; bronze en deux passes DuckDB : 364 674 disques, 3 497 pannes, 580 279 lignes, empreinte des sources. | `02_ingestion.txt` |
| Entraînement et registre | Baseline et HGB, tags `data.*` / `git.commit` / `trigger`, artefacts (manifeste, features, résidus), v1 promue `champion`. | `03`, `04a-b` |
| Rejeu et déclenchement automatique | Le runner traite chaque journée déposée (scrutation 30 s, un run à la fois) : validation, quarantaine, ~345 000 lignes scorées, feedbacks. | `06_rejeu.txt`, `07a` |
| Règle de mise en service | Challenger au 15 mai **rejeté** : `mae_lt30_gain_insuffisant:-0.0340<0.05`, champion réévalué sur la même validation. | `08`, `05b` |
| Retour arrière | Promotion forcée par l'overlay, `/admin/reload`, `rollback`, retour à la v1 ; version retirée taguée `rolled_back`. | `11_rollback.txt`, `05c` |
| Contrôles bloquants | Lot corrompu : 4 règles violées, flow arrêté, deux alertes critiques, rien en aval, API intacte. | `09`, `07c` |
| Quarantaine et non-résurrection | 2 % de `capacity_bytes = -1` : lot accepté, 6 932 lignes écartées. `2BJD2XYD`, déclaré en panne en octobre, écarté chaque jour. | `09` |
| Dépendance indisponible | MLflow arrêté : `/health` en `degraded`, `/admin/reload` en 503 en 3 s, prédictions servies, entraînement en échec propre avec alerte en 10 s. | `10` |
| Supervision | Trois vues Grafana alimentées par les traces : technique, métier, croisée (v1 : 29,0 j de MAE en validation, 53,3 j en production). | `12-14` |
| Alertes | Workflow Teams (Adaptive Card) : alerte de test et incident « lot invalide » reçus. | `00_anomalies_donnees_reelles.md` (n° 12) |
| Charge | 41 paliers, 1 à 8 workers, lots de 1 à 1 000, coût de la persistance. | `15_loadtest.md` |

### Anomalies rencontrées, et ce qui a été fait

Avant le passage sur données réelles :

1. **FastAPI 0.155 casse le serveur Prefect 3.8** (paramètre `min_items`
   supprimé) : `fastapi>=0.115,<0.155`, motif en commentaire.
2. **Le planificateur du serveur Prefect conteneurisé** ne crée aucun run
   (SQLAlchemy 2.1 embarqué) : le conteneur réinstalle `sqlalchemy<2.1`.
3. **MLflow 3 refuse de recharger un pipeline** contenant une fonction du projet
   au format skops : sérialisation `cloudpickle` explicite.

Le premier passage sur données réelles en a révélé **douze autres**, toutes
corrigées, commitées et le plus souvent couvertes par un test
(`preuves/00_anomalies_donnees_reelles.md`) :

- **intégration** : le port 5000 pris par AirPlay sur macOS, un serveur MLflow
  2.19 face au client 3.18, des artefacts écrits dans un dossier local ;
- **volume** : des `NaN` refusés par `jsonb` qui faisaient échouer le scoring,
  des runs empilés jusqu'à l'OOM du conteneur ;
- **configuration** : des variables exportées par `mise` qui faisaient écrire
  les tests et les démos dans les vraies données, et qui effaçaient l'URL du
  webhook dans les conteneurs ;
- **méthode** : une règle de promotion qui comparait deux jeux de validation
  différents (+3,55 % affiché, −3,4 % réel), un entraînement planifié qui
  apprenait sur l'avenir du rejeu ;
- **découvertes en produisant les preuves** : la règle de non-résurrection
  jamais active en production (et qui, active, aurait bloqué tout Q2), et un
  `/admin/reload` qui bloquait 4 minutes quand MLflow était arrêté.

La campagne de charge en a ajouté une treizième, la plus coûteuse : la
contention des threads OpenMP, qui plafonnait l'API à ~30 req/s (section 4).

---

## 3. Ce que le pipeline produit, concrètement

Sur les données réelles, champion v1 entraîné au 1er avril 2026 (validation :
12 973 lignes, disques jamais vus à l'entraînement) :

| | Baseline (médiane) | HistGradientBoosting |
|---|---|---|
| MAE globale | 45,7 j | **29,0 j** |
| MAE sur RUL < 30 j | 76,2 j | **48,0 j** |
| Précision de l'alerte < 30 j | — | 0,65 |
| Rappel de l'alerte < 30 j | 0,00 | **0,06** |

La baseline sert exactement à cela : sans elle, « MAE = 29 jours » ne veut rien
dire. Avec elle, on sait que le modèle réduit l'erreur d'un tiers. Mais le
**rappel de l'alerte reste très bas** : le modèle prédit rarement moins de
30 jours, et rate la plupart des disques en fin de vie (section 5). La
performance n'est pas l'objet du projet ; la chaîne, elle, rend ce défaut
visible et mesurable en production.

En exploitation simulée (10 journées de Q2 rejouées) : environ 345 000 disques
scorés par jour, 180 à 205 alertes « RUL < 30 j » par jour, 940 feedbacks
réconciliés automatiquement, une MAE de production de 53,3 j (sur des feedbacks
qui ne concernent encore que des pannes, voir section 5).

---

## 4. Charge et dimensionnement

Protocole, tableau complet, biais déclarés : [`loadtest/PROTOCOLE.md`](loadtest/PROTOCOLE.md)
et `preuves/15_loadtest.md`. Campagne du 18 novembre sur le champion réel,
1 à 8 workers, Apple M5 10 cœurs, Locust sur le même hôte.

### Le résultat qui détermine tout le reste

Avec le réglage par défaut, **le débit plafonnait vers 30 req/s quel que soit
le nombre de workers**, et baissait dès deux requêtes simultanées. Cause :
chaque prédiction HistGradientBoosting lance une équipe de threads OpenMP sur
tous les cœurs ; deux prédictions concurrentes se les disputent. Avec
`OMP_NUM_THREADS=1` (service `api` de `compose.yml`) :

| Unitaire | Par défaut | `OMP_NUM_THREADS=1` |
|---|---|---|
| 1 worker, 1 utilisateur | 79 req/s, p95 24 ms | 267 req/s, p95 4 ms |
| 2 workers, 2 utilisateurs | 38 req/s, p95 63 ms | 463 req/s, p95 5 ms |
| 4 workers, 8 utilisateurs | 30 req/s, p95 310 ms | 463 req/s, p95 36 ms |

La mesure du coût de calcul hors HTTP confirme : avec 10 threads, prédire une
seule ligne prend 4 fois plus longtemps (5,9 ms contre 1,5 ms) ; le parallélisme
ne devient rentable qu'au-delà de ~10 000 lignes. Le runner, qui score 345 000
lignes d'un coup, le garde donc.

### Recommandation

Hypothèse : **parc de 300 000 disques scoré une fois par jour**, plus quelques
milliers d'appels unitaires.

| Mode | Débit mesuré | 300 000 disques |
|---|---|---|
| Runner `daily_batch`, traces comprises | ~11 500 lignes/s | ~30 s |
| API, lots de 100, 2 workers | 34 000 lignes/s | ~9 s |
| API, lots de 1 000, 4 workers | 136 000 lignes/s | ~2 s |
| API, unitaire, 2 workers | 463 req/s | ~11 min |

**2 workers uvicorn avec `OMP_NUM_THREADS=1`, 2 vCPU et 1 Go de mémoire**
(~610 Mo mesurés) ; scoring quotidien dans le runner ; lots de 100 pour les
appels par lot (p95 < 25 ms). Le facteur limitant suivant est l'écriture des
traces (~10 000 lignes/s avec `executemany`, mesuré) : passer à `COPY` avant
d'ajouter des workers.

### Marge de sécurité

p95 de 5 à 12 ms dans la configuration recommandée contre 200 ms de budget ;
marge ×100 sur le trafic en ligne prévu ; un parc dix fois plus grand
demanderait ~5 min de scoring quotidien.

---

## 5. Limites connues

### Sur la modélisation

- **Censure à droite** : les disques sains ne sont étiquetés que lorsque
  l'horizon de 90 jours est effectivement observé. Les dernières journées d'un
  trimestre ne sont donc pas utilisables pour l'entraînement, et le jeu
  d'entraînement est toujours en retard de 90 jours sur les données disponibles.
- **Features limitées à l'état du jour** : les tendances SMART seraient plus
  informatives. Cette v2 est identifiée mais restreindrait le service à la
  prédiction par lot.
- **Peu de pannes par modèle de disque** : le regroupement top-K soigne le
  volume mais laisse les modèles rares mal servis.
- **Sémantique SMART hétérogène** entre constructeurs. La feature `vendor` est
  un palliatif, pas une solution.
- **Pas de recherche d'hyperparamètres** : les valeurs de `config/base.yaml` sont
  raisonnables, pas optimisées. La performance n'étant pas notée, l'effort est
  passé ailleurs.
- **Rappel de l'alerte « RUL < 30 j » à 5-6 %** : précision correcte (0,65 à
  0,72), mais la plupart des disques en fin de vie ne sont pas signalés. Beaucoup
  de pannes Backblaze surviennent sans dégradation SMART visible ; des features
  de tendance (v2) et une pondération de la zone critique seraient les pistes.
- **Validation peu représentative du parc** : avec `as_of = 2026-04-01`, la
  coupure temporelle tombe fin décembre ; après le 1er janvier, seuls les
  disques en panne ont un label résolu (les sains sont censurés à 90 j). La
  validation contient donc 40 % de RUL < 30 j, bien plus que le parc.
- **MAE de production biaisée pendant le rejeu** : un feedback de survie
  (RUL réelle ≥ 90 j) n'arrive que 90 jours après la prédiction. Les 940
  feedbacks actuels sont tous des pannes, d'où 53,3 j de MAE de production
  contre 29,0 j en validation.

### Sur la chaîne

- **Le rejeu n'est qu'une approximation d'un flux de production** : pas de
  retards, pas de doublons, pas d'arrivée partielle d'une journée.
- **Le générateur synthétique valide la chaîne, pas le modèle.** Sa dégradation
  est monotone et simple ; les corrélations réelles entre attributs SMART ne sont
  pas reproduites.
- **Pas de détection de dérive automatique** : les vues Grafana permettent de
  *voir* une dérive (MAE de validation contre MAE de production), mais rien ne
  déclenche encore un réentraînement ou un rollback automatiquement. C'est un
  choix : un rollback automatique sur une métrique bruitée est plus dangereux
  qu'un rollback décidé.
- **Alertes sur un seul canal** (workflow Teams). Pas de politique d'escalade,
  pas de déduplication : un lot refusé qui reste dans `landing/` est refusé de
  nouveau à chaque tick de scrutation, avec une alerte à chaque fois.
- **Mesures de charge sur un seul hôte** (API, Locust et pile Compose sur les
  mêmes 10 cœurs) et une exécution par point (variation jusqu'à ~40 %) : les
  débits à 8 workers sont des minorants. Prochaine amélioration du protocole :
  un hôte séparé pour Locust et trois exécutions par point.
- **Écriture des traces par `executemany`** : ~10 000 lignes/s mesurées, premier
  facteur limitant d'un scoring de parc avec persistance.
- **Redémarrer le runner pendant un run** laisse l'emplacement de la limite de
  concurrence occupé ~5 minutes : les ticks de scrutation sont annulés jusqu'à
  son expiration.
- **Secrets en clair dans `.env`** pour le développement local. Acceptable ici,
  à remplacer par un gestionnaire de secrets dans un vrai déploiement. Aucun
  secret n'est commité.
- **Pas d'authentification sur l'API ni sur Grafana** (au-delà du mot de passe
  administrateur par défaut). Hors périmètre du sujet, mais à signaler.
