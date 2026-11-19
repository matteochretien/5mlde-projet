# Anomalies du premier passage sur données réelles (18 novembre 2026)

Premier passage de la chaîne complète sur les données Backblaze (Q4 2025 à
Q2 2026, 273 fichiers journaliers d'environ 345 000 lignes), avec la pile Docker
Compose. Le jeu synthétique et les tests n'avaient révélé aucun de ces problèmes.
Chacun est corrigé, commité et couvert par un test quand c'est possible.

| # | Symptôme | Cause | Correctif | Commit |
|---|---|---|---|---|
| 1 | Le navigateur reçoit un `403` vide sur `localhost:5000` ; MLflow répond pourtant `200` depuis son conteneur. | Sur macOS, le récepteur AirPlay (`ControlCenter`, en-tête `Server: AirTunes`) écoute sur le port 5000. | MLflow publié sur **5050** côté hôte (`MLFLOW_PORT`). Les conteneurs utilisent toujours `mlflow:5000`. | `b006f97` |
| 2 | L'API n'aurait jamais pu charger le champion : l'expérience pointait vers `/mlartifacts/1`. | `--default-artifact-root /mlartifacts` : chaque client écrit dans **son** `/mlartifacts` local, qui n'existe ni sur l'hôte ni dans le conteneur de l'API. | `--serve-artifacts --artifacts-destination /mlartifacts` : les artefacts passent par le serveur (`mlflow-artifacts:/`). | `b006f97` |
| 3 | La publication d'un modèle aurait échoué. | Serveur MLflow 2.19 face au client 3.18 de `uv.lock` : un client 3.x publie par l'API `logged-models`, absente d'un serveur 2.x. | Image `mlflow:v3.18.0`, plus `--allowed-hosts` (MLflow 3 n'accepte par défaut que localhost et les IP privées) et le pilote `psycopg` 3. | `b006f97` |
| 4 | `daily_batch` échoue au scoring et relance indéfiniment le même lot : `generator didn't stop after throw()`. | (a) Un attribut SMART absent arrive en `NaN`, que `json.dumps` écrit tel quel et que `jsonb` refusé. (b) Le gestionnaire de connexion cédait une seconde fois après l'exception. Une trace perdue faisait donc échouer toute la prédiction, à l'inverse de l'intention du module. | `NaN` et infinis écrits en `null`. Une erreur d'écriture est journalisée et absorbée. Deux tests de non-régression. | `700c64b` |
| 5 | Runs `daily_batch` tués par l'OS (`SIGKILL`, code -9). | La scrutation part toutes les 30 s, et un lot réel prend plus longtemps : plusieurs runs chargeaient le même lot de 345 000 lignes en parallèle, jusqu'à saturer la mémoire du conteneur. | `concurrency_limit=1` avec `CANCEL_NEW` sur `daily_batch` et `train_pipeline`. Le lot reste dans `data/landing/` et passe au tick suivant. | `700c64b` |
| 6 | `pytest` reconstruit la **vraie** couche bronze et traite de vrais lots. | `mise` exporte `.env.example` (`HDD_RUL_DATA_DIR=data`) ; les flows rechargent leur configuration et la variable l'emporte sur les chemins temporaires des tests. | `tests/conftest.py` retire ces variables avant tout test. Vérifié avec un dossier de données « témoin », reste vide. Bronze reconstruit à l'identique (même empreinte). | `700c64b` |
| 7 | Même cause que 6, côté démonstration : `--config config/demo.yaml` écrivait dans `data/` au lieu de `data/demo/`. | `HDD_RUL_DATA_DIR` l'emporte sur le `paths.root` de l'overlay. | `.env.example` n'exporte plus `HDD_RUL_DATA_DIR` ni `HDD_RUL_CONFIG`. | `af6874c` |
| 8 | Un challenger semblait améliorer la zone critique de 3,55 %. | La règle comparait ses métriques à celles **stockées** du champion, calculées sur une autre fenêtre et d'autres disques. | Le champion est réévalué sur la validation du challenger. Sur les mêmes lignes, le challenger est en fait **3,4 % moins bon**. Les tags `promotion.champion_*` gardent les valeurs comparées. | `8d469c6` |
| 9 | La règle « pas de résurrection d'un disque en panne », annoncée comme bloquante, n'apparaît dans aucun rapport de validation de production. | Le flow acceptait la liste des disques déjà en panne en paramètre, mais rien ne la calculait. Une fois la liste calculée, une règle *bloquante* aurait refusé toutes les journées de Q2 : `2BJD2XYD`, déclaré en panne le 2025-10-14, émet encore chaque jour jusqu'au 30 juin (seul cas sur 3 497 pannes). | Liste calculée à chaque lot (historique ingéré et lots déjà traités, avant la date du lot). Les lignes concernées partent en quarantaine. Une résurrection massive dépasse le plafond de 5 % et bloque. | `0ccc3de` |
| 10 | MLflow arrêté : `/admin/reload` ne répond pas en 200 s, `/health` reste `ok`, et un entraînement reste bloqué environ 4 min par tentative. | Le client MLflow relance 7 fois avec un délai croissant. `/health` ne vérifiait pas le registre. Une exception MLflow brute (500) remontait, et un rechargement raté aurait remis le modèle à `None`. | Relances courtes (`MLFLOW_HTTP_REQUEST_*` dans `compose.yml`). `/health` passe en `degraded` sans le registre. Un rechargement raté renvoie `503` et garde le modèle servi. L'entraînement échoue en 10 s, avec une alerte. | `0ccc3de` |
| 11 | L'entraînement planifié de la v6 s'entraîne jusqu'au 30 juin alors que le rejeu en est au 9 avril. | Sans `as_of`, le flow prend tout le bronze, donc l'avenir de la simulation. | Sans date explicite, `as_of` vaut la date du dernier lot traité (`data/processed/`). | `0ccc3de` |
| 12 | Aucune alerte n'arrivait dans le canal, sans aucune erreur côté pipeline. | (a) `.env.example` contenait `HDD_RUL_WEBHOOK_URL=` vide ; `mise` l'exporte dans tout le shell, et Docker Compose donne la priorité au shell sur `.env` : les conteneurs démarraient sans URL et se contentaient de journaliser. (b) Le message était au format `MessageCard` ; un workflow Teams (Power Automate) attend une Adaptive Card dans `attachments` et répond 202 même sans. | La variable n'est plus exportée par `.env.example` (seulement documentée). La charge utile porte une Adaptive Card en plus des champs `MessageCard`. Vérifié : alerte de test et incident « lot invalide » reçus par le webhook. | voir `git log` |

## Ce que ces anomalies ont en commun

- **Les problèmes 1 à 3 sont des problèmes d'intégration**, invisibles tant que
  tout tourne dans un seul processus avec un MLflow SQLite local. Ils n'apparaissent
  qu'avec la vraie topologie : hôte, conteneurs et serveur distants.
- **Les problèmes 4 et 5 sont des problèmes de volume.** Le jeu synthétique
  (quelques centaines de lignes par jour, sans valeur manquante) ne déclenche ni
  le `NaN` ni la saturation mémoire.
- **Les problèmes 6 et 7 sont des problèmes de configuration** : un même mécanisme
  de surcharge par variable d'environnement, pratique en conteneur, devient
  dangereux quand un outil l'exporte dans tout le shell.
- **Les problèmes 8 et 11 sont méthodologiques** : la règle et le pipeline
  « fonctionnaient », mais leurs décisions ne mesuraient pas ce qu'elles
  prétendaient mesurer, ou utilisaient des données que la simulation ne
  devait pas encore connaître.
- **Les problèmes 9 et 10 n'ont été vus qu'en produisant les preuves** : la
  règle existait et était testée, mais sa mise en production ne l'activait pas,
  et les délais de panne n'avaient jamais été mesurés.
