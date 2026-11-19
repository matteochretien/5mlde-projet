# Preuves d'intégration

Ce dossier rassemble les traces d'exécution présentées à la soutenance : ce qui
a réellement tourné, pas ce qui est censé tourner. Toutes ont été produites le
18 novembre 2026 sur la pile Compose et les données Backblaze réelles.

Convention de nommage : `NN_sujet.<ext>`, dans l'ordre de la démonstration.

| Fichier | Contenu | Comment le produire |
|---|---|---|
| `01_setup.txt` | Installation dans un clone vierge, lint, tests | `mise run setup && mise run lint && mise run test` |
| `02_ingestion.txt` | Archives en cache revérifiées (sha256), couche bronze | `mise run download && mise run ingest` |
| `03_entrainement_initial.txt` | Entraînement au 1er avril, champion v1, ce que garde le registre | `uv run python -m hdd_rul.cli train --as-of 2026-04-01` |
| `04a_mlflow_run.png` | Run de la v1 : métriques, paramètres, tags `data.*`, `git.commit`, `trigger` | <http://localhost:5050>, expérience `hdd-rul` |
| `04b_mlflow_artefacts.png` | Artefacts du run : manifeste des données, features, résidus | idem, onglet Artifacts |
| `05a_registry_versions.png` | Les 6 versions et les alias `champion`, `challenger`, `previous` | MLflow, Models, `hdd-rul-hgb` |
| `05b_version_rejetee.png` | v4 rejetée : `promotion.reason`, `promotion.champion_eval = same_validation` | idem, Version 4 |
| `05c_version_retiree.png` | v5 promue par l'overlay puis retirée : `rolled_back`, `regression_toleree` | idem, Version 5 |
| `06_rejeu.txt` | Rejeu jour par jour par le runner : validation, scoring, feedbacks | `mise run replay` (dépôt seul, le runner traite) |
| `07a_prefect_runs.png` | Les runs de la journée (réussis, en échec, annulés par la limite de concurrence) | <http://localhost:4200>, Runs |
| `07b_prefect_entrainement_parallele.png` | `train_pipeline` : deux tâches `training` en parallèle, retries pendant la panne de MLflow | idem, run `slick-foxhound` |
| `07c_prefect_lot_refuse.png` | `daily_batch` refusé par le contrat : `DataContractError`, rien en aval | idem, run `humorous-boa` |
| `07d_prefect_deployments.png` | Déclencheurs automatiques : scrutation toutes les 30 s, cron nocturne | idem, Deployments |
| `08_promotion_rejetee.txt` | Challenger rejeté avec sa raison chiffrée | `uv run python -m hdd_rul.cli train --as-of 2026-05-15` |
| `09_incident_lot_invalide.txt` | Contrat violé, alerte, flow arrêté ; quarantaine ; non-résurrection | `uv run python scripts/faults.py lot-invalide` |
| `10_incident_dependance.txt` | MLflow arrêté : `/health` dégradé, 503 rapide, échec propre et alerte | `docker compose stop mlflow` |
| `11_rollback.txt` | Promotion forcée, retour arrière, rechargement de l'API | `mise run rollback && curl -X POST localhost:8001/admin/reload` |
| `12_grafana_technique.png` | Vue technique | <http://localhost:3000>, plage « Today » |
| `13_grafana_metier.png` | Vue métier | idem |
| `14_grafana_croise.png` | Vue croisée validation / production (v1 : 29,0 j contre 53,3 j) | idem |
| `15_loadtest.md` | Campagne Locust, coût de calcul, dimensionnement | `mise run loadtest:campaign` (cf. `loadtest/PROTOCOLE.md`) |

Anomalies rencontrées et contournées (à signaler à l'enseignant) :

- `00_anomalies_donnees_reelles.md` — les douze problèmes du premier passage sur
  données réelles (18 novembre) : symptôme, cause, correctif et commit.

- `00_anomalie_fastapi_prefect.md` — FastAPI 0.155 supprime `min_items`, encore
  utilisé par le serveur Prefect 3.8 : plus aucun serveur Prefect ne démarre.
  Contournement : `fastapi < 0.155` dans `pyproject.toml`.
- `00_anomalie_scheduler_prefect.md` — l'image `prefecthq/prefect:3.8.7` embarque
  SQLAlchemy 2.1, incompatible avec le planificateur (`schedule_deployments`
  plante toutes les 60 s sur « Can't evaluate bulk DML statement »).
  Contournement : le conteneur réinstalle `sqlalchemy < 2.1` avant de démarrer.
