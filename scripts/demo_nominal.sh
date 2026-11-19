#!/usr/bin/env bash
# Démonstration du cycle nominal, de bout en bout, sur la pile Compose démarrée.
#   bash scripts/demo_nominal.sh [chemin/config.yaml]
#
# Enchaînement : données -> couche bronze -> entraînement (champion) -> rejeu de
# quelques jours (validation, scoring, feedbacks) -> réentraînement et décision
# de mise en service -> retour arrière.
set -euo pipefail

CONFIG="${1:-config/demo.yaml}"
RUN=(uv run python)
export HDD_RUL_CONFIG="$CONFIG"

titre() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

titre "1/6 — Jeu de données Drive Stats (synthétique, reproductible)"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" synthetic

titre "2/6 — Ingestion : couche bronze DuckDB en deux passes"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" ingest

titre "3/6 — Entraînement initial : baseline + HGB, publication du champion"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" train --trigger demo

titre "4/6 — Version servie par l'API"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" model

titre "5/6 — Rejeu de 3 jours : validation, scoring, feedbacks automatiques"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" replay --limit 3

titre "6/6 — Réentraînement sur une fenêtre plus récente et décision"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" train --trigger demo --as-of 2026-05-01

titre "Retour arrière (champion <- previous)"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" rollback || {
  cat <<'MSG'
Aucune version `previous` : le challenger a été REJETÉ par la règle de mise en
service, donc le champion n'a pas changé — et il n'y a rien à annuler. C'est un
résultat valide de la démonstration, pas une erreur.

Pour montrer le retour arrière, il faut d'abord une promotion. L'overlay de
démonstration tolère une régression (aucune modification de code) :

  uv run python -m hdd_rul.cli --config config/promotion_permissive.yaml train
  uv run python -m hdd_rul.cli --config CONFIG_ICI rollback
MSG
}

titre "Termine"
cat <<'MSG'
Runs et versions du modèle de démonstration : http://localhost:5050
Exécutions des flows : http://localhost:4200/runs
MSG
