#!/usr/bin/env bash
# Démonstration des incidents contrôles.
#   bash scripts/demo_incident.sh [chemin/config.yaml]
#
# Incident 1 : lot de données invalide -> le contrat bloque, alerte, aucun
#              modèle publié, l'API continue de servir le champion.
# Incident 2 : règle non bloquante -> les lignes fautives partent en quarantaine
#              et le lot continue.
# Incident 3 : dépendance indisponible (Compose) -> retries, échec propre.
set -euo pipefail

CONFIG="${1:-config/demo.yaml}"
RUN=(uv run python)
export HDD_RUL_CONFIG="$CONFIG"

titre() { printf '\n\033[1;33m=== %s ===\033[0m\n' "$1"; }

titre "Incident 1 — lot de données invalide"
LOT="$("${RUN[@]}" scripts/faults.py lot-invalide --config "$CONFIG" --clean-landing | tail -1)"
echo "lot injecté : $LOT"
echo "--- le flow doit échouer (DataContractError) et alerter ---"
if "${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" batch --batch "$LOT"; then
  echo "ECHEC DE LA DEMONSTRATION : le lot invalide est passé !" >&2
  exit 1
fi
echo "OK : le lot a été refusé, le pipeline s'est arrêté."
echo "--- l'API sert toujours le même champion ---"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" model | head -20

titre "Nettoyage du lot refusé"
rm -f "$LOT"

titre "Incident 2 — lignes non conformes mises en quarantaine"
LOT2="$("${RUN[@]}" scripts/faults.py lot-quarantaine --config "$CONFIG" --clean-landing | tail -1)"
"${RUN[@]}" -m hdd_rul.cli --config "$CONFIG" batch --batch "$LOT2"
echo "OK : le lot est passé, les lignes fautives sont en quarantaine."

titre "Incident 3 — dépendance indisponible (nécessite Docker Compose)"
cat <<'MSG'
  docker compose stop mlflow      # ou: uv run python scripts/faults.py dépendance --service mlflow
  curl -s localhost:8001/health   # -> status degraded (registry injoignable), prédictions servies
  uv run python -m hdd_rul.cli train   # -> retries puis échec propre avec alerte
  docker compose start mlflow     # ou: uv run python scripts/faults.py rétablir --service mlflow
MSG
