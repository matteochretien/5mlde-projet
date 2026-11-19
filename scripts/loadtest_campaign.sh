#!/usr/bin/env bash
# Campagne de charge complète : balayage de concurrence pour 1, 2 et 4 workers,
# sur les trois tailles de lot. Produit les CSV lus par
# `scripts/collect_loadtest_results.py`.
#
#   bash scripts/loadtest_campaign.sh [config] [durée_palier]
#
# Protocole appliqué (cf. loadtest/PROTOCOLE.md) :
#   * mise en chauffe jusqu'à ce que *chaque* worker ait servi une requête ;
#   * un palier de rodage, écarté, avant chaque série ;
#   * un CSV par palier, pour pouvoir tracer la courbe latence/débit.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config/demo.yaml}"
DUREE="${2:-20s}"
# 8011 : l'API de la pile Compose occupe déjà 8001.
PORT="${PORT:-8011}"
OUT=loadtest/results
mkdir -p "$OUT"

export HDD_RUL_CONFIG="$CONFIG"
export HDD_RUL_TRACING_ENABLED="${HDD_RUL_TRACING_ENABLED:-0}"
# Comme en production (compose.yml, service `api`) : un thread OpenMP par
# prédiction. Sans ce réglage, le débit plafonne vers 30 req/s quel que soit le
# nombre de workers (contention des threads de HistGradientBoosting).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MLFLOW_DISABLE_AGENT_HINT=1

PAYLOAD=@loadtest/exemple_observation.json

demarrer_api() {
  local workers=$1
  uv run uvicorn api.main:app --host 127.0.0.1 --port "$PORT" \
      --workers "$workers" --log-level warning >"$OUT/uvicorn_w${workers}.log" 2>&1 &
  API_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  # Chaque worker charge sa propre copie du modèle. `/health` répond dès qu'UN
  # worker est prêt : sans cette attente, les premières requêtes du palier sont
  # servies par des workers encore froids et le p95 est inexploitable.
  sleep $((2 * workers + 4))
  for _ in $(seq 1 $((40 * workers))); do
    curl -s -X POST "localhost:$PORT/predict" \
        -H 'Content-Type: application/json' -d "$PAYLOAD" >/dev/null
  done
}

arreter_api() {
  kill "${API_PID:-0}" 2>/dev/null
  wait "${API_PID:-0}" 2>/dev/null
}

palier() {  # $1 classe  $2 utilisateurs  $3 étiquette
  local klass=$1 users=$2 tag=$3
  # Locust sur plusieurs processus : un seul processus (client `requests`)
  # plafonne vers 500-600 req/s et mesurerait le client, pas l'API.
  uv run locust -f loadtest/locustfile.py --headless -u "$users" -r "$users" -t "$DUREE" \
      --processes "${LOCUST_PROCESSES:-4}" \
      --host "http://127.0.0.1:$PORT" --class-picker --only-summary \
      --csv "$OUT/${tag}" "$klass" >"$OUT/locust_${tag}.log" 2>&1
}

serie() {  # $1 workers  $2 classe  $3.. utilisateurs (étiquette : SUFFIXE)
  local workers=$1 klass=$2
  shift 2
  local label="${klass}${SUFFIXE:-}"
  echo ">>> $label — $workers worker(s) — concurrences : $*"
  demarrer_api "$workers"
  # Rodage à la concurrence MAXIMALE de la série : tous les workers reçoivent du
  # trafic et chargent leur modèle avant la première mesure. Un rodage à la
  # concurrence minimale laissait des workers froids (mesures du 5 novembre :
  # 4 workers et 2 utilisateurs à 1 req/s et 2 s de latence).
  local max_users="${*: -1}"
  palier "$klass" "$max_users" "rodage_${label}_w${workers}"   # écarté volontairement
  for users in "$@"; do
    palier "$klass" "$users" "sw_${label}_w${workers}_u${users}"
    echo "    concurrence $users : fait"
  done
  arreter_api
}

trap arreter_api EXIT

serie 1 PredictionUnitaire 1 2 4 8 16
serie 2 PredictionUnitaire 2 4 8 16
serie 4 PredictionUnitaire 4 8 16 32
serie 8 PredictionUnitaire 8 16 32 64
serie 1 PredictionLot100   1 2 4
serie 2 PredictionLot100   2 4 8
serie 4 PredictionLot100   2 4 8 16
serie 8 PredictionLot100   4 8 16
serie 1 PredictionLot1000  1 2
serie 4 PredictionLot1000  1 2 4
serie 4 TraficRealiste     8 16 32

# Coût de la persistance : même série, mais chaque prédiction est écrite dans
# PostgreSQL (traces activées). Nécessite la pile Compose (PostgreSQL sur 5432).
if [ "${AVEC_PERSISTANCE:-0}" = "1" ]; then
  HDD_RUL_TRACING_ENABLED=1 LOADTEST_PERSIST=1 SUFFIXE=Persist \
    serie 2 PredictionLot100 2 4 8
fi

echo
echo "Campagne terminée. Tableau de synthèse :"
uv run python scripts/collect_loadtest_results.py "$OUT"
