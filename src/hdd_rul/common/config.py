"""Configuration applicative : YAML + overlay + variables d'environnement.

La configuration est volontairement séparée du code (critère « configuration
séparée » du sujet). Elle est validée au chargement par Pydantic : une clé
inconnue ou une valeur hors bornes fait échouer le démarrage, pas le run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config/base.yaml")


class _Strict(BaseModel):
    """Interdit les clés inconnues : une faute de frappe dans le YAML échoue tôt."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    root: Path = Path("data")
    artifacts: Path = Path("artifacts")

    @property
    def cache(self) -> Path:
        """Zips Backblaze téléchargés (immuables, adresses par sha256)."""
        return self.root / "cache"

    @property
    def raw(self) -> Path:
        """CSV journaliers décompressés, tels que publiés par Backblaze."""
        return self.root / "raw"

    @property
    def bronze(self) -> Path:
        """Parquet filtré : fenêtres de pannes + échantillon de disques sains."""
        return self.root / "bronze"

    @property
    def landing(self) -> Path:
        """Dépôt surveillé : l'arrivée d'un fichier déclenche `daily_batch`."""
        return self.root / "landing"

    @property
    def quarantine(self) -> Path:
        """Lignes écartées par une règle non bloquante du contrat de données."""
        return self.root / "quarantine"

    @property
    def processed(self) -> Path:
        """Lots journaliers déjà traités (déplacés depuis landing/)."""
        return self.root / "processed"

    def all_dirs(self) -> list[Path]:
        return [
            self.root,
            self.cache,
            self.raw,
            self.bronze,
            self.landing,
            self.quarantine,
            self.processed,
            self.artifacts,
        ]

    def ensure(self) -> None:
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)


class QuartersConfig(_Strict):
    history: list[str] = Field(default_factory=list)
    train: list[str] = Field(default_factory=list)
    replay: list[str] = Field(default_factory=list)

    def all(self) -> list[str]:
        """Trimestres distincts, dans l'ordre chronologique de déclaration."""
        seen: dict[str, None] = {}
        for quarter in [*self.history, *self.train, *self.replay]:
            seen.setdefault(quarter, None)
        return list(seen)


class DataConfig(_Strict):
    base_url: str = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
    quarters: QuartersConfig = QuartersConfig()
    checksums: dict[str, str] = Field(default_factory=dict)

    def archive_name(self, quarter: str) -> str:
        return f"data_{quarter}.zip"

    def archive_url(self, quarter: str) -> str:
        return f"{self.base_url.rstrip('/')}/{self.archive_name(quarter)}"


class FeaturesConfig(_Strict):
    smart_raw: list[int] = Field(
        default_factory=lambda: [5, 9, 12, 187, 188, 193, 194, 197, 198, 199]
    )
    top_k_models: int = Field(default=20, ge=1, le=500)
    other_model_label: str = "__other__"
    use_vendor: bool = True
    use_capacity_tb: bool = True

    @property
    def smart_columns(self) -> list[str]:
        return [f"smart_{attr}_raw" for attr in self.smart_raw]

    @property
    def numeric_columns(self) -> list[str]:
        columns = list(self.smart_columns)
        if self.use_capacity_tb:
            columns.append("capacity_tb")
        return columns

    @property
    def categorical_columns(self) -> list[str]:
        columns = ["model"]
        if self.use_vendor:
            columns.append("vendor")
        return columns

    @property
    def all_columns(self) -> list[str]:
        return [*self.numeric_columns, *self.categorical_columns]


class SamplingConfig(_Strict):
    healthy_rate_pct: int = Field(default=4, ge=0, le=100)
    healthy_stride_days: int = Field(default=7, ge=1)
    hash_salt: str = "hdd-rul-v1"
    seed: int = 20261105


class SplitConfig(_Strict):
    valid_fraction: float = Field(default=0.2, gt=0.0, lt=0.9)
    alert_threshold_days: int = Field(default=30, ge=1)


class ValidationConfig(_Strict):
    required_columns: list[str] = Field(
        default_factory=lambda: ["date", "serial_number", "model", "capacity_bytes", "failure"]
    )
    temperature_range: tuple[int, int] = (0, 80)
    max_daily_failure_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    quarantine_on: list[str] = Field(default_factory=lambda: ["capacity_bytes"])
    max_quarantine_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    tolerated_typo_columns: list[str] = Field(
        default_factory=lambda: ["smart_211_normailized", "smart_212_normailized"]
    )


class BaselineConfig(_Strict):
    strategy: str = "median"

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        allowed = {"mean", "median", "quantile", "constant"}
        if value not in allowed:
            raise ValueError(f"stratégie baseline inconnue: {value!r} (attendu: {sorted(allowed)})")
        return value


class HgbConfig(_Strict):
    max_iter: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=0.08, gt=0.0)
    max_depth: int | None = 8
    min_samples_leaf: int = Field(default=40, ge=1)
    l2_regularization: float = Field(default=0.5, ge=0.0)
    early_stopping: bool = True
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    random_state: int = 20261105


class TrainingConfig(_Strict):
    baseline: BaselineConfig = BaselineConfig()
    hgb: HgbConfig = HgbConfig()


class MlflowConfig(_Strict):
    experiment_name: str = "hdd-rul"
    registered_model_name: str = "hdd-rul-hgb"
    tracking_uri: str | None = None

    def resolved_tracking_uri(self) -> str:
        """Priorité : config explicite > MLFLOW_TRACKING_URI > serveur MLflow
        de la pile Compose, vu depuis l'hôte (port 5050)."""
        if self.tracking_uri:
            return self.tracking_uri
        return os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050"


class PromotionConfig(_Strict):
    # Négatif = tolérer une régression : réservé à la démonstration du retour
    # arrière (config/promotion_permissive.yaml), ou il faut forcer une promotion.
    min_mae_lt30_improvement: float = Field(default=0.05, ge=-1.0, le=1.0)
    max_mae_global_degradation: float = Field(default=0.02, ge=0.0, le=1.0)
    require_validation_passed: bool = True


class AlertLinksConfig(_Strict):
    """Adresses des interfaces, **vues depuis le navigateur** du destinataire.

    Ce ne sont pas celles qu'utilise le code : dans les conteneurs, MLflow est
    `http://mlflow:5000`, un lien inutilisable depuis Teams. Chaque valeur peut
    être surchargée par une variable d'environnement (`HDD_RUL_*_URL`).
    """

    prefect_ui: str = "http://localhost:4200"
    mlflow_ui: str = "http://localhost:5050"
    grafana: str = "http://localhost:3000"
    grafana_dashboard_uid: str = "hdd-rul-main"

    @staticmethod
    def _base(value: str, env_var: str) -> str:
        return (os.environ.get(env_var) or value).rstrip("/")

    def prefect_runs(self) -> str:
        return f"{self._base(self.prefect_ui, 'HDD_RUL_PREFECT_UI_URL')}/runs"

    def prefect_flow_run(self, flow_run_id: str) -> str:
        return (
            f"{self._base(self.prefect_ui, 'HDD_RUL_PREFECT_UI_URL')}/runs/flow-run/{flow_run_id}"
        )

    def mlflow_run(self, run_id: str) -> str:
        return f"{self._base(self.mlflow_ui, 'HDD_RUL_MLFLOW_UI_URL')}/#/runs/{run_id}"

    def mlflow_model_version(self, model_name: str, version: str) -> str:
        base = self._base(self.mlflow_ui, "HDD_RUL_MLFLOW_UI_URL")
        return f"{base}/#/models/{model_name}/versions/{version}"

    def grafana_dashboard(self) -> str:
        base = self._base(self.grafana, "HDD_RUL_GRAFANA_URL")
        return f"{base}/d/{self.grafana_dashboard_uid}"

    def all_local(self) -> bool:
        """Vrai si tous les liens pointent sur la machine qui héberge la pile."""
        bases = [
            self._base(self.prefect_ui, "HDD_RUL_PREFECT_UI_URL"),
            self._base(self.mlflow_ui, "HDD_RUL_MLFLOW_UI_URL"),
            self._base(self.grafana, "HDD_RUL_GRAFANA_URL"),
        ]
        return all("//localhost" in b or "//127.0.0.1" in b for b in bases)


class AlertsConfig(_Strict):
    webhook_url: str | None = None
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    # Préfixe du titre : distingue ce projet des autres alertes du même canal.
    tag: str = "hdd-rul"
    # Où tourne le pipeline. null => HDD_RUL_ALERT_ENV, sinon utilisateur@machine.
    environment: str | None = None
    links: AlertLinksConfig = AlertLinksConfig()

    def resolved_webhook_url(self) -> str | None:
        return self.webhook_url or os.environ.get("HDD_RUL_WEBHOOK_URL") or None

    def resolved_environment(self) -> str:
        if self.environment:
            return self.environment
        if env := os.environ.get("HDD_RUL_ALERT_ENV"):
            return env
        import getpass
        import socket

        try:
            user = getpass.getuser()
        except Exception:  # pragma: no cover - utilisateur sans entrée passwd (conteneur)
            user = "inconnu"
        return f"{user}@{socket.gethostname()}"


class SyntheticConfig(_Strict):
    n_drives: int = Field(default=1200, ge=10)
    start_date: str = "2025-10-01"
    end_date: str = "2026-06-30"
    daily_failure_rate: float = Field(default=0.0004, gt=0.0, lt=0.1)
    seed: int = 20261105


class AppConfig(_Strict):
    """Racine de la configuration, injectée dans toutes les étapes du pipeline."""

    horizon_days: int = Field(default=90, ge=7, le=365)
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    sampling: SamplingConfig = SamplingConfig()
    split: SplitConfig = SplitConfig()
    validation: ValidationConfig = ValidationConfig()
    training: TrainingConfig = TrainingConfig()
    mlflow: MlflowConfig = MlflowConfig()
    promotion: PromotionConfig = PromotionConfig()
    alerts: AlertsConfig = AlertsConfig()
    synthetic: SyntheticConfig = SyntheticConfig()
    source_path: Path | None = None

    @model_validator(mode="after")
    def _check_alert_threshold(self) -> AppConfig:
        if self.split.alert_threshold_days >= self.horizon_days:
            raise ValueError(
                "split.alert_threshold_days doit être strictement inférieur à horizon_days "
                f"({self.split.alert_threshold_days} >= {self.horizon_days})"
            )
        return self


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : l'overlay ne remplace que les clés qu'il déclare."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"fichier de configuration introuvable: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"configuration invalide (objet attendu à la racine): {path}")
    return content


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Charge `config/base.yaml`, applique l'overlay puis les variables d'env.

    Ordre de priorité croissant : base.yaml < overlay (--config) < variables
    d'environnement < `overrides` passes par le code appelant.
    """
    overlay_path = Path(path) if path else None
    env_path = os.environ.get("HDD_RUL_CONFIG")
    if overlay_path is None and env_path:
        overlay_path = Path(env_path)

    raw = _read_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    if overlay_path and overlay_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        raw = _deep_merge(raw, _read_yaml(overlay_path))
        raw["source_path"] = str(overlay_path)
    elif DEFAULT_CONFIG_PATH.exists():
        raw["source_path"] = str(DEFAULT_CONFIG_PATH)

    if data_dir := os.environ.get("HDD_RUL_DATA_DIR"):
        raw = _deep_merge(raw, {"paths": {"root": data_dir}})
    if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        raw = _deep_merge(raw, {"mlflow": {"tracking_uri": tracking_uri}})
    if experiment := os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        raw = _deep_merge(raw, {"mlflow": {"experiment_name": experiment}})
    if webhook := os.environ.get("HDD_RUL_WEBHOOK_URL"):
        raw = _deep_merge(raw, {"alerts": {"webhook_url": webhook}})

    if overrides:
        raw = _deep_merge(raw, overrides)

    return AppConfig.model_validate(raw)
