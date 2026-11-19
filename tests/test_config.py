"""La configuration est validée au chargement : une erreur échoue tôt."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hdd_rul.common.config import AppConfig, _deep_merge, load_config


def test_charge_la_configuration_de_reference() -> None:
    config = load_config()
    assert config.horizon_days == 90
    assert "smart_5_raw" in config.features.smart_columns
    assert config.paths.bronze.name == "bronze"


def test_overlay_ne_remplace_que_les_cles_declarees() -> None:
    base = load_config()
    demo = load_config("config/demo.yaml")
    assert demo.synthetic.n_drives != base.synthetic.n_drives
    # L'overlay de démo ne touche pas aux attributs SMART : ils doivent survivre.
    assert demo.features.smart_raw == base.features.smart_raw


def test_fusion_profonde() -> None:
    merged = _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


def test_cle_inconnue_refusee() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"horizon_jours": 90})


def test_seuil_d_alerte_doit_etre_sous_l_horizon() -> None:
    with pytest.raises(ValidationError, match="alert_threshold_days"):
        AppConfig.model_validate({"horizon_days": 30, "split": {"alert_threshold_days": 30}})


def test_strategie_de_baseline_inconnue_refusee() -> None:
    with pytest.raises(ValidationError, match="baseline"):
        AppConfig.model_validate({"training": {"baseline": {"strategy": "mode"}}})
