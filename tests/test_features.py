"""Features v1 : dérivations, échantillonnage déterministe, prétraitement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hdd_rul.common.config import AppConfig
from hdd_rul.common.hashing import stable_bucket
from hdd_rul.features.build import (
    build_feature_steps,
    derive_features,
    required_input_columns,
    select_feature_frame,
    vendor_from_model,
)


@pytest.mark.parametrize(
    ("modele", "attendu"),
    [
        ("ST12000NM0008", "Seagate"),
        ("TOSHIBA MG07ACA14TA", "Toshiba"),
        ("HGST HUH721212ALN604", "HGST"),
        ("WDC WUH721414ALE6L4", "WDC"),
        ("Hitachi HDS5C3030ALA630", "Hitachi"),
        ("MODELE INCONNU", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_constructeur_deduit_du_modele(modele: str | None, attendu: str) -> None:
    assert vendor_from_model(modele) == attendu


def test_capacite_negative_devient_manquante() -> None:
    """Backblaze publie parfois -1 : la valeur doit devenir NaN, pas -1e-12 To."""
    frame = pd.DataFrame({"model": ["ST4000DM000", "ST4000DM000"], "capacity_bytes": [4e12, -1]})
    derived = derive_features(frame)
    assert derived.loc[0, "capacity_tb"] == pytest.approx(4.0)
    assert np.isnan(derived.loc[1, "capacity_tb"])


def test_colonne_smart_absente_devient_manquante(config: AppConfig) -> None:
    """Le schéma varie d'un trimestre à l'autre : l'absence n'est pas une erreur."""
    frame = pd.DataFrame({"model": ["ST4000DM000"], "capacity_bytes": [4e12], "smart_5_raw": [3]})
    selected = select_feature_frame(frame, config)
    assert set(required_input_columns(config)).issubset(selected.columns)
    assert selected["smart_5_raw"].iloc[0] == 3
    assert selected["smart_197_raw"].isna().all()


def test_echantillonnage_des_sains_est_deterministe() -> None:
    """Deux exécutions doivent tirer le même échantillon, sans stocker de liste."""
    serials = [f"ZA{i:08d}" for i in range(2_000)]
    premier = [stable_bucket(s, salt="hdd-rul-v1") for s in serials]
    second = [stable_bucket(s, salt="hdd-rul-v1") for s in serials]
    assert premier == second
    assert all(0 <= bucket < 100 for bucket in premier)
    # Répartition raisonnablement uniforme : ~4 % des disques pour un taux de 4.
    retenus = sum(1 for bucket in premier if bucket < 4)
    assert 30 <= retenus <= 130


def test_le_sel_change_l_echantillon() -> None:
    serials = [f"ZA{i:08d}" for i in range(500)]
    a = {s for s in serials if stable_bucket(s, salt="a") < 20}
    b = {s for s in serials if stable_bucket(s, salt="b") < 20}
    assert a != b


def test_pretraitement_regroupe_les_modeles_rares(config: AppConfig) -> None:
    """top-K + modalité « autre » : un modèle inconnu ne fait pas échouer la prédiction."""
    from sklearn.pipeline import Pipeline

    petit = config.model_copy(
        update={"features": config.features.model_copy(update={"top_k_models": 2})}
    )
    frame = pd.DataFrame(
        {
            "model": ["ST4000DM000"] * 10 + ["TOSHIBA MG07ACA14TA"] * 10 + ["RARE1", "RARE2"],
            "capacity_bytes": [4e12] * 22,
            **{column: [1.0] * 22 for column in petit.features.smart_columns},
        }
    )
    pipeline = Pipeline(build_feature_steps(petit))
    encoded = pipeline.fit_transform(select_feature_frame(frame, petit))
    assert encoded.shape[0] == 22

    inconnu = pd.DataFrame(
        {
            "model": ["MODELE-JAMAIS-VU"],
            "capacity_bytes": [8e12],
            **{column: [2.0] for column in petit.features.smart_columns},
        }
    )
    # Ne doit pas lever : le modèle inconnu tombe dans la modalité regroupée.
    assert pipeline.transform(select_feature_frame(inconnu, petit)).shape[0] == 1
