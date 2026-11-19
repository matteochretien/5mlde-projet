"""Règle de mise en service : fonction pure, donc testable sans serveur MLflow."""

from __future__ import annotations

from hdd_rul.common.config import AppConfig
from hdd_rul.training.registry import decide_promotion


def test_premier_modele_est_promu(config: AppConfig) -> None:
    decision = decide_promotion({"mae": 10.0, "mae_lt30": 6.0}, None, config)
    assert decision.promote
    assert decision.reason == "no_champion_yet"


def test_promu_si_la_zone_critique_s_ameliore(config: AppConfig) -> None:
    decision = decide_promotion(
        {"mae": 9.9, "mae_lt30": 5.0},  # -16 % sur la zone critique
        {"mae": 10.0, "mae_lt30": 6.0},
        config,
    )
    assert decision.promote
    assert decision.reason == "ameliore_zone_critique"
    assert decision.details["mae_lt30_improvement"] > 0.05


def test_rejete_si_le_gain_est_trop_faible(config: AppConfig) -> None:
    decision = decide_promotion(
        {"mae": 10.0, "mae_lt30": 5.9},  # -1,7 %, sous le seuil de 5 %
        {"mae": 10.0, "mae_lt30": 6.0},
        config,
    )
    assert not decision.promote
    assert decision.reason.startswith("mae_lt30_gain_insuffisant")


def test_rejete_si_la_mae_globale_se_degrade_trop(config: AppConfig) -> None:
    decision = decide_promotion(
        {"mae": 10.5, "mae_lt30": 4.0},  # +5 % de MAE globale : trop
        {"mae": 10.0, "mae_lt30": 6.0},
        config,
    )
    assert not decision.promote
    assert decision.reason.startswith("mae_globale_degradee")


def test_rejete_si_la_validation_a_echoue(config: AppConfig) -> None:
    """Un modèle entraîné sur des données non conformes ne part jamais en production."""
    decision = decide_promotion(
        {"mae": 1.0, "mae_lt30": 0.5},  # métriques excellentes...
        {"mae": 10.0, "mae_lt30": 6.0},
        config,
        validation_passed=False,  # ...mais données refusées
    )
    assert not decision.promote
    assert decision.reason == "validation_failed"


def test_rejete_si_une_metrique_manque(config: AppConfig) -> None:
    decision = decide_promotion({"mae": 10.0}, {"mae": 9.0, "mae_lt30": 6.0}, config)
    assert not decision.promote
    assert "metrics_missing" in decision.reason


def test_les_seuils_sont_configurables(config: AppConfig) -> None:
    exigeant = config.model_copy(
        update={"promotion": config.promotion.model_copy(update={"min_mae_lt30_improvement": 0.5})}
    )
    decision = decide_promotion(
        {"mae": 9.0, "mae_lt30": 5.0}, {"mae": 10.0, "mae_lt30": 6.0}, exigeant
    )
    assert not decision.promote


def test_l_overlay_de_demonstration_force_la_promotion() -> None:
    """config/promotion_permissive.yaml doit se charger et promouvoir un challenger
    moins bon : c'est ce qui permet de montrer le retour arrière."""
    from hdd_rul.common.config import load_config

    permissive = load_config("config/promotion_permissive.yaml")
    decision = decide_promotion(
        {"mae": 30.0, "mae_lt30": 50.0}, {"mae": 28.0, "mae_lt30": 45.0}, permissive
    )
    assert decision.promote
    # La raison ne doit jamais présenter une régression comme une amélioration.
    assert decision.reason.startswith("regression_toleree:-0.1111")
