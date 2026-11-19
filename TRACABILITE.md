# Traçabilité — stratégie de tags, d'alias et de versions

Ce document répond à trois questions, pour n'importe quel modèle déjà parti en
production : **sur quelles données** a-t-il été entraîné, **avec quel code**, et
**pourquoi** a-t-il été mis (ou non) en service.

Un tag n'est posé que s'il sert à répondre à l'une de ces questions, ou à
retrouver une exécution. Tout le reste va dans les paramètres ou les métriques.

## 1. Pourquoi des tags, et pas seulement des paramètres

| Mécanisme MLflow | Ce qu'on y met | Pourquoi |
|---|---|---|
| **Paramètres** (`log_params`) | Hyperparamètres, horizon, taux d'échantillonnage, bornes du split | Valeurs d'entrée de l'entraînement : elles expliquent *comment* le modèle a été obtenu. |
| **Métriques** (`log_metrics`) | MAE, MAE < 30 j, RMSE, précision et rappel de l'alerte | Valeurs mesurées : elles servent à *décider*. |
| **Tags de run** | Provenance des données, commit, déclencheur, identifiant de flow | Ils servent à **chercher** : « tous les runs entraînés sur la fenêtre de mars », « tous les runs du commit abc123 ». Un paramètre ne se cherche pas aussi bien dans l'interface. |
| **Tags de version** de modèle | Statut de validation, décision et raison de promotion | Ils portent sur **l'objet servi**, pas sur l'expérience. On veut les lire sans ouvrir le run. |
| **Alias** | `champion`, `challenger`, `previous` | Ils désignent un rôle, pas un numéro : l'API ne connaît que `champion`. |

## 2. Tags de run

| Tag | Exemple | À quoi il sert |
|---|---|---|
| `data.window_start` | `2026-01-01` | Début de la fenêtre d'entraînement réellement utilisée. |
| `data.window_end` | `2026-03-31` | Fin de cette fenêtre. Avec le précédent, c'est la réponse à « sur quelles données ? ». |
| `data.as_of` | `2026-04-15` | Date simulée du run : seules les lignes dont le label était résolu à cette date ont servi. C'est ce qui fait **varier la période d'entraînement d'un run à l'autre** sur un jeu historique figé. |
| `data.sha256` | `8ce9aa54…` | Empreinte du périmètre source (noms et tailles des fichiers). Deux runs avec la même empreinte ont vu les mêmes données. |
| `data.source_files` | `182` | Nombre de fichiers journaliers ingérés : détecte un périmètre tronqué. |
| `data.horizon_days` | `90` | Un modèle entraîné avec H = 90 n'est pas comparable à un modèle H = 30. |
| `git.commit` | `a1b2c3d` | Réponse à « avec quel code ? ». |
| `trigger` | `manual`, `schedule`, `replay`, `demo` | D'où vient ce run : à la main, par planification, ou par le rejeu. |
| `prefect.flow_run_id` | `5e190d8d-…` | Lien direct vers l'exécution Prefect, et donc vers ses logs et ses retries. |
| `training.group_id` | `951f5d87…` | **Relie la baseline et le candidat d'un même entraînement.** Sans ce tag, les deux runs seraient orphelins. |
| `training.role` | `baseline` / `candidate` | Distingue les deux au sein du groupe. |
| `validation.status` | `passed` / `failed` | Statut du contrat de qualité au moment de l'entraînement. |
| `config.source` | `config/base.yaml` | Quel fichier de configuration était actif. |
| `eval.mae`, `eval.mae_lt30` | `6.6483` | Duplique les métriques clés en tags, uniquement pour pouvoir les lire dans la liste des runs sans ouvrir chaque run. |
| `split.cutoff_date`, `split.rows_*` | `2026-03-02` | Forme du split, utile pour comprendre une MAE surprenante. |

### Pourquoi des runs frères plutôt que des runs imbriques

La baseline et le candidat sont deux runs **de premier niveau**, reliés par
`training.group_id`, et non un run parent avec deux runs imbriques.

Raison : les deux entraînements tournent **en parallèle** dans des tâches
Prefect concurrentes, et l'état « run actif » de MLflow est global au processus.
Des runs imbriques lancés depuis plusieurs threads se mélangeraient — un
artefact finirait dans le mauvais run. Le tag de groupe donne le même
regroupement, sans ce risque.

La comparaison à la baseline est tout de même journalisée **sur le run du
candidat** (`baseline_mae`, `baseline_mae_lt30`, `gain_mae_vs_baseline`) : c'est
la que la question « ce modèle apporte-t-il quelque chose ? » se pose.

## 3. Tags de version de modèle

| Tag | Valeurs | À quoi il sert |
|---|---|---|
| `validation.status` | `passed` / `failed` | Un modèle entraîné sur des données non conformes ne doit jamais être promu, même si ses métriques sont bonnes. |
| `promotion.decision` | `promoted`, `rejected`, `rolled_back` | Ce qui a été décidé pour cette version. |
| `promotion.reason` | `no_champion_yet`, `ameliore_zone_critique`, `mae_lt30_gain_insuffisant:0.0149<0.05`, `mae_globale_degradee:…`, `validation_failed`, `metrics_missing:…` | **Pourquoi**. La raison est enregistrée avec les valeurs : on n'a pas à relire les logs pour comprendre un rejet. |
| `eval.mae`, `eval.mae_lt30` | `6.6483` | Permet de comparer les versions dans la liste du registre. |
| `data.window_end` | `2026-05-31` | Jusqu'ou vont les données qu'à vues cette version. |
| `trigger` | `schedule`, `replay`, … | Ce qui a déclenché l'entraînement de cette version. |

## 4. Alias et cycle de vie

```
            entrainement
                 |
                 v
          nouvelle version  ──────────────► alias `challenger`  (toujours)
                 |
        regle de mise en service
           /            \
      promu            rejete
        |                 \
        v                  `-- reste `challenger`, avec la raison en tag
  ancien champion ──► `previous`
  nouvelle version ──► `champion`  ◄── seul alias lu par l'API
```

- **`champion`** : la version servie. L'API charge `models:/<nom>@champion`, donc
  changer de champion ne demande aucun changement de configuration de l'API.
- **`challenger`** : le dernier candidat, promu ou non. Un candidat rejeté reste
  ainsi identifiable, avec sa raison de rejet.
- **`previous`** : l'ancien champion. C'est ce qui rend le retour arrière
  immédiat : pas besoin de retrouver un numéro de version à la main.

### Règle de mise en service

Le challenger est promu si, et seulement si :

1. le contrat de qualité est au vert (`validation.status = passed`) ;
2. sa **MAE sur la zone critique (RUL réelle < 30 j)** s'améliore d'au moins
   **5 %** par rapport au champion ;
3. sa **MAE globale** ne se dégrade pas de plus de **2 %**.

Pourquoi ces deux métriques ensemble : la zone critique est celle qui compte en
exploitation (se tromper de 10 jours sur un disque qui lâche dans 5 jours a des
conséquences ; sur un disque qui a 3 mois à vivre, non). Mais optimiser cette
seule zone laisserait passer un modèle qui s'effondre partout ailleurs ; d'où le
garde-fou sur la MAE globale.

**Comparaison sur les mêmes lignes.** Avant d'appliquer la règle, le champion
est rechargé depuis le registre et **réévalué sur le jeu de validation du
challenger**. Comparer le challenger aux métriques *stockées* du champion
reviendrait à comparer deux modèles sur deux fenêtres et deux échantillons de
disques différents : l'écart mélangerait qualité du modèle et difficulté du jeu.
Les valeurs retenues sont tracées sur la version candidate
(`promotion.champion_eval = same_validation`, `promotion.champion_version`,
`promotion.champion_mae`, `promotion.champion_mae_lt30`). Si le champion ne peut
pas être rechargé, la règle se rabat sur ses métriques stockées et le tag vaut
`stored_metrics`.

S'il n'y a pas encore de champion, la première version est promue
(`no_champion_yet`).

Les seuils sont dans `config/base.yaml` (`promotion.*`) : les durcir ne demande
aucune modification de code. La décision elle-même est une **fonction pure**
(`decide_promotion`), testée unitairement sur chaque cas de figure.

### Retour arrière

```bash
uv run python -m hdd_rul.cli rollback      # champion <-> previous
curl -X POST localhost:8001/admin/reload   # l'API recharge sans redemarrer
curl -s localhost:8001/model               # la nouvelle version servie est visible
```

L'échange est volontaire : après un rollback, `previous` pointe sur la version
retirée, donc un second appel annule le premier. La version retirée est taguée
`promotion.decision = rolled_back`.

## 5. Artefacts attachés à chaque run

| Artefact | Contenu | Pourquoi il est indispensable |
|---|---|---|
| `pipeline/` | **Prétraitement + modèle** dans un seul `Pipeline` sklearn | Le prétraitement est versionné avec le modèle. L'API applique exactement les mêmes transformations que l'entraînement, sans code dupliqué : c'est ce qui évite le décalage entraînement/service. |
| `rapports/features.json` | Colonnes d'entrée, features construites, nom du label | Permet de savoir ce que le modèle attend, sans le désérialiser. |
| `rapports/data_manifest.json` | Manifeste bronze, fenêtre, statistiques de labels, forme du split | Reconstitution complète du jeu d'entraînement. |
| `rapports/residus_rul.png` | Prédit vs réel, et résidus en fonction de la RUL | Lecture immédiate des biais (ex. sur-estimation systématique des RUL courtes). |
| `rapports/<lot>_<horodatage>.json` | Rapport du contrat de qualité | Le modèle voyage avec la preuve que ses données étaient conformes. |

## 6. Traces en base, et pourquoi elles doublent MLflow

MLflow dit ce que le modèle valait **à l'entraînement**. PostgreSQL dit ce qu'il
vaut **en production** :

- `tracing.model_versions` — versions vues en production, avec leurs métriques
  d'évaluation ;
- `tracing.predictions` — chaque prédiction servie, avec la version qui l'a
  produite ;
- `tracing.feedbacks` — la RUL réellement constatée ;
- `tracing.pipeline_runs` — statut de chaque exécution de flow.

La vue `tracing.v_production_error` croisé les deux : elle donne la MAE réelle
par version de modèle. Le panneau Grafana « MAE de validation vs MAE de
production » est ce qui permet de voir une dérive — un modèle dont la MAE
annoncée est bonne mais dont la MAE réelle se dégrade — et donc de déclencher un
retour arrière en connaissance de cause.
