# Données — provenance, licence, schéma

## Source

**Backblaze Hard Drive Data (« Drive Stats »)**
<https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>

Backblaze publie chaque trimestre un instantané quotidien de l'état de son parc
de disques durs : une ligne par disque et par jour, avec les attributs SMART
remontés par le disque et un indicateur de panne.

Les archives suivent le motif :

```
https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q{n}_{yyyy}.zip
```

Une fois décompressée, chaque archive contient un CSV par journée, nommé
`AAAA-MM-JJ.csv`.

### Conditions d'utilisation

Ce que le projet respecte :

- la source est citée ici, dans le README et dans la soutenance ;
- **aucune donnée n'est redistribuée** : `data/` et `*.zip` sont exclus du dépôt
  par `.gitignore`, et le code télécharge les archives depuis leur URL d'origine ;
- l'usage est strictement pédagogique, sans revente ni rediffusion.

## Périmètre retenu

| Trimestre | Rôle dans le projet |
|---|---|
| Q4 2025 | Historique seul : fournit les 90 jours qui précèdent les pannes du début de Q1 2026. |
| Q1 2026 | Entraînement initial. |
| Q2 2026 | **Rejoué jour par jour** pour simuler l'arrivée de nouvelles données. |

Le périmètre se déclare dans `config/base.yaml` (`data.quarters`) : changer de
trimestres ne demande aucune modification de code.

## Reproductibilité : cache et empreintes

`uv run python -m hdd_rul.cli download` télécharge les trimestres déclarés,
vérifie leur `sha256` et écrit `data/raw/_manifest.json` :

```json
{
  "Q1_2026": {
    "url": "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2026.zip",
    "sha256": "cad66574292b89bae8ec8370dbb5e90d2e0b45d61341502708df328fdb15ed5c",
    "bytes": 1312921361,
    "downloaded_at": "2026-11-16T20:00:29+00:00"
  }
}
```

Les empreintes relevées au premier téléchargement sont figées dans
`config/base.yaml` sous `data.checksums`. Dès lors, tout téléchargement dont l'empreinte diffère
est **rejeté** (`ChecksumMismatchError`) : une archive corrompue ou republiée par
Backblaze ne peut plus entrer silencieusement dans le pipeline.

```yaml
data:
  checksums:
    Q1_2026: "cad66574292b89bae8ec8370dbb5e90d2e0b45d61341502708df328fdb15ed5c"
    Q2_2026: "b30be47290580c1aee9a5155afadc97c26e36220722bc5ee689108e949c74acc"
    Q4_2025: "57d0667cc03f23f16ded693e1e83e1ed28b7ea42d54b88d221734f5926030cb5"
```

La couche bronze porte sa propre empreinte (`source_fingerprint` dans
`data/bronze/bronze_manifest.json`), calculée sur les noms et tailles des
fichiers sources. Elle est reprise en tag MLflow `data.sha256`, ce qui permet de
retrouver, pour un modèle donné, le périmètre exact de données utilisé.

## Volume

| Étape | Ordre de grandeur |
|---|---|
| Archives zip | 1,2 à 2,1 Go par trimestre |
| CSV décompressés | ~10 à 12 Go par trimestre |
| Couche bronze (Parquet, après filtrage) | quelques centaines de Mo |

D'où le choix de **DuckDB** : les CSV sont lus en streaming, avec
`union_by_name = true` pour absorber les variations de schéma, et jamais
chargés intégralement en mémoire.

## Schéma

`Drive_Stats_Schema_Current.csv` (à la racine du dépôt) liste le schéma publié
par Backblaze : 11 colonnes d'identification plus environ 180 colonnes SMART
(`smart_<n>_normalized` et `smart_<n>_raw`).

### Colonnes exploitées

| Colonne | Usage |
|---|---|
| `date` | Jour de l'observation ; sert aussi au split temporel. |
| `serial_number` | Identifiant du disque ; clé de groupe anti-fuite. |
| `model` | Feature catégorielle (top-K + modalité « autre ») et source du constructeur. |
| `capacity_bytes` | Feature `capacity_tb`. Valeur `<= 0` traitée comme inconnue. |
| `failure` | `1` = le disque est tombé en panne **ce jour-là** (sa dernière ligne). |
| `smart_5_raw` | Secteurs réalloués. |
| `smart_9_raw` | Heures de fonctionnement (âge du disque). |
| `smart_12_raw` | Cycles d'alimentation. |
| `smart_187_raw` | Erreurs non corrigées signalées. |
| `smart_188_raw` | Commandes expirées. |
| `smart_193_raw` | Cycles de chargement/déchargement. |
| `smart_194_raw` | Température (°C). |
| `smart_197_raw` | Secteurs en attente de réallocation. |
| `smart_198_raw` | Secteurs incorrigibles hors ligne. |
| `smart_199_raw` | Erreurs CRC UDMA. |

La liste se règle dans `config/base.yaml` (`features.smart_raw`).

### Pièges du schéma, et ce que le projet en fait

1. **Le schéma varie d'un trimestre à l'autre** : Backblaze ajoute et retire des
   colonnes SMART. `union_by_name = true` (DuckDB) et `select_feature_frame`
   (côté Python) transforment une colonne absente en valeur manquante, que
   `HistGradientBoostingRegressor` gère nativement. Aucun plantage, aucune
   imputation inventée.
2. **Deux coquilles dans les noms de colonnes** : `smart_211_normailized` et
   `smart_212_normailized` (au lieu de `normalized`). Elles sont déclarées
   explicitement dans `validation.tolerated_typo_columns` et reproduites par le
   générateur synthétique, pour que le contrat de données ne s'étonne pas de les
   voir et ne les « corrige » pas.
3. **`capacity_bytes` vaut parfois `-1`** : règle **non bloquante**, les lignes
   partent en quarantaine (`data/quarantine/`) et sont comptées dans Grafana.
   Au-delà de 5 % du lot (`validation.max_quarantine_fraction`), la règle
   redevient bloquante : ce n'est plus du bruit, c'est un fichier abîmé.
4. **Les valeurs `normalized` sont propres à chaque constructeur** : une même
   valeur ne veut pas dire la même chose chez Seagate et chez Toshiba. Le projet
   n'utilise donc **que les valeurs `raw`**, et ajoute une feature
   `vendor` pour que le modèle puisse différencier les sémantiques.

## Jeu synthétique : pourquoi, et quelle honnêteté

Télécharger ~35 Go et le traiter prend plusieurs dizaines de minutes. Pour que
la chaîne soit testable, démontrable et exécutable en intégration continue, le
projet embarque un générateur :

```bash
uv run python -m hdd_rul.cli --config config/demo.yaml synthetic
```

Il produit des fichiers journaliers **au même format** que Backblaze (mêmes
colonnes, mêmes coquilles, même invariant « un disque disparaît après sa panne »,
mêmes sorties de parc sans panne qui créent de la censure à droite).

Ce qu'il **ne** reproduit **pas** : la distribution réelle des pannes, les
corrélations entre attributs SMART, l'hétérogénéité réelle du parc. La
dégradation y est monotone et volontairement simple. Les métriques obtenues sur
ce jeu **ne valent donc rien comme mesure de performance prédictive** ; elles ne
servent qu'à démontrer le fonctionnement de la chaîne. Les résultats présentés
comme tels sont ceux obtenus sur les données réelles.

Le pipeline, lui, ne fait aucune différence : les deux sources passent par
`data/raw/`, puis par le même contrat de qualité et le même code.
