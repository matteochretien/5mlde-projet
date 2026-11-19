# Anomalie — FastAPI 0.155 empêche tout serveur Prefect 3.8 de démarrer

**Constatée** pendant les TPs, puis à la mise en place du projet. **Statut** :
contournée, version épinglée.

## Symptôme

Aucun serveur Prefect ne démarre, ni le serveur éphémère des tests, ni le serveur
conteneurisé :

```
TypeError: Body() got an unexpected keyword argument 'min_items'
```

## Cause

FastAPI 0.155 supprime le paramètre déprécié `min_items` de `Body()`. Prefect
3.8 l'utilise encore, dans `prefect/server/api/concurrency_limits_v2.py`. Une
dichotomie sur les versions situe la rupture : FastAPI 0.154 fonctionne, la 0.155
non.

## Contournement

`pyproject.toml` déclare `fastapi>=0.115,<0.155`, avec le motif en commentaire
à côté de la contrainte, et `uv.lock` fige la version résolue. La contrainte est
à lever quand une version de Prefect compatible sera publiée.

## Ce qu'on en retient

Une dépendance transitive commune (ici FastAPI, utilisée à la fois par l'API du
projet et par le serveur Prefect) peut casser un outil d'infrastructure sans que
le code du projet change. D'où le fichier de verrouillage et des installations
qui ne sortent jamais de `uv sync --frozen`.
