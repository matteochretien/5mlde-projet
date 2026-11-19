# Anomalie — le planificateur du serveur Prefect conteneurisé ne crée aucun run

**Constatée** au TP 6 (exécution planifiée), puis vérifiée sur la pile du projet.
**Statut** : contournée dans `compose.yml`.

## Symptôme

Le deployment est bien enregistré et son planning actif, mais le serveur ne crée
aucun run planifié. Dans les logs du serveur, le service `schedule_deployments`
plante toutes les 60 secondes :

```
sqlalchemy.exc.InvalidRequestError: Can't evaluate bulk DML statement
```

## Cause

L'image `prefecthq/prefect:3.8.7-python3.12` embarque SQLAlchemy 2.1.x, alors que
le code du planificateur de Prefect 3.8.7 n'est compatible qu'avec la série 2.0.

## Ce qui a été vérifié au TP 6

Sans toucher au conteneur commun, un run « Scheduled » créé à la main à T+60 s
a bien été exécuté à l'heure prévue par le runner `serve`. L'exécution différée
fonctionne donc ; c'est la **génération automatique** des runs par le serveur qui
ne fonctionne pas.

## Contournement retenu pour le projet

Le service `prefect` de `compose.yml` installe
`sqlalchemy[asyncio]>=2.0.40,<2.1` avant de lancer `prefect server start`. Le
correctif reste visible et commenté à côté du service, plutôt que caché dans une
image dérivée. Avec lui, la scrutation de `data/landing/` toutes les 30 s et
l'entraînement nocturne (`17 2 * * *`) produisent bien leurs runs : voir
`06_rejeu.txt`, où chaque journée est traitée par un run créé par le
planificateur.
