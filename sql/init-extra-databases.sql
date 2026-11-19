-- Bases dédiées aux backends MLflow et Prefect, créées au premier démarrage
-- du conteneur PostgreSQL. Le registre de modèles MLflow exige un backend
-- relationnel : un store fichier ne gère ni les versions ni les alias.
SELECT 'CREATE DATABASE mlflow'  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
SELECT 'CREATE DATABASE prefect' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect')\gexec
