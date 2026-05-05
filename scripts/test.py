# Pega esto en un script rápido o en el REPL
import mlflow
from src.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = mlflow.tracking.MlflowClient()

# Obtener las versiones en producción
versions = client.search_model_versions(f"name='{settings.model_registry_name}'")
prod = [v for v in versions if v.current_stage == "Production"][0]

# Cargar el modelo y ver qué features espera
model = mlflow.xgboost.load_model(f"models:/{settings.model_registry_name}/Production")
print(f"Features esperadas: {model.feature_types}")
print(f"Total: {len(model.feature_types)}")