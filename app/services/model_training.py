import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from app.core.logger import logger
from app.services.data_processing import FEATURES

ONNX_MODEL_DIR = "models_onnx"
os.makedirs(ONNX_MODEL_DIR, exist_ok=True)

def train_and_export_onnx(df: pd.DataFrame, athlete_id: str):
    """
    Treina o StandardScaler e o IsolationForest para um atleta específico
    e salva o pipeline como arquivo .pkl (joblib).
    Nota: mantemos a nomenclatura 'onnx' no diretório por compatibilidade,
    mas usamos joblib para evitar bugs de conversão do skl2onnx.
    """
    # Filtrar dados do atleta
    grupo = df[df['Athlete ID'] == athlete_id].copy()

    # Verificar colunas disponíveis
    features_usadas = [f for f in FEATURES if f in grupo.columns]

    if len(grupo) < 5 or not features_usadas:
        logger.warning(f"Atleta {athlete_id} não possui dados suficientes para treinar (N={len(grupo)}).")
        return None

    X = grupo[features_usadas].values.astype(float)

    # Criar pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(n_estimators=200, contamination=0.10, max_samples='auto', random_state=42))
    ])

    # Treinar
    pipeline.fit(X)

    # Salvar como pkl (confiável e sem bugs de conversão)
    model_path = os.path.join(ONNX_MODEL_DIR, f"atleta_{athlete_id}.pkl")
    joblib.dump({'pipeline': pipeline, 'features_used': features_usadas}, model_path)

    logger.info(f"Modelo salvo em {model_path} para o atleta {athlete_id}")
    return {
        "athlete_id": athlete_id,
        "model_path": model_path,
        "features_used": features_usadas
    }

def retrain_all_athletes(df_ws: pd.DataFrame) -> list:
    """
    Função para ser executada em background após o upload de uma nova planilha.
    Retreina os modelos para todos os atletas.
    """
    resultados = []
    atletas = df_ws['Athlete ID'].unique()
    for atleta in atletas:
        res = train_and_export_onnx(df_ws, atleta)
        if res:
            resultados.append(res)
    return resultados
