from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from app.services.onnx_inference import run_onnx_inference
from app.services.data_processing import FEATURES
from app.core.logger import logger
import os
import numpy as np

router = APIRouter()

@router.post("/predizer/{athlete_id}")
async def predizer_sessao(athlete_id: str, payload: Dict[str, float]):
    """
    Recebe um JSON com as features de uma sessão e retorna se é anomalia ou não usando o modelo ONNX.
    """
    model_path = os.path.join("models_onnx", f"atleta_{athlete_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Modelo não encontrado para o atleta {athlete_id}. É necessário treinar os dados primeiro.")

    # Extrair features na ordem correta
    data_row = []
    # Usamos todas as features disponíveis no FEATURES, ou precisamos saber exatamente quais o modelo usou.
    # Como salvamos models_onnx/atleta_ID.onnx e treinamos com FEATURES disponíveis, vamos tentar com as chaves fornecidas.
    # O ideal é salvar metadados do modelo, mas assumiremos que `payload` contém as features na ordem de FEATURES
    features_usadas = []
    for f in FEATURES:
        if f in payload:
            data_row.append(payload[f])
            features_usadas.append(f)

    if not data_row:
        raise HTTPException(status_code=400, detail="Nenhuma feature válida fornecida no payload.")

    X = np.array([data_row], dtype=np.float32)

    resultado = run_onnx_inference(model_path, X, features_usadas)

    label = int(resultado['labels'][0])
    score = float(resultado['scores'][0])

    return {
        "athlete_id": athlete_id,
        "if_label": label,
        "if_score": score,
        "is_anomaly": label == -1
    }
