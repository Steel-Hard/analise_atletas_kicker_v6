import numpy as np
import joblib
from typing import Dict, Any, List

def run_onnx_inference(model_path: str, data: np.ndarray, features_used: List[str]) -> Dict[str, Any]:
    """
    Executa inferência em um modelo salvo via joblib (IsolationForest pipeline).
    `data` deve ser um array NumPy 2D.
    Retorna IF_Score e IF_Label.
    """
    try:
        saved = joblib.load(model_path)
        pipeline = saved['pipeline']

        X = data.astype(np.float64)

        # IsolationForest: predict retorna 1 (normal) ou -1 (anomalia)
        labels = pipeline.predict(X)

        # score_samples: valores negativos = mais anômalo
        scores = pipeline.score_samples(X)

        return {
            "labels": labels,
            "scores": scores
        }
    except Exception as e:
        import logging
        logging.getLogger("analise_atletas_api").error(f"Erro na inferencia: {e}", exc_info=True)
        return {
            "labels": np.ones(len(data), dtype=int),  # default: normal
            "scores": np.zeros(len(data))
        }

