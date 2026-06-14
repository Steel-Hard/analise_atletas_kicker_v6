from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AtletaResponse(BaseModel):
    athlete_id: str
    athlete_name: Optional[str] = None
    total_sessions: int
    anomalies_count: int
    performance_status: str

class RadarData(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]] # e.g., [{"label": "Atleta A", "data": [1, 2, 3]}, ...]

class SimilarityResponse(BaseModel):
    atleta_candidato: str
    distancia_euclidiana: float
    distancia_cosseno: float
    similaridade: float

class DashboardStats(BaseModel):
    total_atletas: int
    total_sessoes: int
    altas_desempenho: int
    quedas_desempenho: int
    desempenho_medio: int
    total_anomalias: int

class HistoricoSerie(BaseModel):
    datas: List[str]
    valores: List[float]
    anomalias: List[bool]
