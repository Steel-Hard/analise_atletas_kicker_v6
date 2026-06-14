from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class Atleta(BaseModel):
    athlete_id: str
    athlete_name: Optional[str] = None
    athlete_position: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SessaoFisica(BaseModel):
    athlete_id: str
    start_date: datetime
    segment_name: str
    
    # Features base
    duration_mins: Optional[float] = None
    workload: Optional[float] = None
    distance_m: Optional[float] = None
    metres_per_minute: Optional[float] = None
    high_intensity_running_m: Optional[float] = None
    sprint_distance_m: Optional[float] = None
    top_speed_kph: Optional[float] = None
    avg_speed_kph: Optional[float] = None
    accelerations: Optional[float] = None
    decelerations: Optional[float] = None
    no_of_sprints: Optional[float] = None
    
    # Resultados analiticos (calculados/preditos)
    if_score: Optional[float] = None
    if_anomalia: Optional[bool] = None
    z_composto: Optional[float] = None
    classificacao: Optional[str] = None
    
    # Z-scores individuais
    z_scores: Dict[str, float] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class OnnxModelMetadata(BaseModel):
    model_config = {'protected_namespaces': ()}
    athlete_id: str
    model_path: str
    scaler_path: str # Embora tudo possa estar num único ONNX via pipeline, se estivermos usando skl2onnx pode ser um só, ou dois se separados. Pipeline scikit-learn vira 1 onnx.
    features_used: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
