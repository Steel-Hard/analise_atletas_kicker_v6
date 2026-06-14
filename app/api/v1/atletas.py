from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.atleta_repository import AtletaRepository
from app.repositories.sessao_repository import SessaoRepository
from app.api.deps import get_current_user
from app.schemas.responses import AtletaResponse, SimilarityResponse
from app.services.analytics import sugerir_substitutos, construir_perfis, montar_df_features_brutas
from app.services.data_processing import FEATURES
import pandas as pd

router = APIRouter()

@router.get("/", response_model=List[AtletaResponse])
async def list_atletas(current_user: str = Depends(get_current_user)):
    atleta_repo = AtletaRepository()
    sessao_repo = SessaoRepository()
    
    atletas = await atleta_repo.get_all()
    sessoes = await sessao_repo.get_all()
    
    df = pd.DataFrame([s.model_dump() for s in sessoes])
    
    res = []
    for a in atletas:
        if not df.empty:
            df_a = df[df['athlete_id'] == a.athlete_id]
            total_sessoes = len(df_a)
            anomalies_count = int(df_a['if_anomalia'].fillna(False).sum()) if total_sessoes > 0 and 'if_anomalia' in df_a.columns else 0
            
            # Pega o status da última sessao
            if total_sessoes > 0 and 'classificacao' in df_a.columns:
                last_session = df_a.sort_values('start_date').iloc[-1]
                status = last_session['classificacao'] if not pd.isna(last_session['classificacao']) else "Desempenho Médio"
            else:
                status = "Sem dados"
        else:
            total_sessoes = 0
            anomalies_count = 0
            status = "Sem dados"
            
        res.append(AtletaResponse(
            athlete_id=a.athlete_id,
            athlete_name=a.athlete_name,
            total_sessions=total_sessoes,
            anomalies_count=anomalies_count,
            performance_status=status
        ))
    return res

@router.get("/{athlete_id}/similaridade", response_model=List[SimilarityResponse])
async def get_similaridade(athlete_id: str, top_n: int = 3, current_user: str = Depends(get_current_user)):
    sessao_repo = SessaoRepository()
    sessoes = await sessao_repo.get_all()
    
    if not sessoes:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado para calcular similaridade.")
        
    df = montar_df_features_brutas([s.model_dump() for s in sessoes])
    perfis = construir_perfis(df, FEATURES)
    
    substitutos = sugerir_substitutos(athlete_id, perfis, top_n=top_n)
    return substitutos
