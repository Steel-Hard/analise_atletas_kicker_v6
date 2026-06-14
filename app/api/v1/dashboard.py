from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.repositories.sessao_repository import SessaoRepository
from app.repositories.atleta_repository import AtletaRepository
from app.schemas.responses import DashboardStats, RadarData, HistoricoSerie
from app.schemas.requests import CompareRequest
from app.services.data_processing import FEATURES
from app.services.analytics import construir_perfis, montar_df_features_brutas
import pandas as pd

router = APIRouter()

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: str = Depends(get_current_user)):
    sessao_repo = SessaoRepository()
    atleta_repo = AtletaRepository()
    
    atletas = await atleta_repo.get_all()
    sessoes = await sessao_repo.get_all()
    
    total_atletas = len(atletas)
    total_sessoes = len(sessoes)
    
    df = pd.DataFrame([s.model_dump() for s in sessoes])
    
    if df.empty:
        return DashboardStats(
            total_atletas=total_atletas,
            total_sessoes=total_sessoes,
            altas_desempenho=0,
            quedas_desempenho=0,
            desempenho_medio=0,
            total_anomalias=0
        )
        
    altas = int((df['classificacao'] == 'Alta de Desempenho').sum()) if 'classificacao' in df.columns else 0
    quedas = int((df['classificacao'] == 'Queda de Desempenho').sum()) if 'classificacao' in df.columns else 0
    medias = int((df['classificacao'] == 'Desempenho Médio').sum()) if 'classificacao' in df.columns else 0
    anomalias = int(df['if_anomalia'].fillna(False).sum()) if 'if_anomalia' in df.columns else 0
    
    return DashboardStats(
        total_atletas=total_atletas,
        total_sessoes=total_sessoes,
        altas_desempenho=altas,
        quedas_desempenho=quedas,
        desempenho_medio=medias,
        total_anomalias=anomalias
    )

@router.post("/radar", response_model=RadarData)
async def get_radar_data(req: CompareRequest, current_user: str = Depends(get_current_user)):
    sessao_repo = SessaoRepository()
    sessoes = await sessao_repo.get_all()
    
    if not sessoes:
        raise HTTPException(status_code=404, detail="Sem dados.")
        
    df = montar_df_features_brutas([s.model_dump() for s in sessoes])
    perfis = construir_perfis(df, FEATURES)
    
    features_radar = [f for f in req.features if f in FEATURES]
    labels_radar = [f.replace(' (m)', '').replace(' (kph)', '').replace('No. of ', '') for f in features_radar]
    
    datasets = []
    for atleta in req.athlete_ids:
        if atleta in perfis.index:
            cols_radar = [f'Z_{f}' for f in features_radar if f'Z_{f}' in perfis.columns]
            valores = perfis.loc[atleta, cols_radar].values.tolist()
            datasets.append({
                "label": f"Atleta {atleta}",
                "data": [float(v) for v in valores]
            })
            
    return RadarData(labels=labels_radar, datasets=datasets)

@router.get("/historico/{athlete_id}", response_model=HistoricoSerie)
async def get_historico(athlete_id: str, current_user: str = Depends(get_current_user)):
    sessao_repo = SessaoRepository()
    sessoes = await sessao_repo.get_by_athlete(athlete_id)
    
    if not sessoes:
        raise HTTPException(status_code=404, detail="Atleta não encontrado ou sem sessões.")
        
    datas = [s.start_date.isoformat() for s in sessoes]
    valores = [float(s.if_score) if s.if_score is not None else 0.0 for s in sessoes]
    anomalias = [bool(s.if_anomalia) for s in sessoes]
    
    return HistoricoSerie(datas=datas, valores=valores, anomalias=anomalias)
