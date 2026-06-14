from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from app.services.data_processing import process_upload_excel, FEATURES
from app.services.model_training import retrain_all_athletes
from app.services.analytics import calcular_zscore_por_atleta, calcular_z_composto, classificar_df
from app.repositories.sessao_repository import SessaoRepository
from app.repositories.atleta_repository import AtletaRepository
from app.models.domain import SessaoFisica, Atleta
from app.api.deps import get_current_user
from app.core.logger import logger
import pandas as pd

router = APIRouter()

from fastapi.concurrency import run_in_threadpool

def _cpu_bound_processing(file_content: bytes):
    # 1. Processamento base
    df_ws = process_upload_excel(file_content)
    
    # 2. Retreinar ONNX Models
    logger.info("Iniciando retreinamento dos modelos ONNX...")
    retrain_all_athletes(df_ws)
    logger.info("Retreinamento concluido.")
    
    # 3. Inferência local
    from app.services.onnx_inference import run_onnx_inference
    import os
    
    resultados_if = []
    for athlete_id, grupo in df_ws.groupby('Athlete ID'):
        model_path = os.path.join("models_onnx", f"atleta_{athlete_id}.pkl")
        features_usadas = [f for f in FEATURES if f in grupo.columns]
        if os.path.exists(model_path) and features_usadas:
            X = grupo[features_usadas].values
            inf_res = run_onnx_inference(model_path, X, features_usadas)
            grupo_c = grupo.copy()
            grupo_c['if_label'] = inf_res['labels']
            grupo_c['if_score'] = inf_res['scores']
            grupo_c['if_anomalia'] = grupo_c['if_label'] == -1
            resultados_if.append(grupo_c)
        else:
            grupo_c = grupo.copy()
            grupo_c['if_label'] = 1
            grupo_c['if_score'] = 0
            grupo_c['if_anomalia'] = False
            resultados_if.append(grupo_c)
    
    if resultados_if:
        df_if = pd.concat(resultados_if, ignore_index=True)
    else:
        df_if = df_ws.copy()
        df_if['if_score'] = 0
        df_if['if_anomalia'] = False
        
    # 4. Z-Scores
    df_z = calcular_zscore_por_atleta(df_if, FEATURES)
    df_z = calcular_z_composto(df_z, FEATURES)
    df_z = classificar_df(df_z)
    return df_z

async def process_and_train_background(file_content: bytes):
    try:
        # Executar tarefas pesadas (pandas/scikit) fora do event loop principal
        df_z = await run_in_threadpool(_cpu_bound_processing, file_content)
        
        # 5. Salvar no Banco Asincronamente usando o MESMO loop do FastAPI
        sessao_repo = SessaoRepository()
        atleta_repo = AtletaRepository()
        
        sessoes_to_insert = []
        for _, row in df_z.iterrows():
            # Preparar Atleta
            ath = Atleta(
                athlete_id=str(row['Athlete ID']),
                athlete_name=str(row.get('Athlete Name', '')),
                athlete_position=str(row.get('Athlete Position', ''))
            )
            await atleta_repo.create_or_update(ath)
            
            # Preparar Z-scores
            z_dict = {}
            for f in FEATURES:
                if f'Z_{f}' in row:
                    z_dict[f] = float(row[f'Z_{f}'])
                    
            # Preparar Sessao
            sf = SessaoFisica(
                athlete_id=str(row['Athlete ID']),
                start_date=row['Start Date'],
                segment_name=str(row['Segment Name']),
                duration_mins=row.get('Duration (mins)'),
                workload=row.get('Workload'),
                distance_m=row.get('Distance (m)'),
                metres_per_minute=row.get('Metres per Minute (m)'),
                high_intensity_running_m=row.get('High Intensity Running (m)'),
                sprint_distance_m=row.get('Sprint Distance (m)'),
                top_speed_kph=row.get('Top Speed (kph)'),
                avg_speed_kph=row.get('Avg Speed (kph)'),
                accelerations=row.get('Accelerations'),
                decelerations=row.get('Decelerations'),
                no_of_sprints=row.get('No. of Sprints'),
                if_score=row.get('if_score'),
                if_anomalia=row.get('if_anomalia'),
                z_composto=row.get('Z_Composto'),
                classificacao=row.get('Classificacao'),
                z_scores=z_dict
            )
            # handle nan to none
            for k, v in sf.model_dump().items():
                if isinstance(v, float) and pd.isna(v):
                    setattr(sf, k, None)
            sessoes_to_insert.append(sf)
            
        await sessao_repo.get_collection().delete_many({}) # clear old
        if sessoes_to_insert:
            await sessao_repo.insert_many(sessoes_to_insert)
            
        logger.info("Dados salvos no MongoDB com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no processamento background: {e}", exc_info=True)

@router.post("/")
async def upload_planilha(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    if not file.filename.endswith('.xlsx'):
        return {"error": "Apenas arquivos .xlsx são permitidos."}
        
    content = await file.read()
    background_tasks.add_task(process_and_train_background, content)
    
    return {"message": "Upload recebido com sucesso. Processamento e treinamento iniciados em background."}
