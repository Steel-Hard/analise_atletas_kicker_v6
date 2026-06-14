from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.api.v1 import auth, upload, atletas, dashboard, inferencia

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Autenticação"])
app.include_router(upload.router, prefix=f"{settings.API_V1_STR}/upload", tags=["Upload de Planilhas"])
app.include_router(atletas.router, prefix=f"{settings.API_V1_STR}/atletas", tags=["Atletas"])
app.include_router(dashboard.router, prefix=f"{settings.API_V1_STR}/dashboard", tags=["Dashboard Analytics"])
app.include_router(inferencia.router, prefix=f"{settings.API_V1_STR}/inferencia", tags=["Inferência ONNX"])

@app.get("/")
def root():
    return {"message": f"Bem vindo a API {settings.PROJECT_NAME} v{settings.VERSION}"}

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page():
    """
    Painel estático de QA: gráficos de radar, barra e ponto para verificar
    visualmente se a pipeline de Machine Learning (Isolation Forest +
    Z-Score) e o cálculo de comparação entre atletas estão corretos.
    """
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
