# ⚽ Análise de Atletas — Kicker

API para análise de performance física de atletas a partir de dados de GPS/wearables (Catapult/Kicker), com detecção de anomalias via **Isolation Forest**, direção do desvio via **Z-Score**, comparação entre atletas e um painel visual de verificação (radar, barras e séries temporais).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248?logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

---

## 📑 Sumário

- [Visão geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Como funciona a análise de Machine Learning](#-como-funciona-a-análise-de-machine-learning)
- [Arquitetura e stack](#-arquitetura-e-stack)
- [Pré-requisitos](#-pré-requisitos)
- [Como rodar com Docker](#-como-rodar-com-docker-recomendado)
- [Como rodar localmente (sem Docker)](#-como-rodar-localmente-sem-docker)
- [Variáveis de ambiente](#-variáveis-de-ambiente)
- [Autenticação](#-autenticação)
- [Rotas da API](#-rotas-da-api)
- [Passo a passo no Swagger](#-passo-a-passo-no-swagger-docs)
- [Painel de verificação (/dashboard)](#-painel-de-verificação-dashboard)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Testes](#-testes)
- [Notas e limitações](#-notas-e-limitações)

---

## 🧭 Visão geral

O projeto recebe planilhas `.xlsx` exportadas do sistema de monitoramento (GPS/wearables), processa os dados de cada sessão de treino/jogo, treina **um modelo de Isolation Forest por atleta** e expõe via API:

- a **classificação de cada sessão** (desempenho normal, alta ou queda de desempenho);
- **comparação entre atletas** (perfil físico, radar, sugestão de substitutos);
- **inferência on-demand** para uma nova sessão ainda não salva no banco;
- um **painel HTML de verificação** com gráficos (radar, barra e ponto) para validar visualmente se a pipeline está funcionando.

---

## ✨ Funcionalidades

- 📤 **Upload de planilhas** `.xlsx` com processamento e treinamento em background (não bloqueia a requisição).
- 🌲 **Detecção de anomalias por atleta** com `IsolationForest` (pipeline `StandardScaler` + `IsolationForest`, uma instância treinada por atleta).
- 📈 **Direção do desvio via Z-Score composto**, combinando múltiplas métricas físicas (distância, sprints, acelerações/desacelerações, etc.).
- 🏷️ **Classificação automática da sessão**: `Alta de Desempenho`, `Desempenho Médio` ou `Queda de Desempenho`.
- 🧬 **Perfil de desempenho por atleta**, padronizado globalmente em relação ao elenco (não em relação ao próprio histórico) — usado para comparações justas entre atletas.
- 🔁 **Sugestão de substitutos**: ranking de atletas com perfil físico mais semelhante (distância euclidiana + cosseno).
- 🕸️ **Radar de comparação** entre 2+ atletas nas métricas escolhidas.
- 🔮 **Inferência on-demand** (`/inferencia/predizer/{athlete_id}`) para testar uma sessão hipotética contra o modelo já treinado do atleta.
- 📊 **Painel de QA** em `/dashboard` com gráficos de radar, barras e histórico (ponto/linha) consumindo a própria API.
- 🔐 Autenticação via **JWT** (OAuth2 Password Flow), pronta para uso no Swagger.

---

## 🧠 Como funciona a análise de Machine Learning

A pergunta "essa sessão foi normal, foi um pico de desempenho ou uma queda?" é respondida combinando **dois sinais diferentes**, pois nenhum dos dois sozinho responde à pergunta completa.

### 1. Isolation Forest → "isso é incomum?"

Para cada atleta é treinado um pipeline `StandardScaler + IsolationForest` (contaminação de 10%) usando as métricas físicas da sessão (`Distance (m)`, `Workload`, `Sprint Distance (m)`, `Accelerations`, `Decelerations`, etc.). Ele responde apenas:

- `if_label = 1` → sessão estatisticamente normal para esse atleta;
- `if_label = -1` (`if_anomalia = True`) → sessão estatisticamente **incomum** para esse atleta.

O `if_score` (de `score_samples`) indica *o quão* incomum: **quanto mais negativo, mais anômalo**. Mas o Isolation Forest **não sabe se o desvio é bom ou mau**.

### 2. Z-Score Composto → "incomum em que direção?"

Para cada sessão é calculado o Z-Score de cada métrica em relação ao **histórico do próprio atleta**, e a média dessas métricas forma o `Z_Composto`:

- `Z_Composto > 0` → sessão **acima** da média do atleta naquelas métricas;
- `Z_Composto < 0` → sessão **abaixo** da média do atleta.

### 3. Classificação final

A regra de negócio combina os dois sinais — o Isolation Forest funciona como **gatilho** e o Z-Score dá a **direção**:

```python
if not if_anomalia:
    return "Desempenho Médio"          # nada incomum, mesmo que o Z oscile um pouco

if z_composto > 1.5:
    return "Alta de Desempenho"        # incomum + acima da média do atleta

if z_composto < -1.5:
    return "Queda de Desempenho"       # incomum + abaixo da média do atleta

return "Desempenho Médio"              # incomum, mas sem direção clara/dominante
```

### 4. Comparação entre atletas (perfis, radar e substitutos)

Para **comparar atletas entre si**, o Z-Score por atleta (que por definição matemática tem média ≈ 0 para qualquer atleta) não serve — todos os perfis ficariam idênticos. Em vez disso, o "perfil" de cada atleta é calculado assim:

1. calcula-se a **média dos valores brutos** de cada atleta (volume e intensidade típicos);
2. essa média é padronizada **globalmente**, em relação ao elenco (não ao próprio histórico do atleta).

O resultado é um vetor que mostra, em desvios-padrão, **como o atleta se compara à média do elenco** em cada métrica — usado tanto no **radar** quanto na **sugestão de substitutos** (similaridade = 40% distância euclidiana + 60% distância de cosseno entre perfis).

---

## 🏗️ Arquitetura e stack

| Camada | Tecnologia |
|---|---|
| API | FastAPI + Uvicorn |
| Banco de dados | MongoDB (driver `motor`/`pymongo`) |
| ML | scikit-learn (`StandardScaler` + `IsolationForest`), `joblib` para persistência |
| Processamento de planilhas | `pandas` + `openpyxl` |
| Autenticação | JWT (`python-jose`) via OAuth2 Password Flow |
| Painel de QA | HTML + Chart.js (servido pela própria API em `/dashboard`) |
| Containerização | Docker + Docker Compose |

---

## 📋 Pré-requisitos

**Opção recomendada (Docker):**

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (já incluso no Docker Desktop)

**Opção local (sem Docker):**

- Python 3.11+
- Uma instância do MongoDB rodando localmente (ou acessível via URI)

---

## 🚀 Como rodar com Docker (recomendado)

Na raiz do projeto:

```bash
docker compose up --build
```

Isso vai:

1. construir a imagem da API a partir do `Dockerfile`;
2. subir um container `mongodb` (MongoDB 6.0, com volume persistente);
3. subir o container `api`, expondo a porta `8000`.

Depois de subir, acesse:

- Swagger UI: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**
- Painel de verificação: **http://localhost:8000/dashboard**
- Health-check: **http://localhost:8000/**

Para rodar em segundo plano:

```bash
docker compose up -d --build
```

Para parar:

```bash
docker compose down
```

Para parar e **apagar também os dados do Mongo** (reset completo):

```bash
docker compose down -v
```

> 💡 O `docker-compose.yml` monta `./app` e `./models_onnx` como volumes — então os modelos treinados (`.pkl`) ficam persistidos no host em `./models_onnx`, e alterações no código em `./app` são refletidas no container sem precisar rebuildar (mas o Uvicorn precisa ser reiniciado para recarregar, a menos que esteja em modo `--reload`).

---

## 🐍 Como rodar localmente (sem Docker)

```bash
# 1. Criar e ativar um ambiente virtual
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Garantir que o MongoDB esteja rodando (ex: localhost:27017)

# 4. Subir a API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em `http://localhost:8000`.

---

## ⚙️ Variáveis de ambiente

Configuradas via `.env` (ou diretamente no `docker-compose.yml`):

| Variável | Descrição | Padrão |
|---|---|---|
| `MONGODB_URI` | URI de conexão do MongoDB | `mongodb://localhost:27017/` |
| `DATABASE_NAME` | Nome do banco usado pela API | `analise_atletas` |
| `SECRET_KEY` | Chave usada para assinar os tokens JWT | `your-super-secret-key-change-me` |
| `ALGORITHM` | Algoritmo de assinatura do JWT | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Validade do token em minutos | `1440` (24h) |

> ⚠️ **Importante**: troque `SECRET_KEY` antes de qualquer uso fora do ambiente local/demo.

---

## 🔐 Autenticação

A API usa **OAuth2 Password Flow + JWT**. As credenciais de demonstração são fixas no código (`app/api/v1/auth.py`):

```
usuário: admin
senha:   admin
```

### Pegando um token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=admin"
```

Resposta:

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

### Usando o token

Todas as rotas (exceto `/`, `/docs`, `/redoc` e `/dashboard`) exigem o header:

```
Authorization: Bearer <access_token>
```

No **Swagger UI** (`/docs`), clique no botão **"Authorize"** (cadeado, no topo direito), preencha `username=admin` e `password=admin` e clique em **Authorize** — o Swagger passa a enviar o token automaticamente em todas as chamadas "Try it out".

---

## 📡 Rotas da API

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| `GET` | `/` | não | Health-check simples |
| `GET` | `/docs` | não | Swagger UI |
| `GET` | `/redoc` | não | ReDoc |
| `GET` | `/dashboard` | parcial* | Painel HTML de verificação |
| `POST` | `/api/v1/auth/login` | não | Login, retorna JWT |
| `POST` | `/api/v1/upload/` | sim | Upload de planilha `.xlsx`, processa e treina em background |
| `GET` | `/api/v1/atletas/` | sim | Lista atletas com total de sessões, anomalias e status atual |
| `GET` | `/api/v1/atletas/{athlete_id}/similaridade?top_n=3` | sim | Sugestão de substitutos por similaridade de perfil |
| `GET` | `/api/v1/dashboard/stats` | sim | Estatísticas agregadas (contagens por classificação) |
| `POST` | `/api/v1/dashboard/radar` | sim | Dados de radar para comparar atletas |
| `GET` | `/api/v1/dashboard/historico/{athlete_id}` | sim | Série temporal do `if_score` e flags de anomalia |
| `POST` | `/api/v1/inferencia/predizer/{athlete_id}` | sim | Roda o modelo do atleta sobre uma sessão hipotética |

\* A página HTML em si não exige token, mas todas as chamadas que ela faz à API exigem login (tela de login embutida no painel).

---

## 🧪 Passo a passo no Swagger (`/docs`)

### 1. Login e autorização

1. Abra `http://localhost:8000/docs`.
2. Expanda `POST /api/v1/auth/login` → **Try it out** → preencha `username=admin`, `password=admin` → **Execute**.
3. Copie o `access_token` da resposta (ou apenas use o botão **Authorize** no topo, que faz isso automaticamente).

### 2. Upload da planilha

1. Expanda `POST /api/v1/upload/` → **Try it out**.
2. Selecione um arquivo `.xlsx` exportado do sistema de GPS (precisa conter uma coluna `Segment Name` com linhas `Whole Session`, além de `Athlete ID`, `Start Date` e as métricas físicas: `Distance (m)`, `Workload`, `Sprint Distance (m)`, `Top Speed (kph)`, `Accelerations`, `Decelerations`, etc.).
3. **Execute**. A resposta é imediata (`"Upload recebido... processamento em background"`), mas o processamento (limpeza dos dados + treino de um Isolation Forest por atleta) roda em segundo plano — aguarde alguns segundos antes de consultar os outros endpoints.

### 3. Listando atletas e status

`GET /api/v1/atletas/` retorna, para cada atleta:

```json
{
  "athlete_id": "2432114111",
  "athlete_name": "Nome do Atleta",
  "total_sessions": 24,
  "anomalies_count": 3,
  "performance_status": "Desempenho Médio"
}
```

### 4. Testando o `/predizer` (inferência on-demand)

`POST /api/v1/inferencia/predizer/{athlete_id}` recebe um JSON **plano** (`Dict[str, float]`), onde as chaves precisam ser exatamente os nomes das features usadas no treino. Para um resultado confiável, envie **todas** as 11 chaves abaixo (mesmos nomes, com parênteses/espaços):

```json
{
  "Workload": 6.5,
  "Distance (m)": 8315.0,
  "Metres per Minute (m)": 55.0,
  "High Intensity Running (m)": 449.0,
  "Sprint Distance (m)": 24.0,
  "Top Speed (kph)": 27.3,
  "Avg Speed (kph)": 3.3,
  "Accelerations": 67.0,
  "Decelerations": 52.0,
  "No. of Sprints": 2.0,
  "Duration (mins)": 150.0
}
```

Resposta:

```json
{
  "athlete_id": "2432114111",
  "if_label": 1,
  "if_score": -0.42,
  "is_anomaly": false
}
```

> ⚠️ Requer que o atleta já tenha um modelo treinado (mínimo de 5 sessões no upload), senão retorna `404`. Se enviar menos features do que o modelo foi treinado, a inferência cai num *fallback* e sempre retorna `is_anomaly: false`.

### 5. Radar e similaridade

- `POST /api/v1/dashboard/radar` com body:

  ```json
  {
    "athlete_ids": ["2432114111", "1234567890"],
    "features": ["Distance (m)", "Sprint Distance (m)", "Top Speed (kph)", "Accelerations", "Decelerations", "High Intensity Running (m)"]
  }
  ```

- `GET /api/v1/atletas/2432114111/similaridade?top_n=5` retorna os atletas com perfil mais semelhante.

---

## 📊 Painel de verificação (`/dashboard`)

Acesse **http://localhost:8000/dashboard** para um painel HTML standalone, pensado para validar visualmente a pipeline:

- **Cards de resumo**: total de atletas, sessões, e contagem por classificação.
- **Barra — Distribuição de classificação**: quantas sessões caíram em Alta / Médio / Queda.
- **Barra — Anomalias por atleta**: quantas sessões o Isolation Forest marcou como incomuns, por atleta.
- **Radar — Comparação de perfis**: selecione 2 a 4 atletas e compare seus perfis padronizados.
- **Ponto/linha — Histórico do IF Score**: evolução do `if_score` por sessão, com pontos em destaque (âmbar) para anomalias.
- **Barra — Similaridade**: ranking de substitutos sugeridos para o atleta selecionado.
- **Tabela de atletas**: visão geral com sessões, anomalias e status.

Ao abrir, será exibida uma tela de login (mesmas credenciais `admin` / `admin`) — o painel guarda o token apenas em memória (na aba aberta) e faz as chamadas à API a partir do próprio navegador.

---

## 📁 Estrutura do projeto

```
analise_atletas_kicker_new/
├── app/
│   ├── main.py                  # ponto de entrada FastAPI, registra rotas e /dashboard
│   ├── core/                    # config, segurança (JWT), conexão com Mongo, logger
│   ├── models/                  # modelos de domínio (SessaoFisica, Atleta)
│   ├── schemas/                 # schemas Pydantic de request/response
│   ├── repositories/            # acesso ao MongoDB (atletas, sessões)
│   ├── services/
│   │   ├── data_processing.py   # limpeza/normalização da planilha
│   │   ├── model_training.py    # treino do pipeline StandardScaler + IsolationForest
│   │   ├── onnx_inference.py    # inferência (joblib) por atleta
│   │   └── analytics.py         # Z-Score, classificação, perfis e similaridade
│   ├── api/v1/                  # routers: auth, upload, atletas, dashboard, inferencia
│   └── static/dashboard.html    # painel de verificação (radar, barra, ponto)
├── tests/                        # testes unitários (pytest)
├── models_onnx/                  # modelos treinados (.pkl), 1 por atleta
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## ✅ Testes

```bash
# com a venv ativada e dependências instaladas
pytest tests/ -v
```

Os testes cobrem:

- a regra de classificação combinando Isolation Forest (gatilho) + Z-Score (direção);
- a construção dos perfis de atleta (padronização global) e a diferenciação na sugestão de substitutos;
- o mapeamento dos campos brutos das sessões para as features usadas na análise.

---

## ⚠️ Notas e limitações

- O login é **fixo/demo** (`admin` / `admin`) — substitua por um sistema de usuários real antes de qualquer uso em produção.
- Modelos são treinados **por atleta**; é necessário um mínimo de 5 sessões para que o Isolation Forest daquele atleta seja treinado.
- O processamento do upload roda em **background** — os dados podem demorar alguns segundos para aparecer nos demais endpoints após o upload.
- A pasta `models_onnx/` mantém esse nome por motivos históricos, mas os modelos são salvos via `joblib` (pipelines scikit-learn), não em formato ONNX.
- O limiar de `±1.5` no Z-Score Composto é configurável em `app/services/analytics.py` (`classificar_sessao`) — ajuste conforme a sensibilidade desejada para os alertas de Alta/Queda de Desempenho.
