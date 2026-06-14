from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class CompareRequest(BaseModel):
    athlete_ids: list[str]
    features: list[str] = ["Distance (m)", "Sprint Distance (m)", "Top Speed (kph)", "Accelerations", "Decelerations", "High Intensity Running (m)"]
