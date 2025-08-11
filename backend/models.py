# backend/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ProfileInitRequest(BaseModel):
    name: str
    birth_place: str
    birth_date: str   # ISO date
    birth_time: str   # "HH:MM"
    current_place: str
    current_local_dt: Optional[str] = None  # ISO local datetime (optional)

class SessionUpdateRequest(BaseModel):
    current_place: str
    current_local_dt: Optional[str] = None

class MeResponse(BaseModel):
    username: str

class ChartStatusOut(BaseModel):
    ready: bool
    message: str
    charts: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
