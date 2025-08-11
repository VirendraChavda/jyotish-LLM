# backend/main.py
import os
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from .db import users_col, profiles_col, charts_col, sessions_col
from .auth import hash_password, verify_password, make_token, decode_token
from .models import *
from .astro_worker import spawn_chart_job
from dotenv import load_dotenv

load_dotenv()

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")

app = FastAPI(title="Astro Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ",1)[1]
    user = decode_token(token)
    if not user:
        raise HTTPException(401, "Invalid/expired token")
    return user

@app.post("/auth/signup", response_model=TokenResponse)
def signup(req: SignupRequest):
    if users_col.find_one({"username": req.username}):
        raise HTTPException(400, "Username already exists")
    users_col.insert_one({"username": req.username, "password_hash": hash_password(req.password)})
    return TokenResponse(access_token=make_token(req.username))

@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    u = users_col.find_one({"username": req.username})
    if not u or not verify_password(req.password, u["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=make_token(req.username))

@app.get("/users/me", response_model=MeResponse)
def me(user: str = Depends(get_current_user)):
    return MeResponse(username=user)

@app.get("/charts/static", response_model=ChartStatusOut)
def get_static_charts(user: str = Depends(get_current_user)):
    doc = charts_col.find_one({"username": user}, {"_id": 0})
    if doc is None:
        return ChartStatusOut(ready=False, message="No charts yet")
    if "error" in doc:
        return ChartStatusOut(ready=False, message=f"Chart error: {doc['error']}")
    return ChartStatusOut(ready=True, message="OK", charts=doc)

@app.post("/profile/init")
def profile_init(req: ProfileInitRequest, user: str = Depends(get_current_user)):
    profiles_col.replace_one(
        {"username": user},
        {
            "username": user,
            "name": req.name,
            "birth_place": req.birth_place,
            "birth_date": req.birth_date,
            "birth_time": req.birth_time,
        },
        upsert=True
    )
    sessions_col.replace_one(
        {"username": user},
        {"username": user, "current_place": req.current_place, "current_local_dt": req.current_local_dt},
        upsert=True
    )
    spawn_chart_job(user, req.name, req.birth_date, req.birth_time, req.birth_place)
    return {"status": "queued"}

@app.post("/session/update")
def session_update(req: SessionUpdateRequest, user: str = Depends(get_current_user)):
    sessions_col.replace_one(
        {"username": user},
        {"username": user, "current_place": req.current_place, "current_local_dt": req.current_local_dt},
        upsert=True
    )
    return {"status": "ok"}

# Optional: LLM chat proxied here
@app.post("/chat")
def chat(req: ChatRequest, user: str = Depends(get_current_user)):
    # Try OpenAI, else return a friendly placeholder
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI()
            messages = [{"role":"system","content":"You are an empathetic Indian astrology assistant."}]
            messages += [m.model_dump() for m in req.messages]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.4)
            return {"reply": resp.choices[0].message.content.strip()}
        except Exception as e:
            return {"reply": f"(LLM error) {e}. How can I help you today?"}
    else:
        last_user = next((m.content for m in reversed(req.messages) if m.role=="user"), "")
        return {"reply": f"I’m here to help. You said: “{last_user}”. What would you like to explore first—career, relationships, or timing?"}
