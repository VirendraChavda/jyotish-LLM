# backend/auth.py
import os, time
import bcrypt, jwt
from typing import Optional

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = "HS256"
JWT_TTL = 60 * 60 * 12  # 12h

def hash_password(pw: str) -> bytes:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())

def verify_password(pw: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(pw.encode("utf-8"), hashed)

def make_token(username: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + JWT_TTL}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload.get("sub")
    except Exception:
        return None
    
def get_current_user():
    # Fake user object for testing
    return {"id": "test-user-123"}