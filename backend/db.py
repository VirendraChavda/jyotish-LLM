# backend/db.py
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)

auth_db = client["auth_db"]
astro_db = client["astro_db"]

users_col = auth_db["users"]                 # credentials only
profiles_col = astro_db["users_profile"]     # birth-dependent identity
charts_col = astro_db["charts_static"]       # birth-dependent computed charts
sessions_col = astro_db["sessions"]          # current time/location (no dynamic charts)

# indexes
users_col.create_index([("username", ASCENDING)], unique=True)
profiles_col.create_index([("username", ASCENDING)], unique=True)
charts_col.create_index([("username", ASCENDING)], unique=True)
sessions_col.create_index([("username", ASCENDING)], unique=True)