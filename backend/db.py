# backend/db.py
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("mongo_uri")
uri = mongo_uri

# Create a new client and connect to the server
client = MongoClient(uri)

auth_db = client["auth_db"]
astro_db = client["astro_db"]

users_col = auth_db["users"]                 # credentials only
profiles_col = astro_db["users_profile"]     # birth-dependent identity
sessions_col = astro_db["sessions"]          # current time/location (no dynamic charts)

# indexes
users_col.create_index([("username", ASCENDING)], unique=True)
profiles_col.create_index([("username", ASCENDING)], unique=True)
sessions_col.create_index([("username", ASCENDING)], unique=True)