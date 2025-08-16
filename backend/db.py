# backend/db.py
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from auth import hash_password

import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, Dict

from datetime import datetime, date, timezone
from decimal import Decimal
import numpy as np

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

def mongo_sanitize(obj):
    """Recursively make data BSON-safe for MongoDB."""
    if isinstance(obj, dict):
        return {str(k): mongo_sanitize(v) for k, v in obj.items()}  # keys -> str
    if isinstance(obj, (list, tuple, set)):
        return [mongo_sanitize(v) for v in obj]

    # --- dates & datetimes ---
    if isinstance(obj, date) and not isinstance(obj, datetime):
        # store as UTC midnight so it's queryable with $gte/$lte
        return datetime(obj.year, obj.month, obj.day, tzinfo=timezone.utc)
    if isinstance(obj, datetime):
        # ensure tz-aware UTC (Mongo stores UTC)
        return obj if obj.tzinfo else obj.replace(tzinfo=timezone.utc)

    # --- numerics & arrays often coming from numpy/Decimal ---
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj
