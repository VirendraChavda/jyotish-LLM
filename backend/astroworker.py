# backend/astro_worker.py
from datetime import datetime, time as dtime, timezone
import threading
from typing import Optional
from backend.db import users_col, profiles_col, sessions_col
# import your compute function
from pathlib import Path
import sys
# ensure project root on path if needed
#sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.astrologer import build_charts
from backend.db import mongo_sanitize  

def compute_and_save_static_charts(username: str, name: str,
                                   birth_date_iso: str, birth_time_str: str, birth_place: str, current_place: str):
    try:
        #d = datetime.fromisoformat(birth_date_iso).date()
        y, m, d = map(int, birth_date_iso.split("-"))
        hh, mi = map(int, birth_time_str.split(":"))

        out = build_charts(y, m, d, hh, mi, e["birth_place"], e["current_place"])
        out_s = mongo_sanitize(out)

        # Store a document shaped like your worker writes
        profile_doc = {
            "username": e["username"],
            "name": e["name"],
            "birth_place": e["birth_place"],
            "birth_date": e["birth_date"],
            "birth_time": e["birth_time"],
            "ascendant_longitude": out.get("ascendant_longitude"),
            "rasi_chart": out.get("rasi_chart"),
            "navamsa_chart": out.get("navamsa_chart"),
            "dasamsa_chart": out.get("dasamsa_chart"),
            "transit_chart": out.get("transit_chart"),
            "transit_positions": out_s.get("transit_positions", {}),
            "mahadasha": out.get("mahadasha"),
            "antardasha": out.get("antardasha"),
            "natal_strengths": out.get("natal_strengths"),
            "birth_location": out.get("birth_location")
        }

        session_doc = {
            "username": e["username"],
            "name": e["name"],
            "transit_events": out_s.get("transit_events"),
            "influence_now": out.get("influence_now"),
            "current_md": out.get("current_md"),
            "current_ad": out.get("current_ad"),
            "current_location": out.get("current_location")
        }

        profiles_col.replace_one({"username": e["username"]}, profile_doc, upsert=True)
        sessions_col.replace_one({"username": e["username"]}, session_doc, upsert=True)
    except Exception as e:
        profiles_col.replace_one({"username": e["username"]}, {"username": username, "error": str(e)}, upsert=True)
        sessions_col.replace_one({"username": e["username"]}, {"username": username, "error": str(e)}, upsert=True)


def spawn_chart_job(*args, **kwargs):
    t = threading.Thread(target=compute_and_save_static_charts, args=args, kwargs=kwargs, daemon=True)
    t.start()
