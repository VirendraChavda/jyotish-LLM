# backend/astro_worker.py
from datetime import datetime, time as dtime, timezone
import threading
from typing import Optional
from backend.db import charts_col
# import your compute function
from pathlib import Path
import sys
# ensure project root on path if needed
sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.astrologer import build_charts  # <- your module

def compute_and_save_static_charts(username: str, name: str,
                                   birth_date_iso: str, birth_time_str: str, birth_place: str):
    try:
        d = datetime.fromisoformat(birth_date_iso).date()
        hh, mm = map(int, birth_time_str.split(":"))
        t = dtime(hh, mm)

        out = build_charts(d.year, d.month, d.day, t.hour, t.minute, birth_place)

        doc = {
            "username": username,
            "name": name,
            "birth_place": birth_place,
            "birth_date": birth_date_iso,
            "birth_time": birth_time_str,
            "ascendant_longitude": out.get("ascendant_longitude") or out.get("ascendant", {}).get("ascendant"),
            "rasi_chart": out.get("rasi_chart"),
            "navamsa_chart": out.get("navamsa_chart"),
            "dasamsa_chart": out.get("dasamsa_chart"),
            "mahadasha": out.get("mahadasha"),
            "antardasha": out.get("antardasha"),
            "natal_strengths": out.get("natal_strengths"),
            "compute_meta": {"ayanamsha": "Lahiri", "saved_at_utc": datetime.now(timezone.utc).isoformat()}
        }
        charts_col.replace_one({"username": username}, doc, upsert=True)
    except Exception as e:
        charts_col.replace_one({"username": username}, {"username": username, "error": str(e)}, upsert=True)

def spawn_chart_job(*args, **kwargs):
    t = threading.Thread(target=compute_and_save_static_charts, args=args, kwargs=kwargs, daemon=True)
    t.start()
