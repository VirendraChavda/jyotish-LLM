# backend/astro_worker.py
from datetime import datetime, time as dtime, timezone
import threading
# import your compute function
import time
import traceback
from pymongo.errors import PyMongoError

from astrologer import build_charts
from db import users_col, profiles_col, sessions_col, mongo_sanitize

def compute_and_save_static_charts(username: str, name: str, birth_date_iso: str, birth_time_str: str, birth_place: str, current_place: str,
                                   include_transit_events=True, transit_span_years=3):
    try:
        y, m, d = map(int, birth_date_iso.split("-"))
        hh, mm = map(int, birth_time_str.split(":"))

        out = build_charts(y, m, d, hh, mm, birth_place, current_place, include_transit_events=include_transit_events, transit_span_years=transit_span_years)
        out_s = mongo_sanitize(out)

        profile_doc = {
            "username": username,
            "name": name,
            "birth_place": birth_place,
            "birth_date": birth_date_iso,
            "birth_time": birth_time_str,
            "ascendant_longitude": out.get("ascendant_longitude"),
            "rasi_chart": out.get("rasi_chart"),
            "navamsa_chart": out.get("navamsa_chart"),
            "dasamsa_chart": out.get("dasamsa_chart"),
            "transit_chart": out.get("transit_chart"),
            "transit_positions": out_s.get("transit_positions", {}),
            "mahadasha": out.get("mahadasha"),
            "antardasha": out.get("antardasha"),
            "natal_strengths": out.get("natal_strengths"),
            "birth_location": out.get("birth_location"),
            "updated_at": datetime.now(),
        }

        session_doc = {
            "username": username,
            "name": name,
            "transit_events": out_s.get("transit_events"),
            "influence_now": out.get("influence_now"),
            "current_md": out.get("current_md"),
            "current_ad": out.get("current_ad"),
            "current_location": out.get("current_location"),
            "updated_at": datetime.now(),
        }

        profiles_col.replace_one({"username": username}, profile_doc, upsert=True)
        sessions_col.replace_one({"username": username}, session_doc, upsert=True)
        #print("Uploaded successfully!")

    except (ValueError, KeyError, PyMongoError, Exception) as exc:
        err_doc = {
            "username": username,
            "error": str(exc),
            "traceback": traceback.format_exc(),  # super helpful for debugging
            "updated_at": datetime.now(),
        }
        # Use $set so you don't blow away existing good fields
        profiles_col.update_one({"username": username}, {"$set": err_doc}, upsert=True)
        sessions_col.update_one({"username": username}, {"$set": err_doc}, upsert=True)
        # optionally also: logging.exception("Failed to build charts for %s", username)
        #print("Error!")

def spawn_chart_job(*args, **kwargs):
    t = threading.Thread(target=compute_and_save_static_charts, args=args, kwargs=kwargs, daemon=True)
    t.start()

if __name__ == "__main__":
    # spawn_chart_job("param", "Param Sundari", "1962-4-08", "2:30", "Pune, India", "Mumbai, Gujarat")
    # time.sleep(60)
    compute_and_save_static_charts("param", "Param Sundari", "1962-4-08", "2:30", "Pune, India", "Mumbai, Gujarat", include_transit_events=True, transit_span_years=3)