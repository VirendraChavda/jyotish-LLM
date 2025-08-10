import swisseph as swe
from collections import namedtuple
from jhora.panchanga import drik
from jhora.horoscope.chart import charts

EPHE_PATH = r"ephe"
AYANAMSA_MODE = swe.SIDM_LAHIRI

def jd_ut(y, m, d, hh, mi, tz):
    ut_hour = hh + mi/60 - tz
    return swe.julday(y, m, d, ut_hour)

def main():
    # ---- inputs ----
    y, m, d = 1995, 8, 10
    hh, mi = 14, 30
    lat, lon = 28.6139, 77.2090
    tz = 5.5
    Place = namedtuple("Place", ["name", "latitude", "longitude", "timezone"])
    place_obj = Place("Delhi, India", lat, lon, tz)
    place_tuple = (place_obj.name, place_obj.latitude, place_obj.longitude, place_obj.timezone)

    # ---- swiss ephemeris ----
    swe.set_ephe_path(EPHE_PATH)
    swe.set_sid_mode(AYANAMSA_MODE, 0, 0)

    jd = jd_ut(y, m, d, hh, mi, tz)

    # drik.* wants tuple
    asc = drik.ascendant(jd, place_tuple)

    # charts.* wants object with attributes
    rasi = charts.rasi_chart(jd, place_obj)

    # 🔧 KEY CHANGE: pass Rāśi positions into navāṁśa
    try:
        navamsa = charts.navamsa_chart(rasi)
    except TypeError:
        # fallback if your version exposes a generic helper
        navamsa = charts.divisional_positions_from_rasi_positions(rasi, 9)

    houses = charts.bhava_houses(jd, place_obj)

    print("JD (UT):", jd)
    print("Ascendant (deg):", asc)
    print("\nRāśi (D1):", rasi)
    print("\nNavāṁśa (D9):", navamsa)
    print("\nBhāva houses:", houses)

if __name__ == "__main__":
    main()
