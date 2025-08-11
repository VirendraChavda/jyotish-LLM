"""
Requirements for astrological predictions:

- **Date of Birth**
- **Time of Birth**
- **Place of Birth**
- **Location and timezone**
- **Today's date (for creating transit chart)**
- **Lagna Chart**
- **Rasi Chart (D-1 Chart)**
- **Navamsa Chart (D-9 Chart)**
- **Dasamsa Chart (D-10 Chart)**
- **Transit Chart**
- **Partner's Birth Data (for relationship scenarios)**
- **Child's Birth Data (for parenting scenarios)**
- **Transit Calculations (to predict change is dasha)**
- **Panchang data - Tithi, Nakṣatra, Yoga, Karaṇa(for muhurta, compatibility, and event timing)**
- **Vimśottarī Daśā table (and optionally Yogini, Chara, Kalachakra, etc.) (for sequence of planetary periods and sub-periods for the native)**
- **Saptāṁśa (D7)(children & creativity)**
- **Shadbala, Ashtakavarga(to explain why a planet acts in a certain way during dashā/transit)**
"""
    # y, m, d = 1995, 8, 10
    # hh, mi = 14, 30

import swisseph as swe
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

from datetime import datetime, timedelta
import pytz
from collections import namedtuple

from jhora.panchanga import drik
from jhora.horoscope.chart import charts

# --------------------------- CONFIG ---------------------------------

EPHE_PATH = r"ephe"
AYANAMSA_MODE = swe.SIDM_LAHIRI

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN
}

# Vimshottari order & years
VIM_DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
VIM_DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}

# Exaltation degrees (sidereal sign degrees; Aries=0..30 etc.)
EXALT_DEG = {
    swe.SUN:      (0*30 + 10),  # Sun exalted 10° Aries
    swe.MOON:     (1*30 + 3),   # Moon 3° Taurus
    swe.MARS:     (9*30 + 28),  # Mars 28° Capricorn
    swe.MERCURY:  (5*30 + 15),  # Mercury 15° Virgo
    swe.JUPITER:  (3*30 + 5),   # Jupiter 5° Cancer
    swe.VENUS:    (11*30 + 27), # Venus 27° Pisces
    swe.SATURN:   (6*30 + 20)   # Saturn 20° Libra
}
DEBIL_DEG = {
    swe.SUN:      (6*30 + 10),  # Sun 10° Libra
    swe.MOON:     (7*30 + 3),   # Moon 3° Scorpio
    swe.MARS:     (3*30 + 28),  # Mars 28° Cancer
    swe.MERCURY:  (11*30 + 15), # Mercury 15° Pisces
    swe.JUPITER:  (9*30 + 5),   # Jupiter 5° Capricorn
    swe.VENUS:    (5*30 + 27),  # Venus 27° Virgo
    swe.SATURN:   (0*30 + 20)   # Saturn 20° Aries
}

# ------------------------ TIME & GEO HELPERS ------------------------

def jd_ut(y, m, d, hh, mi, tz_hours):
    """Julian day at UT for local date/time with given time zone hours."""
    ut_hour = hh + mi/60.0 - tz_hours
    return swe.julday(y, m, d, ut_hour, swe.GREG_CAL)

def revjul_to_datetime_utc(jd):
    """Swiss Ephem revjul -> Python datetime in UTC. revjul returns hours, not fractional days."""
    y, m, d, ut_hours = swe.revjul(jd, swe.GREG_CAL)
    return datetime(y, m, d) + timedelta(hours=ut_hours)

def get_location_details(city_name):
    """Geocode + timezone + CURRENT UTC offset hours (good enough for present-tense use)."""
    geolocator = Nominatim(user_agent="astro_app")
    location = geolocator.geocode(city_name)
    if not location:
        raise ValueError(f"City not found: {city_name}")

    lat, lon = location.latitude, location.longitude

    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        raise ValueError(f"Timezone not found for: {city_name}")

    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    offset_hours = now.utcoffset().total_seconds() / 3600.0

    return {
        "city": city_name,
        "latitude": lat,
        "longitude": lon,
        "timezone_name": tz_name,
        "timezone_offset": offset_hours
    }

# ------------------------ VIMSHOTTARI DASHA -------------------------

def normalize_deg(x):
    return x % 360.0

def moon_nakshatra_index(moon_longitude_deg):
    """0..26 index for 27 nakshatras (each 13°20')."""
    return int(moon_longitude_deg // (360.0 / 27.0))

def get_vimshottari_dasha_table(jd_birth):
    """
    Mahadasha sequence from birth moon nakshatra.
    Returns list of dicts: {lord, start_dt_utc, end_dt_utc, years}
    First MD is partial based on remaining portion of Moon's nakshatra.
    """
    res = swe.calc_ut(jd_birth, swe.MOON)
    moon_long = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
    moon_long = normalize_deg(moon_long)

    idx = moon_nakshatra_index(moon_long)               # 0..26
    start_lord = VIM_DASHA_LORDS[idx % 9]               # dasha lord per nakshatra cycle

    # Portion of nakshatra left (to go) at birth
    span = 360.0 / 27.0
    nak_start = idx * span
    frac_left = 1.0 - ((moon_long - nak_start) / span)  # 0..1

    # Remaining years of the first MD
    first_md_years = VIM_DASHA_YEARS[start_lord] * frac_left

    # Birth moment as UTC datetime
    birth_dt_utc = revjul_to_datetime_utc(jd_birth)

    # Build 9 MDs (first partial, then 8 full to cover a full cycle)
    table = []
    lord_index = VIM_DASHA_LORDS.index(start_lord)
    current_start = birth_dt_utc
    years_remaining_for_first = first_md_years

    for i in range(9):
        lord = VIM_DASHA_LORDS[(lord_index + i) % 9]
        years = years_remaining_for_first if i == 0 else VIM_DASHA_YEARS[lord]
        delta_days = years * 365.25
        end_dt_utc = current_start + timedelta(days=delta_days)
        table.append({
            "lord": lord,
            "start_dt_utc": current_start,
            "end_dt_utc": end_dt_utc,
            "years": years
        })
        current_start = end_dt_utc
        years_remaining_for_first = None

    return table

def expand_antardashas_for_mahadasha(md_entry):
    """
    Expand one Mahadasha into 9 Antardashas in Vimshottari order starting from the MD lord.
    Each AD duration (in years) = MD_years * (Years_of_AD_lord / 120).
    Returns list of dicts: {md_lord, ad_lord, start_dt_utc, end_dt_utc, years}
    """
    md_lord = md_entry["lord"]
    md_years = md_entry["years"]
    start_idx = VIM_DASHA_LORDS.index(md_lord)
    order = [VIM_DASHA_LORDS[(start_idx + k) % 9] for k in range(9)]

    ad_list = []
    cur_start = md_entry["start_dt_utc"]
    for ad_lord in order:
        years = md_years * (VIM_DASHA_YEARS[ad_lord] / 120.0)
        end_dt = cur_start + timedelta(days=years * 365.25)
        ad_list.append({
            "md_lord": md_lord,
            "ad_lord": ad_lord,
            "start_dt_utc": cur_start,
            "end_dt_utc": end_dt,
            "years": years
        })
        cur_start = end_dt
    return ad_list

def get_all_mahadasha_antardasha(jd_birth):
    """Return (md_table, ad_table) where ad_table is a flat list across all MDs."""
    md_table = get_vimshottari_dasha_table(jd_birth)
    ad_table = []
    for md in md_table:
        ad_table.extend(expand_antardashas_for_mahadasha(md))
    return md_table, ad_table

def find_current_md_ad(md_table, ad_table, when_dt_utc):
    """Return (current_md_lord, current_ad_lord) for the given UTC datetime."""
    cur_md = None
    for md in md_table:
        if md["start_dt_utc"] <= when_dt_utc < md["end_dt_utc"]:
            cur_md = md["lord"]
            break
    cur_ad = None
    for ad in ad_table:
        if ad["start_dt_utc"] <= when_dt_utc < ad["end_dt_utc"]:
            cur_ad = ad["ad_lord"]
            break
    return cur_md, cur_ad

# ----------------------- NATAL STRENGTH (SIMPLE) --------------------

def circular_distance_deg(a, b):
    """Shortest distance on circle in degrees."""
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff

def house_index_from_asc(longitude, asc_long):
    """0..11 house index using 30° sectors from the ecliptic point of Ascendant."""
    return int(((longitude - asc_long) % 360.0) // 30.0)

def calculate_natal_strength_simple(jd_birth, asc_long):
    """
    Return dict {planet_name: score_0_to_60}.
    Combines:
      - Dignity (proximity to exaltation vs debilitation, scaled 0..40)
      - Angularity by house from Asc (angles>succedents>cadent, scaled 0..20)
    """
    strengths = {}
    for name, pid in PLANETS.items():
        res = swe.calc_ut(jd_birth, pid)
        lon = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
        lon = normalize_deg(lon)

        # Dignity
        ex = EXALT_DEG[pid]
        de = DEBIL_DEG[pid]
        d_ex = circular_distance_deg(lon, ex)
        d_de = circular_distance_deg(lon, de)
        dig = max(0.0, (d_de - d_ex) / 180.0)   # 0..1
        dignity_points = 40.0 * dig

        # Angularity: house from Asc
        h = house_index_from_asc(lon, asc_long)
        if h in (0, 3, 6, 9):       # houses 1,4,7,10
            ang = 1.0
        elif h in (1, 4, 7, 10):    # 2,5,8,11
            ang = 0.6
        else:                       # 3,6,9,12
            ang = 0.3
        angularity_points = 20.0 * ang

        score = dignity_points + angularity_points  # 0..60
        strengths[name] = round(score, 2)
    return strengths

# -------------------------- TRANSITS & EVENTS -----------------------

def get_transit_house_for_planet(jd_when, asc_long, pid):
    res = swe.calc_ut(jd_when, pid, swe.FLG_SPEED)  # returns (xx, retflag)
    xx = res[0] if isinstance(res, tuple) else res
    lon = normalize_deg(xx[0])
    lon_speed = xx[3]
    house = house_index_from_asc(lon, asc_long) + 1
    retro = lon_speed < 0
    return house, retro, lon

def get_transit_events(jd_start, jd_end, ascendant_long):
    """
    Sign and house change dates for the classic 7 planets, stepping daily (simple).
    """
    swe.set_ephe_path(EPHE_PATH)
    events = []
    for planet_name, planet_id in PLANETS.items():
        prev_sign = None
        prev_house = None
        jd = jd_start
        while jd <= jd_end:
            res = swe.calc_ut(jd, planet_id)
            lon = normalize_deg(res[0][0] if isinstance(res[0], (list, tuple)) else res[0])
            sign_index = int(lon // 30)
            house_index = int(((lon - ascendant_long) % 360) // 30)

            if prev_sign is not None and sign_index != prev_sign:
                y, m, d, _ = swe.revjul(jd, swe.GREG_CAL)
                events.append({
                    "planet": planet_name,
                    "type": "sign_change",
                    "date": datetime(y, m, d).date(),
                    "from": SIGNS[prev_sign],
                    "to": SIGNS[sign_index]
                })
            if prev_house is not None and house_index != prev_house:
                y, m, d, _ = swe.revjul(jd, swe.GREG_CAL)
                events.append({
                    "planet": planet_name,
                    "type": "house_change",
                    "date": datetime(y, m, d).date(),
                    "from_house": prev_house + 1,
                    "to_house": house_index + 1
                })

            prev_sign = sign_index
            prev_house = house_index
            jd += 1.0
    return events

# ---------------------- INFLUENCE SCORE (BASIC) ---------------------

def dasha_weight_for_planet(planet_name, current_md, current_ad):
    """
    Multiplicative weight for timing (MD/AD rulers).
      - both MD & AD ruler: 1.4
      - MD only:            1.3
      - AD only:            1.15
      - neither:            1.0
    """
    if planet_name == current_md and planet_name == current_ad:
        return 1.4
    if planet_name == current_md:
        return 1.3
    if planet_name == current_ad:
        return 1.15
    return 1.0

def transit_context_factor(planet_name, jd_when, asc_long):
    """
    ~0.3..1.2 factor based on transit house & simple conditions.
      - Base by house (angular 1.0, succedent 0.6, cadent 0.3)
      - Retrograde +0.1
      - Combust (within 8° of Sun) ×0.8 (penalty); Moon ignored
    """
    pid = PLANETS[planet_name]
    house, retro, lon = get_transit_house_for_planet(jd_when, asc_long, pid)

    # base house factor
    h_idx = house - 1
    if h_idx in (0, 3, 6, 9):       # angular
        base = 1.0
    elif h_idx in (1, 4, 7, 10):    # succedent
        base = 0.6
    else:                           # cadent
        base = 0.3

    # retrograde bonus
    bonus = 0.1 if retro else 0.0

    # combustion penalty (skip Moon)
    res_sun = swe.calc_ut(jd_when, swe.SUN)
    sun_lon = res_sun[0][0] if isinstance(res_sun[0], (list, tuple)) else res_sun[0]
    combust_penalty = 1.0
    if planet_name != "Moon":
        from math import fmod
        def circ_dist(a,b): return abs((a - b + 180.0) % 360.0 - 180.0)
        if circ_dist(normalize_deg(lon), normalize_deg(sun_lon)) < 8.0:
            combust_penalty = 0.8

    factor = (base + bonus) * combust_penalty
    factor = max(0.1, min(1.2, factor))
    return factor, house

def influence_scores_table(natal_strengths_0_60, current_md, current_ad, jd_when, asc_long):
    """
    Returns list of rows: {Planet, NatalStrength, RulingNow?, TransitHouse, InfluenceScore}
    InfluenceScore (0..100 scale) = 100 × (dasha_weight × (natal/60) × transit_factor)
    """
    rows = []
    for name in PLANETS.keys():
        natal = natal_strengths_0_60[name]               # 0..60
        nat_norm = max(0.0, min(1.0, natal / 60.0))
        w = dasha_weight_for_planet(name, current_md, current_ad)
        trans_factor, house = transit_context_factor(name, jd_when, asc_long)
        score = 100.0 * (w * nat_norm * trans_factor)
        ruling_now = ("MD" if name == current_md else "") + ("/AD" if name == current_ad else "")
        ruling_now = ruling_now.strip("/") if ruling_now else ""
        rows.append({
            "Planet": name,
            "NatalStrength": round(natal, 2),
            "RulingNow?": ruling_now,
            "TransitHouse": house,
            "InfluenceScore": round(score, 2)
        })
    rows.sort(key=lambda r: r["InfluenceScore"], reverse=True)
    return rows

def print_influence_table(rows):
    col_names = ["Planet", "NatalStrength", "RulingNow?", "TransitHouse", "InfluenceScore"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in col_names}
    sep = " | "
    header = sep.join(c.ljust(widths[c]) for c in col_names)
    line = "-+-".join("-" * widths[c] for c in col_names)
    print(header)
    print(line)
    for r in rows:
        print(sep.join(str(r[c]).ljust(widths[c]) for c in col_names))

# ----------------------------- MAIN API -----------------------------

def build_charts(y, m, d, hh, mi, place_name):
    """
    Builds charts + dasha + transit + influence table ingredients.
    Returns dict with everything, plus prints the influence table for 'now'.
    """
    # Setup ephemeris
    swe.set_ephe_path(EPHE_PATH)
    swe.set_sid_mode(AYANAMSA_MODE, 0, 0)

    # Location & timezone
    location_details = get_location_details(place_name)
    lat, lon, tz_offset, tz_name = (
        location_details["latitude"],
        location_details["longitude"],
        location_details["timezone_offset"],
        location_details["timezone_name"]
    )

    Place = namedtuple("Place", ["name", "latitude", "longitude", "timezone"])
    place_obj = Place(place_name, lat, lon, tz_offset)
    place_tuple = (place_obj.name, place_obj.latitude, place_obj.longitude, place_obj.timezone)

    # Birth JD in UT
    jd_birth = jd_ut(y, m, d, hh, mi, tz_offset)

    # Ascendant (Swiss Ephemeris) — robust float, avoids list/dict ambiguity
    # houses() returns (cusps[1..12], ascmc[0..9]); ascmc[0] is Ascendant longitude
    cusps, ascmc = swe.houses(jd_birth, lat, lon, b'P')  # P=Placidus (Asc longitude is independent of house system)
    asc_long = normalize_deg(ascmc[0])

    # Charts from jhora (kept as in your code)
    rasi = charts.rasi_chart(jd_birth, place_obj)
    try:
        navamsa = charts.navamsa_chart(rasi)
    except TypeError:
        navamsa = charts.divisional_positions_from_rasi_positions(rasi, 9)
    try:
        dasamsa = charts.dasamsa_chart(rasi)
    except TypeError:
        dasamsa = charts.divisional_positions_from_rasi_positions(rasi, 10)

    # Today's positions
    tz_obj = pytz.timezone(tz_name)
    now_local = datetime.now(tz_obj)
    jd_today = jd_ut(now_local.year, now_local.month, now_local.day, now_local.hour, now_local.minute, tz_offset)
    transit_rasi = charts.rasi_chart(jd_today, place_obj)
    transit_positions = {planet: drik.sidereal_longitude(jd_today, planet) for planet in range(9)}

    # Dasha tables (MD + AD)
    md_table, ad_table = get_all_mahadasha_antardasha(jd_birth)

    # Current MD/AD at "now" (use UTC)
    now_utc = datetime.now()
    current_md, current_ad = find_current_md_ad(md_table, ad_table, now_utc)

    # Natal strength (simple dignity + angularity)
    natal_strengths = calculate_natal_strength_simple(jd_birth, asc_long)

    # Transit events (3 years ahead)
    jd_next3y = jd_ut(now_local.year + 3, now_local.month, now_local.day, now_local.hour, now_local.minute, tz_offset)
    transit_events = get_transit_events(jd_today, jd_next3y, asc_long)

    # Influence table for now
    rows = influence_scores_table(natal_strengths, current_md, current_ad, jd_today, asc_long)

    # Print neat table
    print("\nInfluence (now): MD = {0}, AD = {1}".format(current_md, current_ad))
    print_influence_table(rows)

    return {
        "ascendant_longitude": asc_long,
        "rasi_chart": rasi,
        "navamsa_chart": navamsa,
        "dasamsa_chart": dasamsa,
        "transit_chart": transit_rasi,
        "transit_positions": transit_positions,
        "mahadasha": md_table,
        "antardasha": ad_table,
        "natal_strengths": natal_strengths,
        "transit_events": transit_events,
        "influence_rows_now": rows,
        "current_md": current_md,
        "current_ad": current_ad,
        "location": {
            "lat": lat,
            "lon": lon,
            "tz_name": tz_name,
            "tz_offset": tz_offset
        }
    }

# ----------------------- OPTIONAL: QUICK DEMO -----------------------

if __name__ == "__main__":
    # Example usage:
    birth_y, birth_m, birth_d = 1990, 1, 1
    birth_h, birth_min = 12, 0
    place = "Kodinar, Gujarat"

    _ = build_charts(birth_y, birth_m, birth_d, birth_h, birth_min, place)
    print("ascendant: ", _['ascendant_longitude'])
    print("/n")
    print("/n")
    print("rasi_chart: ", _['rasi_chart'])
    print("/n")
    print("/n")
    print("navamsa_chart: ", _['navamsa_chart'])
    print("/n")
    print("/n")
    print("dasamsa_chart: ", _['dasamsa_chart'])
    print("/n")
    print("/n")
    print("transit_chart: ", _['transit_chart'])
    print("/n")
    print("/n")
    print("transit_positions: ", _['transit_positions'])
    print("/n")
    print("/n")
    print("mahadasha: ", _['mahadasha'])
    print("/n")
    print("/n")
    print("antardasha: ", _['antardasha'])
    print("/n")
    print("/n")
    print("natal_strengths: ", _['natal_strengths'])
    print("/n")
    print("/n")
    print("transit_events: ", _['transit_events'])
    print("/n")
    print("/n")
    print("influence_rows_now: ", _['influence_rows_now'])
    print("/n")
    print("/n")
    print("current_md: ", _['current_md'])
    print("/n")
    print("/n")
    print("current_ad: ", _['current_ad'])
    print("/n")
    print("/n")
    print("location: ", _['location'])

