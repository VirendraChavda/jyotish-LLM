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
from collections.abc import Mapping, Sequence

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

# ------------------------ Cache important functions ------------------------

from functools import lru_cache
_GEOLOC = Nominatim(user_agent="astro_app")
_TF = TimezoneFinder()

@lru_cache(maxsize=1024)
def _swe_calc_ut(jd, pid, flg=0):
    # Return immutable tuple so it is cacheable
    xx, retflag = swe.calc_ut(jd, pid, flg)
    # Some swe builds return lists; normalize to tuple
    return (tuple(xx), retflag)

@lru_cache(maxsize=256)
def _revjul(jd):
    return swe.revjul(jd, swe.GREG_CAL)

# ------------------------ TIME & GEO HELPERS ------------------------

def jd_ut(y, m, d, hh, mi, tz_hours):
    """Julian day at UT for local date/time with given time zone hours."""
    ut_hour = hh + mi/60.0 - tz_hours
    return swe.julday(y, m, d, ut_hour, swe.GREG_CAL)

def revjul_to_datetime_utc(jd):
    """Swiss Ephem revjul -> Python datetime in UTC. revjul returns hours, not fractional days."""
    y, m, d, ut_hours = swe.revjul(jd, swe.GREG_CAL)
    return datetime(y, m, d) + timedelta(hours=ut_hours)

@lru_cache(maxsize=256)
def get_location_details(city_name):
    """Geocode + timezone + CURRENT UTC offset hours (cached)."""
    location = _GEOLOC.geocode(city_name)
    if not location:
        raise ValueError(f"City not found: {city_name}")

    lat, lon = location.latitude, location.longitude

    tz_name = _TF.timezone_at(lat=lat, lng=lon)
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
    xx, _ = _swe_calc_ut(jd_when, pid, swe.FLG_SPEED)
    lon = normalize_deg(xx[0])
    lon_speed = xx[3]
    house = house_index_from_asc(lon, asc_long) + 1
    retro = lon_speed < 0
    return house, retro, lon

def _sign_index(lon): return int(normalize_deg(lon) // 30)
def _house_index_from_asc_deg(lon, asc_long): return int(((normalize_deg(lon) - asc_long) % 360) // 30)

def _refine_change_day(jd_lo, jd_hi, planet_id, asc_long, check_sign=True):
    """Binary search to the day a sign/house index changes."""
    lo, hi = jd_lo, jd_hi
    xx_lo, _ = _swe_calc_ut(lo, planet_id, 0)
    val_lo = _sign_index(xx_lo[0]) if check_sign else _house_index_from_asc_deg(xx_lo[0], asc_long)
    while hi - lo > 0.5:  # ~12h precision; good enough to land on correct date
        mid = (lo + hi) / 2.0
        xx_mid, _ = _swe_calc_ut(mid, planet_id, 0)
        val_mid = _sign_index(xx_mid[0]) if check_sign else _house_index_from_asc_deg(xx_mid[0], asc_long)
        if val_mid != val_lo:
            hi = mid
        else:
            lo = mid
    return hi  # jd close to boundary

def get_transit_events(jd_start, jd_end, ascendant_long, step_days=5.0):
    """
    Sign and house change dates for the classic 7 planets, using coarse steps with refinement.
    ~5-10x fewer ephemeris calls than daily stepping.
    """
    events = []
    for planet_name, planet_id in PLANETS.items():
        # Initialize at start
        xx0, _ = _swe_calc_ut(jd_start, planet_id, 0)
        prev_sign = _sign_index(xx0[0])
        prev_house = _house_index_from_asc_deg(xx0[0], ascendant_long)

        jd = jd_start + step_days
        while jd <= jd_end + 1e-6:
            xx, _ = _swe_calc_ut(jd, planet_id, 0)
            sign_index = _sign_index(xx[0])
            house_index = _house_index_from_asc_deg(xx[0], ascendant_long)

            # Sign change?
            if sign_index != prev_sign:
                # refine in (jd - step_days, jd]
                jd_change = _refine_change_day(jd - step_days, jd, planet_id, ascendant_long, check_sign=True)
                y, m, d, _ = _revjul(jd_change)
                events.append({
                    "planet": planet_name,
                    "type": "sign_change",
                    "date": datetime(y, m, d).date(),
                    "from": SIGNS[prev_sign],
                    "to": SIGNS[sign_index]
                })
                prev_sign = sign_index

            # House change?
            if house_index != prev_house:
                jd_change = _refine_change_day(jd - step_days, jd, planet_id, ascendant_long, check_sign=False)
                y, m, d, _ = _revjul(jd_change)
                events.append({
                    "planet": planet_name,
                    "type": "house_change",
                    "date": datetime(y, m, d).date(),
                    "from_house": prev_house + 1,
                    "to_house": house_index + 1
                })
                prev_house = house_index

            jd += step_days
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

def transit_context_factor(planet_name, jd_when, asc_long, sun_lon=None):
    pid = PLANETS[planet_name]
    house, retro, lon = get_transit_house_for_planet(jd_when, asc_long, pid)

    # base house factor
    h_idx = house - 1
    if h_idx in (0, 3, 6, 9):       base = 1.0
    elif h_idx in (1, 4, 7, 10):    base = 0.6
    else:                           base = 0.3

    bonus = 0.1 if retro else 0.0

    combust_penalty = 1.0
    if planet_name != "Moon":
        if sun_lon is None:
            sun_xx, _ = _swe_calc_ut(jd_when, swe.SUN)
            sun_lon = sun_xx[0]
        def circ_dist(a,b): return abs((a - b + 180.0) % 360.0 - 180.0)
        if circ_dist(normalize_deg(lon), normalize_deg(sun_lon)) < 8.0:
            combust_penalty = 0.8

    factor = (base + bonus) * combust_penalty
    return max(0.1, min(1.2, factor)), house

def influence_scores_table(natal_strengths_0_60, current_md, current_ad, jd_when, asc_long):
    rows = []
    sun_xx, _ = _swe_calc_ut(jd_when, swe.SUN)
    sun_lon = sun_xx[0]
    for name in PLANETS.keys():
        natal = natal_strengths_0_60[name]
        nat_norm = max(0.0, min(1.0, natal / 60.0))
        w = dasha_weight_for_planet(name, current_md, current_ad)
        trans_factor, house = transit_context_factor(name, jd_when, asc_long, sun_lon=sun_lon)
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


# ----------------------------- PLANET NAME REMAPING ----------------

PLANET_NAMES = {
    0: "Sun", 1: "Moon", 2: "Mars", 3: "Mercury", 4: "Jupiter",
    5: "Venus", 6: "Saturn", 7: "Rahu", 8: "Ketu"
    }

def map_planet_key(k):
    # keep Lagna variants as "Lagna"
    if k in ("L", "Asc", "Lagna"):
        return "Lagna"
    # handle int and digit-string keys
    if isinstance(k, int):
        return PLANET_NAMES.get(k, k)
    if isinstance(k, str) and k.isdigit():
        return PLANET_NAMES.get(int(k), k)
    return k

def rename_planet_keys(obj):
    """
    Recursively rename planet-number keys/labels to names.
    - dict: rename keys and recurse values
    - list/tuple: if looks like [key, value] pair, rename the first element; else recurse into elements
    - everything else: return as-is
    """

    if isinstance(obj, Mapping):
        return {map_planet_key(k): rename_planet_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = []
        for item in obj:
            if (isinstance(item, (list, tuple))
                    and len(item) == 2):
                # likely like [planet_key, value]
                new_key = map_planet_key(item[0])
                out.append([new_key, rename_planet_keys(item[1])])
            else:
                out.append(rename_planet_keys(item))
        return out
    if isinstance(obj, tuple):
        # preserve tuple type
        if len(obj) == 2:
            return (rename_planet_keys(obj[0]), rename_planet_keys(obj[1]))
        return tuple(rename_planet_keys(x) for x in obj)
    return obj  # numbers/strings/etc.

# ----------------------------- MAIN API -----------------------------

def build_charts(y, m, d, hh, mi, birth_place_name, current_loc, include_transit_events=True, transit_span_years=3, transit_step_days=5.0):
    """
    Builds charts + dasha + transit + influence table ingredients.
    Returns dict with everything, plus prints the influence table for 'now'.
    """

    # Setup ephemeris
    swe.set_ephe_path(EPHE_PATH)
    swe.set_sid_mode(AYANAMSA_MODE, 0, 0)

    # Location & timezone
    birth_location_details = get_location_details(birth_place_name)
    birth_lat, birth_lon, birth_tz_offset, birth_tz_name = (
        birth_location_details["latitude"],
        birth_location_details["longitude"],
        birth_location_details["timezone_offset"],
        birth_location_details["timezone_name"]
    )

    current_location_details = get_location_details(current_loc)
    current_lat, current_lon, current_tz_offset, current_tz_name = (
        current_location_details["latitude"],
        current_location_details["longitude"],
        current_location_details["timezone_offset"],
        current_location_details["timezone_name"]
    )

    Place = namedtuple("Place", ["name", "latitude", "longitude", "timezone"])
    place_obj = Place(birth_place_name, birth_lat, birth_lon, birth_tz_offset)
    place_tuple = (place_obj.name, place_obj.latitude, place_obj.longitude, place_obj.timezone)

    # Birth JD in UT
    jd_birth = jd_ut(y, m, d, hh, mi, birth_tz_offset)

    # Ascendant (Swiss Ephemeris) — robust float, avoids list/dict ambiguity
    # houses() returns (cusps[1..12], ascmc[0..9]); ascmc[0] is Ascendant longitude
    cusps, ascmc = swe.houses(jd_birth, birth_lat, birth_lon, b'W')  # P=Placidus (Asc longitude is independent of house system)
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
    current_tz_obj = pytz.timezone(current_tz_name)
    now_local = datetime.now(current_tz_obj)
    jd_today = jd_ut(now_local.year, now_local.month, now_local.day, now_local.hour, now_local.minute, current_tz_offset)
    transit_rasi = charts.rasi_chart(jd_today, place_obj)
    transit_positions = {planet: drik.sidereal_longitude(jd_today, planet) for planet in range(9)}

    # Dasha tables (MD + AD)
    md_table, ad_table = get_all_mahadasha_antardasha(jd_birth)

    # Current MD/AD at "now" (use UTC)
    now_utc = datetime.now()
    current_md, current_ad = find_current_md_ad(md_table, ad_table, now_utc)

    # Natal strength (simple dignity + angularity)
    natal_strengths = calculate_natal_strength_simple(jd_birth, asc_long)

    # Transit events (2 years ahead)
    transit_events = []
    if include_transit_events and transit_span_years > 0:
        end_year = now_local.year + transit_span_years
        jd_future = jd_ut(end_year, now_local.month, now_local.day, now_local.hour, now_local.minute, current_tz_offset)
        transit_events = get_transit_events(jd_today, jd_future, asc_long, step_days=transit_step_days)

    # Influence table for now
    rows = influence_scores_table(natal_strengths, current_md, current_ad, jd_today, asc_long)

    # Conduct planet maping
    rasi = rename_planet_keys(rasi)
    navamsa = rename_planet_keys(navamsa)
    dasamsa = rename_planet_keys(dasamsa)
    transit_rasi = rename_planet_keys(transit_rasi)
    md_table = rename_planet_keys(md_table)
    ad_table = rename_planet_keys(ad_table)
    natal_strengths = rename_planet_keys(natal_strengths)
    transit_events = rename_planet_keys(transit_events)
    rows = rename_planet_keys(rows)
    current_md = rename_planet_keys(current_md)
    current_ad = rename_planet_keys(current_ad)

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
        "influence_now": rows,
        "current_md": current_md,
        "current_ad": current_ad,
        "birth_location": {
            "lat": birth_lat,
            "lon": birth_lon,
            "tz_name": birth_tz_name,
            "tz_offset": birth_tz_offset
        },
        "current_location": {
            "lat": current_lat,
            "lon": current_lon,
            "tz_name": current_tz_name,
            "tz_offset": current_tz_offset
        },
        "birth_datetime": datetime(y, m, d, hh, mi, tzinfo=pytz.timezone(birth_tz_name)),
        "today_datetime": now_local
    }

# Create chart inputs for LLM

def input_for_LLM(username: str, name: str, birth_date_iso: str, birth_time_str: str, birth_place: str, current_place: str,
                  include_transit_events=True, transit_span_years=3):
    try:
        y, m, d = map(int, birth_date_iso.split("-"))
        hh, mm = map(int, birth_time_str.split(":"))

        out = build_charts(y, m, d, hh, mm, birth_place, current_place, include_transit_events=include_transit_events, transit_span_years=transit_span_years)
        return username, name, out

    except:
        print("Error")

# ----------------------- OPTIONAL: QUICK DEMO -----------------------

if __name__ == "__main__":
    # Example usage:
    # birth_y, birth_m, birth_d = 1992, 10, 15
    # birth_h, birth_min = 23, 30
    # place = "Kodinar, Gujarat"
    # current_place = "Colchester, Essex"

    # _ = build_charts(birth_y, birth_m, birth_d, birth_h, birth_min, place, current_place, include_transit_events=True, transit_span_years=3)
    # print("ascendant: ", _['ascendant_longitude'])
    # print("/n")
    # print("/n")
    # print("rasi_chart: ", _['rasi_chart'])
    # print("/n")
    # print("/n")
    # print("navamsa_chart: ", _['navamsa_chart'])
    # print("/n")
    # print("/n")
    # print("dasamsa_chart: ", _['dasamsa_chart'])
    # print("/n")
    # print("/n")
    # print("transit_chart: ", _['transit_chart'])
    # print("/n")
    # print("/n")
    # print("transit_positions: ", _['transit_positions'])
    # print("/n")
    # print("/n")
    # print("mahadasha: ", _['mahadasha'])
    # print("/n")
    # print("/n")
    # print("antardasha: ", _['antardasha'])
    # print("/n")
    # print("/n")
    # print("natal_strengths: ", _['natal_strengths'])
    # print("/n")
    # print("/n")
    # print("transit_events: ", _['transit_events'])
    # print("/n")
    # print("/n")
    # print("influence_rows_now: ", _['influence_now'])
    # print("/n")
    # print("/n")
    # print("current_md: ", _['current_md'])
    # print("/n")
    # print("/n")
    # print("current_ad: ", _['current_ad'])
    # print("/n")
    # print("/n")
    # print("location: ", _['birth_location'])
    # print("/n")
    # print("/n")
    # print("location: ", _['current_location'])

    username, name, charts = input_for_LLM("viren", "virendrasinh chavda", "1992-10-15", "00:30", "kodinar, india", "colchester, united kingdom",
                                           include_transit_events=True, transit_span_years=3)
    print(charts)
