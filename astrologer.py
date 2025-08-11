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
import datetime
from geopy.geocoders import Nominatim
from collections import namedtuple
from jhora.panchanga import drik
from jhora.horoscope.chart import charts

from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
import pytz

EPHE_PATH = r"ephe"
AYANAMSA_MODE = swe.SIDM_LAHIRI

def jd_ut(y, m, d, hh, mi, tz):
    ut_hour = hh + mi/60 - tz
    return swe.julday(y, m, d, ut_hour)

def get_location_details(city_name):
    # Step 1: Geocode city to lat/lon
    geolocator = Nominatim(user_agent="astro_app")
    location = geolocator.geocode(city_name)
    if location:
        lat, lon = location.latitude, location.longitude
    else:
        raise ValueError(f"City not found: {city_name}")

    # Step 2: Get timezone name from lat/lon
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)

    # Step 3: Get current UTC offset in hours
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    offset_hours = now.utcoffset().total_seconds() / 3600

    return {
        "city": city_name,
        "latitude": lat,
        "longitude": lon,
        "timezone_name": tz_name,
        "timezone_offset": offset_hours
    }

# Nakshatra lords in Vimshottari order
VIM_DASHA_LORDS = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
VIM_DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}

def get_vimshottari_dasha_table(jd_birth, tz_offset):
    # Get Moon longitude at birth
    moon_long, _ = swe.calc_ut(jd_birth, swe.MOON)[:2]
    moon_long = moon_long % 360.0

    # Find Nakshatra index (0–26)
    nakshatra_index = int(moon_long // (360 / 27))
    nakshatra_lord = VIM_DASHA_LORDS[nakshatra_index % 9]

    # How far into Nakshatra?
    nakshatra_start_deg = nakshatra_index * (360 / 27)
    fraction_left = 1 - ((moon_long - nakshatra_start_deg) / (360 / 27))

    # Remaining years of starting Mahadasha
    remaining_years = VIM_DASHA_YEARS[nakshatra_lord] * fraction_left

    # Build table
    start_date = swe.revjul(jd_birth + tz_offset/24)  # (year, month, day, frac_day)
    start_dt = datetime(start_date[0], start_date[1], int(start_date[2] + start_date[3]))

    table = []
    lord_index = VIM_DASHA_LORDS.index(nakshatra_lord)

    years_remaining = remaining_years
    current_dt = start_dt

    for i in range(9):
        lord = VIM_DASHA_LORDS[(lord_index + i) % 9]
        dasha_years = years_remaining if i == 0 else VIM_DASHA_YEARS[lord]
        end_dt = current_dt + timedelta(days=dasha_years * 365.25)
        table.append({"lord": lord, "start": current_dt.date(), "end": end_dt.date()})
        current_dt = end_dt
        years_remaining = None  # only partial for first lord

    return table

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

def get_transit_events(jd_start, jd_end, place_obj, ascendant_long):
    """
    Generates planetary transit dates from sign-to-sign and house-to-house.
    jd_start, jd_end -> Julian Days for start & end of period
    place_obj -> birth place object (with lat/lon/tz)
    ascendant_long -> Lagna longitude (from birth chart) to calculate houses
    """
    swe.set_ephe_path(EPHE_PATH)
    events = []

    for planet_name, planet_id in PLANETS.items():
        prev_sign = None
        prev_house = None
        jd = jd_start

        while jd <= jd_end:
            lon, _ = swe.calc_ut(jd, planet_id)[:2]
            lon = lon % 360

            # Sign index (0–11)
            sign_index = int(lon // 30)

            # House index (0–11) relative to Lagna
            house_index = int(((lon - ascendant_long) % 360) // 30)

            # Detect sign change
            if prev_sign is not None and sign_index != prev_sign:
                y, m, d, _ = swe.revjul(jd, swe.GREG_CAL)
                events.append({
                    "planet": planet_name,
                    "type": "sign_change",
                    "date": datetime(y, m, d).date(),
                    "from": SIGNS[prev_sign],
                    "to": SIGNS[sign_index]
                })

            # Detect house change
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
            jd += 1  # Step 1 day

    return events

def calculate_shadbala(jd_birth, tz_offset):
    # Placeholder: assigns 0–60 scale based on exaltation/debilitation
    planet_strengths = {}
    exaltation_points = {swe.SUN: 10, swe.MOON: 30, swe.MARS: 28, swe.MERCURY: 15, swe.JUPITER: 5, swe.VENUS: 27, swe.SATURN: 20}
    debilitation_points = {swe.SUN: 190, swe.MOON: 210, swe.MARS: 118, swe.MERCURY: 195, swe.JUPITER: 245, swe.VENUS: 177, swe.SATURN: 140}

    for planet in [swe.SUN, swe.MOON, swe.MARS, swe.MERCURY, swe.JUPITER, swe.VENUS, swe.SATURN]:
        lon, _ = swe.calc_ut(jd, planet)[:2]
        lon = lon % 360
        if abs(lon - exaltation_points[planet]) < abs(lon - debilitation_points[planet]):
            score = 50  # strong
        else:
            score = 20  # weak
        planet_strengths[planet] = score

    return planet_strengths

def build_charts(y, m, d, hh, mi, place_name):
    # ---- inputs ----
    location_details = get_location_details(place_name)
    lat, lon, timezone, tz_name = (
        location_details["latitude"],
        location_details["longitude"],
        location_details["timezone_offset"],
        location_details["timezone_name"]
    )

    Place = namedtuple("Place", ["name", "latitude", "longitude", "timezone"])
    place_obj = Place(place_name, lat, lon, timezone)
    place_tuple = (place_obj.name, place_obj.latitude, place_obj.longitude, place_obj.timezone)

    # ---- Swiss Ephemeris ----
    swe.set_ephe_path(EPHE_PATH)
    swe.set_sid_mode(AYANAMSA_MODE, 0, 0)

    jd_birth = jd_ut(y, m, d, hh, mi, timezone)

    # 1. Lagna Chart (Ascendant info)
    asc = drik.ascendant(jd_birth, place_tuple)

    # 2. Rāśi Chart (D1)
    rasi = charts.rasi_chart(jd_birth, place_obj)

    # 3. Navāṁśa Chart (D9)
    try:
        navamsa = charts.navamsa_chart(rasi)
    except TypeError:
        navamsa = charts.divisional_positions_from_rasi_positions(rasi, 9)

    # 4. Daśāṁśa Chart (D10)
    try:
        dasamsa = charts.dasamsa_chart(rasi)
    except TypeError:
        dasamsa = charts.divisional_positions_from_rasi_positions(rasi, 10)

    # 5. Transit Chart (today's positions)
    now = datetime.now()
    jd_today = jd_ut(now.year, now.month, now.day, now.hour, now.minute, timezone)
    transit_rasi = charts.rasi_chart(jd_today, place_obj)

    # 6. Transit Calculations (example: current planetary longitudes)
    transit_positions = {planet: drik.sidereal_longitude(jd_today, planet) for planet in range(9)}

    # 7. Vimśottarī Daśā Table
    vimshottari_table = get_vimshottari_dasha_table(jd_birth, timezone)

    # 8. Śaḍbala (if available)
    sadbala = calculate_shadbala(jd_birth, timezone)

    # 9. Transit events
    # Transit events for the next 3 years
    jd_today = jd_ut(now.year, now.month, now.day, now.hour, now.minute, timezone)
    jd_next_year = jd_ut(now.year + 3, now.month, now.day, now.hour, now.minute, timezone)

    transit_events = get_transit_events(jd_today, jd_next_year, place_obj, asc['ascendant'])


    return {
        "ascendant": asc,
        "rasi_chart": rasi,
        "navamsa_chart": navamsa,
        "dasamsa_chart": dasamsa,
        "transit_chart": transit_rasi,
        "transit_positions": transit_positions,
        "vimsottari_dasha": vimshottari_table,
        "shadbala": sadbala,
        "transit_events": transit_events,
        "location": {
            "lat": lat,
            "lon": lon,
            "tz_name": tz_name,
            "tz_offset": timezone
        }
    }

if __name__ == "__main__":
    # Example: Delhi
    charts = build_charts(2025, 1, 2, 22, 15, "Kodinar, Gujarat")

    print(charts)