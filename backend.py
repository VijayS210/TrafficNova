import os
import sqlite3
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
import joblib
import numpy as np
import pandas as pd
import requests
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "traffic_nova_multicity_secret_2026"
DB_PATH = "traffic_app.db"


# =========================================================
# 1. SQLITE DATABASE SETUP & SAFE MIGRATION
# =========================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            preferred_city TEXT DEFAULT 'Bengaluru',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            city_name TEXT DEFAULT 'Bengaluru',
            origin_name TEXT NOT NULL,
            dest_name TEXT NOT NULL,
            travel_date TEXT NOT NULL,
            travel_hour INTEGER NOT NULL,
            weather_label TEXT NOT NULL,
            traffic_status TEXT NOT NULL,
            parking_occ REAL NOT NULL,
            primary_eta INTEGER NOT NULL,
            searched_at TEXT NOT NULL,
            iso_timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    user_cols = [row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()]
    if "preferred_city" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN preferred_city TEXT DEFAULT 'Bengaluru'")

    hist_cols = [row[1] for row in cursor.execute("PRAGMA table_info(search_history)").fetchall()]
    if "city_name" not in hist_cols:
        cursor.execute("ALTER TABLE search_history ADD COLUMN city_name TEXT DEFAULT 'Bengaluru'")
    if "iso_timestamp" not in hist_cols:
        cursor.execute("ALTER TABLE search_history ADD COLUMN iso_timestamp TEXT")

    existing_demo = cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@smarttraffic.in",)).fetchone()
    if not existing_demo:
        cursor.execute(
            "INSERT INTO users (full_name, email, password_hash, preferred_city, created_at) VALUES (?, ?, ?, ?, ?)",
            ("Demo Commuter", "demo@smarttraffic.in", generate_password_hash("demo123"), "Bengaluru", datetime.now().strftime("%Y-%m-%d")),
        )

    conn.commit()
    conn.close()


init_db()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# =========================================================
# 2. LOAD MULTI-CITY XGBOOST MODELS
# =========================================================
DEMO_MODE = False
try:
    road_model = joblib.load("models/road_congestion_model.pkl")
    parking_model = joblib.load("models/parking_occupancy_model.pkl")
except Exception:
    DEMO_MODE = True


# =========================================================
# 3. MULTI-CITY HUBS & BYPASS CORRIDORS (6 INDIAN METROS)
# =========================================================
CITY_DATA = {
    "Bengaluru": {
        "city_id": 0,
        "center": [12.9716, 77.5946],
        "bypasses": [(13.0358, 77.5970), (12.9279, 77.6271), (12.9698, 77.7500), (12.9141, 77.6412)],
        "locations": {
            "Silk Board Junction": {"lat": 12.9177, "lon": 77.6238, "zone_id": 0},
            "KR Puram Hanging Bridge": {"lat": 13.0012, "lon": 77.6865, "zone_id": 0},
            "Hebbal Flyover": {"lat": 13.0358, "lon": 77.5970, "zone_id": 0},
            "MG Road / Brigade Road": {"lat": 12.9756, "lon": 77.6066, "zone_id": 1},
            "Koramangala 80 Feet Road": {"lat": 12.9352, "lon": 77.6245, "zone_id": 1},
            "Indiranagar 100 Feet Road": {"lat": 12.9784, "lon": 77.6408, "zone_id": 1},
            "Commercial Street / Shivajinagar": {"lat": 12.9822, "lon": 77.6083, "zone_id": 2},
            "Jayanagar 4th Block": {"lat": 12.9299, "lon": 77.5838, "zone_id": 2},
            "Malleshwaram / Orion Mall": {"lat": 13.0110, "lon": 77.5549, "zone_id": 2},
            "Whitefield / ITPL": {"lat": 12.9698, "lon": 77.7500, "zone_id": 3},
            "Electronic City Phase 1": {"lat": 12.8452, "lon": 77.6602, "zone_id": 3},
            "Marathahalli Outer Ring Road": {"lat": 12.9592, "lon": 77.6974, "zone_id": 3},
            "Manyata Tech Park / Nagawara": {"lat": 13.0451, "lon": 77.6266, "zone_id": 3},
            "Majestic / KSR Railway Station": {"lat": 12.9767, "lon": 77.5713, "zone_id": 4},
            "Yelahanka New Town": {"lat": 13.0983, "lon": 77.5963, "zone_id": 4},
        },
    },
    "Hyderabad": {
        "city_id": 1,
        "center": [17.3850, 78.4867],
        "bypasses": [(17.4401, 78.3489), (17.4126, 78.4392), (17.3616, 78.4747), (17.4504, 78.3808)],
        "locations": {
            "Panjagutta Cross Roads": {"lat": 17.4254, "lon": 78.4506, "zone_id": 0},
            "Paradise Junction Secunderabad": {"lat": 17.4411, "lon": 78.4876, "zone_id": 0},
            "LB Nagar Ring Road Junction": {"lat": 17.3457, "lon": 78.5522, "zone_id": 0},
            "Banjara Hills Road No. 12": {"lat": 17.4138, "lon": 78.4398, "zone_id": 1},
            "Jubilee Hills Check Post": {"lat": 17.4293, "lon": 78.4116, "zone_id": 1},
            "Begumpet / Somajiguda": {"lat": 17.4375, "lon": 78.4606, "zone_id": 1},
            "Ameerpet Metro Hub": {"lat": 17.4375, "lon": 78.4482, "zone_id": 2},
            "Charminar / Laad Bazaar": {"lat": 17.3616, "lon": 78.4747, "zone_id": 2},
            "Kukatpally KPHB Forum": {"lat": 17.4849, "lon": 78.3888, "zone_id": 2},
            "HITEC City / Cyber Towers": {"lat": 17.4504, "lon": 78.3808, "zone_id": 3},
            "Gachibowli Financial District": {"lat": 17.4401, "lon": 78.3489, "zone_id": 3},
            "Madhapur Durgam Cheruvu": {"lat": 17.4318, "lon": 78.3871, "zone_id": 3},
            "Kondapur Botanical Garden": {"lat": 17.4589, "lon": 78.3662, "zone_id": 3},
            "Secunderabad Railway Station": {"lat": 17.4344, "lon": 78.5013, "zone_id": 4},
            "Uppal Metro Terminal": {"lat": 17.3984, "lon": 78.5583, "zone_id": 4},
        },
    },
    "Mumbai": {
        "city_id": 2,
        "center": [19.0760, 72.8777],
        "bypasses": [(19.0600, 72.8300), (19.1136, 72.8697), (19.0178, 72.8478), (19.1663, 72.8526)],
        "locations": {
            "Sion Circle Junction": {"lat": 19.0434, "lon": 72.8633, "zone_id": 0},
            "Andheri Western Express Highway": {"lat": 19.1197, "lon": 72.8544, "zone_id": 0},
            "Mahim Causeway": {"lat": 19.0402, "lon": 72.8404, "zone_id": 0},
            "Nariman Point / Marine Drive": {"lat": 18.9256, "lon": 72.8242, "zone_id": 1},
            "Lower Parel / Phoenix Palladium": {"lat": 18.9948, "lon": 72.8258, "zone_id": 1},
            "Fort / Flora Fountain": {"lat": 18.9322, "lon": 72.8328, "zone_id": 1},
            "Bandra Linking Road": {"lat": 19.0645, "lon": 72.8358, "zone_id": 2},
            "Colaba Causeway": {"lat": 18.9151, "lon": 72.8259, "zone_id": 2},
            "Dadar Plaza Market": {"lat": 19.0193, "lon": 72.8428, "zone_id": 2},
            "Bandra Kurla Complex (BKC)": {"lat": 19.0674, "lon": 72.8675, "zone_id": 3},
            "Powai Hiranandani Gardens": {"lat": 19.1176, "lon": 72.9060, "zone_id": 3},
            "Goregaon NESCO IT Park": {"lat": 19.1511, "lon": 72.8523, "zone_id": 3},
            "Malad Mindspace": {"lat": 19.1802, "lon": 72.8355, "zone_id": 3},
            "Chhatrapati Shivaji Maharaj Terminus": {"lat": 18.9398, "lon": 72.8355, "zone_id": 4},
            "Thane Ghodbunder Junction": {"lat": 19.2183, "lon": 72.9781, "zone_id": 4},
        },
    },
    "Chennai": {
        "city_id": 3,
        "center": [13.0827, 80.2707],
        "bypasses": [(13.0067, 80.2206), (13.0500, 80.2121), (12.9815, 80.2180), (13.0850, 80.2101)],
        "locations": {
            "Kathipara Flyover Guindy": {"lat": 13.0067, "lon": 80.2016, "zone_id": 0},
            "Koyambedu Roundtana": {"lat": 13.0694, "lon": 80.1948, "zone_id": 0},
            "Madhya Kailash Adyar": {"lat": 13.0063, "lon": 80.2478, "zone_id": 0},
            "Anna Salai / Mount Road": {"lat": 13.0604, "lon": 80.2558, "zone_id": 1},
            "Nungambakkam High Road": {"lat": 13.0587, "lon": 80.2433, "zone_id": 1},
            "Egmore / Pantheon Road": {"lat": 13.0732, "lon": 80.2609, "zone_id": 1},
            "T. Nagar Panagal Park": {"lat": 13.0418, "lon": 80.2341, "zone_id": 2},
            "Phoenix Marketcity Velachery": {"lat": 12.9915, "lon": 80.2166, "zone_id": 2},
            "Anna Nagar 2nd Avenue": {"lat": 13.0850, "lon": 80.2101, "zone_id": 2},
            "OMR Tidel Park / Thiruvanmiyur": {"lat": 12.9897, "lon": 80.2485, "zone_id": 3},
            "Sholinganallur IT Corridor": {"lat": 12.9010, "lon": 80.2279, "zone_id": 3},
            "DLF Cybercity Manapakkam": {"lat": 13.0213, "lon": 80.1755, "zone_id": 3},
            "Perungudi OMR Tech Hub": {"lat": 12.9654, "lon": 80.2461, "zone_id": 3},
            "Chennai Central Railway Station": {"lat": 13.0827, "lon": 80.2750, "zone_id": 4},
            "Tambaram GST Road Hub": {"lat": 12.9249, "lon": 80.1000, "zone_id": 4},
        },
    },
    "Delhi": {
        "city_id": 4,
        "center": [28.6139, 77.2090],
        "bypasses": [(28.5672, 77.2100), (28.6304, 77.2177), (28.5355, 77.3910), (28.4595, 77.0266)],
        "locations": {
            "ITO Crossing": {"lat": 28.6289, "lon": 77.2410, "zone_id": 0},
            "Dhaula Kuan Interchange": {"lat": 28.5918, "lon": 77.1615, "zone_id": 0},
            "Ashram Chowk Ring Road": {"lat": 28.5714, "lon": 77.2591, "zone_id": 0},
            "Connaught Place Rajiv Chowk": {"lat": 28.6328, "lon": 77.2197, "zone_id": 1},
            "Barakhamba Road / Mandi House": {"lat": 28.6280, "lon": 77.2289, "zone_id": 1},
            "Nehru Place Commercial Hub": {"lat": 28.5494, "lon": 77.2519, "zone_id": 1},
            "Karol Bagh Ajmal Khan Road": {"lat": 28.6448, "lon": 77.1901, "zone_id": 2},
            "Saket Select Citywalk": {"lat": 28.5286, "lon": 77.2193, "zone_id": 2},
            "Chandni Chowk / Red Fort": {"lat": 28.6506, "lon": 77.2303, "zone_id": 2},
            "Cyber Hub DLF Phase 2": {"lat": 28.4950, "lon": 77.0895, "zone_id": 3},
            "Noida Sector 62 Electronic City": {"lat": 28.6271, "lon": 77.3649, "zone_id": 3},
            "Aerocity IGI Business District": {"lat": 28.5503, "lon": 77.1206, "zone_id": 3},
            "Okhla Industrial Estate": {"lat": 28.5358, "lon": 77.2725, "zone_id": 3},
            "New Delhi Railway Station": {"lat": 28.6429, "lon": 77.2191, "zone_id": 4},
            "Anand Vihar ISBT Terminal": {"lat": 28.6469, "lon": 77.3160, "zone_id": 4},
        },
    },
    "Kolkata": {
        "city_id": 5,
        "center": [22.5726, 88.3639],
        "bypasses": [(22.5726, 88.4017), (22.5448, 88.3426), (22.5179, 88.3834), (22.6200, 88.4300)],
        "locations": {
            "Ultadanga Hudco Crossing": {"lat": 22.5937, "lon": 88.3932, "zone_id": 0},
            "Science City EM Bypass": {"lat": 22.5396, "lon": 88.3963, "zone_id": 0},
            "Chingrighata Flyover": {"lat": 22.5625, "lon": 88.4071, "zone_id": 0},
            "Park Street / Camac Street": {"lat": 22.5529, "lon": 88.3532, "zone_id": 1},
            "BBD Bagh / Dalhousie Square": {"lat": 22.5715, "lon": 88.3486, "zone_id": 1},
            "Esplanade / Chowringhee": {"lat": 22.5646, "lon": 88.3517, "zone_id": 1},
            "New Market / Lindsay Street": {"lat": 22.5601, "lon": 88.3526, "zone_id": 2},
            "South City Mall Jadavpur": {"lat": 22.5015, "lon": 88.3618, "zone_id": 2},
            "Gariahat Junction": {"lat": 22.5186, "lon": 88.3658, "zone_id": 2},
            "Salt Lake Sector V IT Hub": {"lat": 22.5735, "lon": 88.4331, "zone_id": 3},
            "New Town Action Area 1": {"lat": 22.5816, "lon": 88.4598, "zone_id": 3},
            "Eco Park / Biswa Bangla Gate": {"lat": 22.6012, "lon": 88.4665, "zone_id": 3},
            "Rajarhat IT Corridor": {"lat": 22.6214, "lon": 88.4522, "zone_id": 3},
            "Howrah Junction Station": {"lat": 22.5839, "lon": 88.3428, "zone_id": 4},
            "Sealdah Railway Terminal": {"lat": 22.5667, "lon": 88.3705, "zone_id": 4},
        },
    },
}

WEATHER_LABELS = {0: "Clear Sky", 1: "Cloudy / Humid", 2: "Heavy Monsoon Rain"}
STATUS_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
CITY_NETWORK_CACHE = {}


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def geocode_place(place_name, city_name):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{place_name}, {city_name}, India", "format": "json", "limit": 1}
        headers = {"User-Agent": "TrafficNova-App/2.5"}
        res = requests.get(url, params=params, headers=headers, timeout=5).json()
        if res:
            return float(res[0]["lat"]), float(res[0]["lon"])
    except Exception:
        pass
    return CITY_DATA[city_name]["center"]


def get_routes(start_lat, start_lon, end_lat, end_lon, bypass_waypoints, stops=None):
    stops = stops or []
    parsed_routes = []

    coord_chain = [f"{start_lon},{start_lat}"]
    for s in stops:
        coord_chain.append(f"{s['lon']},{s['lat']}")
    coord_chain.append(f"{end_lon},{end_lat}")
    waypoints_str = ";".join(coord_chain)

    alt_param = "false" if stops else "3"
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{waypoints_str}?overview=full&geometries=geojson&alternatives={alt_param}"
    )
    try:
        res = requests.get(url, timeout=6).json()
        if res.get("code") == "Ok":
            for r in res["routes"]:
                coords = [[c[1], c[0]] for c in r["geometry"]["coordinates"]]
                parsed_routes.append({
                    "coords": coords,
                    "distance_km": round(r["distance"] / 1000.0, 1),
                    "duration_min": round(r["duration"] / 60.0),
                })
    except Exception:
        pass

    if len(parsed_routes) == 1:
        primary_dist = parsed_routes[0]["distance_km"]
        best_bypass = None
        prefix_chain = [f"{start_lon},{start_lat}"] + [f"{s['lon']},{s['lat']}" for s in stops]
        prefix_str = ";".join(prefix_chain)

        for wp_lat, wp_lon in bypass_waypoints:
            wp_url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{prefix_str};{wp_lon},{wp_lat};{end_lon},{end_lat}"
                f"?overview=full&geometries=geojson"
            )
            try:
                wp_res = requests.get(wp_url, timeout=5).json()
                if wp_res.get("code") == "Ok" and wp_res["routes"]:
                    r = wp_res["routes"][0]
                    dist_km = round(r["distance"] / 1000.0, 1)
                    if abs(dist_km - primary_dist) >= 1.0 and dist_km <= primary_dist * 1.45:
                        candidate = {
                            "coords": [[c[1], c[0]] for c in r["geometry"]["coordinates"]],
                            "distance_km": dist_km,
                            "duration_min": round(r["duration"] / 60.0),
                        }
                        if best_bypass is None or candidate["duration_min"] < best_bypass["duration_min"]:
                            best_bypass = candidate
            except Exception:
                continue
        if best_bypass:
            parsed_routes.append(best_bypass)

    if not parsed_routes:
        fallback_coords = [[start_lat, start_lon]] + [[s["lat"], s["lon"]] for s in stops] + [[end_lat, end_lon]]
        mid_lat = (start_lat + end_lat) / 2 + 0.012
        mid_lon = (start_lon + end_lon) / 2 - 0.012
        parsed_routes = [
            {"coords": fallback_coords, "distance_km": round(12.0 + len(stops) * 3.5, 1), "duration_min": 28 + len(stops) * 9},
            {"coords": [[start_lat, start_lon], [mid_lat, mid_lon], [end_lat, end_lon]], "distance_km": 14.1, "duration_min": 32},
        ]
    return parsed_routes


def run_ml_prediction(city_id, zone_id, hour, day_of_week, is_weekend, weather_code):
    if not DEMO_MODE:
        input_df = pd.DataFrame([{
            "city_id": city_id,
            "zone_id": zone_id,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "weather_code": weather_code,
        }])
        road_level = int(road_model.predict(input_df)[0])
        parking_occ = round(float(np.clip(parking_model.predict(input_df)[0], 5, 99)), 1)
        return road_level, parking_occ

    peak = 1 if (8 <= hour <= 11 and not is_weekend) or (17 <= hour <= 21) else 0
    score = 22 + (city_id * 2) + (zone_id * 4) + (peak * 36) + (weather_code * 12)
    road_level = 2 if score >= 68 else (1 if score >= 42 else 0)
    parking_occ = round(min(98.0, 28.0 + (peak * 38) + (weather_code * 7) + (zone_id * 3)), 1)
    return road_level, parking_occ


def get_city_road_network(city_name, city_cfg, hour, day_of_week, is_weekend, weather_code):
    if city_name not in CITY_NETWORK_CACHE:
        locs = list(city_cfg["locations"].items())
        pairs = [
            (0, 3), (3, 5), (5, 9), (9, 10), (10, 11),
            (11, 1), (1, 12), (12, 2), (2, 14), (2, 8),
            (8, 13), (13, 3), (13, 7), (7, 4), (4, 0),
            (6, 3), (6, 5), (1, 9), (4, 11)
        ]
        corridors = []
        for idx_a, idx_b in pairs:
            if idx_a < len(locs) and idx_b < len(locs):
                name_a, info_a = locs[idx_a]
                name_b, info_b = locs[idx_b]
                url = (
                    f"http://router.project-osrm.org/route/v1/driving/"
                    f"{info_a['lon']},{info_a['lat']};{info_b['lon']},{info_b['lat']}"
                    f"?overview=simplified&geometries=geojson"
                )
                try:
                    res = requests.get(url, timeout=4).json()
                    if res.get("code") == "Ok" and res["routes"]:
                        coords = [[c[1], c[0]] for c in res["routes"][0]["geometry"]["coordinates"]]
                        corridors.append({
                            "name": f"{name_a.split('/')[0]} – {name_b.split('/')[0]}",
                            "zone_id": info_b["zone_id"],
                            "coords": coords,
                        })
                except Exception:
                    corridors.append({
                        "name": f"{name_a.split('/')[0]} – {name_b.split('/')[0]}",
                        "zone_id": info_b["zone_id"],
                        "coords": [[info_a["lat"], info_a["lon"]], [info_b["lat"], info_b["lon"]]],
                    })
        CITY_NETWORK_CACHE[city_name] = corridors

    city_id = city_cfg["city_id"]
    network_with_status = []
    for seg in CITY_NETWORK_CACHE[city_name]:
        lvl, _ = run_ml_prediction(city_id, seg["zone_id"], hour, day_of_week, is_weekend, weather_code)
        network_with_status.append({
            "name": seg["name"],
            "road_level": lvl,
            "coords": seg["coords"],
        })
    return network_with_status


# =========================================================
# 5. AUTHENTICATION, CITY SWITCHING & PAGE ROUTES
# =========================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action")
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        selected_city = request.form.get("city", "Bengaluru")
        if selected_city not in CITY_DATA:
            selected_city = "Bengaluru"

        conn = get_db()
        if action == "register":
            full_name = request.form.get("full_name", "").strip()
            if not full_name or not email or not password:
                error = "Please fill in all fields."
            else:
                try:
                    conn.execute(
                        "INSERT INTO users (full_name, email, password_hash, preferred_city, created_at) VALUES (?, ?, ?, ?, ?)",
                        (full_name, email, generate_password_hash(password), selected_city, datetime.now().strftime("%Y-%m-%d")),
                    )
                    conn.commit()
                    success = f"Account created for {selected_city}! Please sign in."
                except sqlite3.IntegrityError:
                    error = "An account with this email already exists."
        else:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                conn.execute("UPDATE users SET preferred_city = ? WHERE id = ?", (selected_city, user["id"]))
                conn.commit()
                session["user_id"] = user["id"]
                session["user_name"] = user["full_name"]
                session["user_email"] = user["email"]
                session["city"] = selected_city
                conn.close()
                return redirect(url_for("index"))
            else:
                error = "Invalid email or password."
        conn.close()

    return render_template("login.html", error=error, success=success, cities=list(CITY_DATA.keys()))


@app.route("/set-city/<city_name>")
@login_required
def set_city(city_name):
    if city_name in CITY_DATA:
        session["city"] = city_name
        conn = get_db()
        conn.execute("UPDATE users SET preferred_city = ? WHERE id = ?", (city_name, session["user_id"]))
        conn.commit()
        conn.close()
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    active_city = session.get("city", "Bengaluru")
    if active_city not in CITY_DATA:
        active_city = "Bengaluru"
    city_info = CITY_DATA[active_city]
    loc_names = list(city_info["locations"].keys())

    return render_template(
        "index.html",
        user_name=session["user_name"],
        user_email=session.get("user_email", "demo@smarttraffic.in"),
        active_city=active_city,
        cities=list(CITY_DATA.keys()),
        city_center=city_info["center"],
        origins=loc_names,
        destinations=loc_names,
        demo_mode=DEMO_MODE,
    )


@app.route("/history")
@login_required
def history_page():
    return render_template(
        "history.html",
        user_name=session["user_name"],
        user_email=session.get("user_email", "demo@smarttraffic.in"),
        active_city=session.get("city", "Bengaluru"),
    )


# =========================================================
# 6. PREDICTION & USER SEARCH HISTORY API ENDPOINTS
# =========================================================
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json()
    active_city = data.get("city") or session.get("city", "Bengaluru")
    if active_city not in CITY_DATA:
        active_city = "Bengaluru"
    session["city"] = active_city

    city_cfg = CITY_DATA[active_city]
    city_id = city_cfg["city_id"]
    locations = city_cfg["locations"]

    origin_choice = data.get("origin")
    custom_origin = data.get("custom_origin", "").strip()
    dest_choice = data.get("destination")
    custom_dest = data.get("custom_dest", "").strip()
    raw_stops = data.get("stops", [])
    date_str = data.get("date")
    hour = int(data.get("hour", 19))
    weather_code = int(data.get("weather", 0))
    save_history = data.get("save_history", True)

    default_names = list(locations.keys())

    if origin_choice == "CUSTOM" and custom_origin:
        o_lat, o_lon = geocode_place(custom_origin, active_city)
        origin_name = custom_origin
    else:
        o_info = locations.get(origin_choice, locations[default_names[0]])
        o_lat, o_lon = o_info["lat"], o_info["lon"]
        origin_name = origin_choice if origin_choice in locations else default_names[0]

    resolved_stops = []
    for idx, st in enumerate(raw_stops):
        st_choice = st.get("location", "")
        st_custom = st.get("custom_name", "").strip()
        if st_choice == "CUSTOM" and st_custom:
            s_lat, s_lon = geocode_place(st_custom, active_city)
            resolved_stops.append({"priority": idx + 1, "name": st_custom, "lat": s_lat, "lon": s_lon})
        elif st_choice in locations:
            s_info = locations[st_choice]
            resolved_stops.append({"priority": idx + 1, "name": st_choice, "lat": s_info["lat"], "lon": s_info["lon"]})

    if dest_choice == "CUSTOM" and custom_dest:
        d_lat, d_lon = geocode_place(custom_dest, active_city)
        dest_name = custom_dest
        zone_id = 3
    else:
        d_info = locations.get(dest_choice, locations[default_names[1]])
        d_lat, d_lon = d_info["lat"], d_info["lon"]
        dest_name = dest_choice if dest_choice in locations else default_names[1]
        zone_id = d_info["zone_id"]

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    road_level, parking_occ = run_ml_prediction(city_id, zone_id, hour, day_of_week, is_weekend, weather_code)
    est_delay = [5, 18, 42][road_level] + (weather_code * 6) + (len(resolved_stops) * 4)

    routes = get_routes(o_lat, o_lon, d_lat, d_lon, city_cfg["bypasses"], resolved_stops)
    primary = routes[0]
    alt = routes[1] if len(routes) > 1 else None
    total_eta = primary["duration_min"] + est_delay

    if save_history:
        now_dt = datetime.now()
        stop_suffix = f" (+{len(resolved_stops)} stops)" if resolved_stops else ""
        conn = get_db()
        conn.execute(
            """
            INSERT INTO search_history
            (user_id, city_name, origin_name, dest_name, travel_date, travel_hour, weather_label, traffic_status, parking_occ, primary_eta, searched_at, iso_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                active_city,
                origin_name,
                dest_name + stop_suffix,
                date_str,
                hour,
                WEATHER_LABELS.get(weather_code, "Clear Sky"),
                STATUS_LABELS[road_level],
                parking_occ,
                total_eta,
                now_dt.strftime("%d %b %Y, %I:%M %p"),
                now_dt.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    hourly_parking = []
    hourly_road = []
    for h in range(24):
        r_lvl, p_occ = run_ml_prediction(city_id, zone_id, h, day_of_week, is_weekend, weather_code)
        hourly_road.append(round((r_lvl + 1) * 33.3, 1))
        hourly_parking.append(p_occ)

    city_network = get_city_road_network(active_city, city_cfg, hour, day_of_week, is_weekend, weather_code)

    return jsonify({
        "city": active_city,
        "origin": {"name": origin_name, "lat": o_lat, "lon": o_lon},
        "stops": resolved_stops,
        "destination": {"name": dest_name, "lat": d_lat, "lon": d_lon},
        "road_level": road_level,
        "parking_occ": parking_occ,
        "available_pct": round(100 - parking_occ, 1),
        "est_delay": est_delay,
        "primary_route": primary,
        "alt_route": alt,
        "city_network": city_network,
        "hourly_parking": hourly_parking,
        "hourly_road": hourly_road,
    })


@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM search_history WHERE user_id = ? ORDER BY id DESC LIMIT 100",
        (session["user_id"],),
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history/clear", methods=["POST"])
@login_required
def api_clear_history():
    conn = get_db()
    conn.execute("DELETE FROM search_history WHERE user_id = ?", (session["user_id"],))
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)