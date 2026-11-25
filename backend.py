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
app.secret_key = "smart_mobility_ai_multicity_2026"
DB_PATH = "traffic_app.db"


# ---------------------------------------------------------
# 1. SQLITE DATABASE SETUP & SAFE MIGRATION
# ---------------------------------------------------------
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cursor.execute("PRAGMA table_info(search_history)")
    cols = [row[1] for row in cursor.fetchall()]
    if "iso_timestamp" not in cols:
        cursor.execute("ALTER TABLE search_history ADD COLUMN iso_timestamp TEXT")
    if "city_name" not in cols:
        cursor.execute("ALTER TABLE search_history ADD COLUMN city_name TEXT DEFAULT 'Bengaluru'")

    cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@smarttraffic.in",))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            ("Demo User", "demo@smarttraffic.in", generate_password_hash("demo123")),
        )
    conn.commit()
    conn.close()


init_db()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------------------------------------
# 2. 6 INDIAN CITIES DATA (15 REAL HUBS + BYPASSES EACH)
# ---------------------------------------------------------
CITY_DATA = {
    "Bengaluru": {
        "city_id": 0,
        "center": [12.9716, 77.5946],
        "locations": {
            "Majestic / KSR Railway Station": {"lat": 12.9767, "lon": 77.5713, "zone_id": 1},
            "Whitefield ITPL Main Rd": {"lat": 12.9855, "lon": 77.7361, "zone_id": 3},
            "Silk Board Junction": {"lat": 12.9177, "lon": 77.6238, "zone_id": 0},
            "MG Road / Brigade Road": {"lat": 12.9756, "lon": 77.6066, "zone_id": 1},
            "Koramangala 4th Block": {"lat": 12.9352, "lon": 77.6245, "zone_id": 2},
            "Indiranagar 100ft Rd": {"lat": 12.9784, "lon": 77.6408, "zone_id": 4},
            "Electronic City Phase 1": {"lat": 12.8452, "lon": 77.6602, "zone_id": 0},
            "Hebbal Flyover": {"lat": 13.0354, "lon": 77.5988, "zone_id": 4},
            "Marathahalli Bridge": {"lat": 12.9592, "lon": 77.6974, "zone_id": 3},
            "Bellandur / EcoSpace": {"lat": 12.9260, "lon": 77.6762, "zone_id": 0},
            "HSR Layout Sector 2": {"lat": 12.9121, "lon": 77.6446, "zone_id": 2},
            "Jayanagar 4th Block": {"lat": 12.9299, "lon": 77.5824, "zone_id": 2},
            "Yeshwanthpur Junction": {"lat": 13.0226, "lon": 77.5549, "zone_id": 1},
            "KR Puram Hanging Bridge": {"lat": 13.0012, "lon": 77.6952, "zone_id": 3},
            "Kempegowda International Airport": {"lat": 13.1986, "lon": 77.7066, "zone_id": 4},
        },
        "bypasses": [(13.0285, 77.6380), (12.9121, 77.6446), (12.9592, 77.6974), (12.9300, 77.5824)],
    },
    "Hyderabad": {
        "city_id": 1,
        "center": [17.3850, 78.4867],
        "locations": {
            "Secunderabad Railway Station": {"lat": 17.4344, "lon": 78.5013, "zone_id": 1},
            "HITEC City / Cyber Towers": {"lat": 17.4504, "lon": 78.3808, "zone_id": 3},
            "Gachibowli Financial District": {"lat": 17.4239, "lon": 78.3482, "zone_id": 3},
            "Banjara Hills Road No. 12": {"lat": 17.4126, "lon": 78.4392, "zone_id": 2},
            "Jubilee Hills Check Post": {"lat": 17.4292, "lon": 78.4116, "zone_id": 0},
            "Ameerpet Metro Junction": {"lat": 17.4375, "lon": 78.4483, "zone_id": 0},
            "Panjagutta Crossroads": {"lat": 17.4254, "lon": 78.4505, "zone_id": 0},
            "Madhapur / Inorbit Mall": {"lat": 17.4348, "lon": 78.3866, "zone_id": 2},
            "Kukatpally KPHB Colony": {"lat": 17.4948, "lon": 78.3996, "zone_id": 4},
            "Begumpet Flyover": {"lat": 17.4447, "lon": 78.4664, "zone_id": 1},
            "Charminar / Old City": {"lat": 17.3616, "lon": 78.4747, "zone_id": 2},
            "LB Nagar Junction": {"lat": 17.3457, "lon": 78.5522, "zone_id": 4},
            "Uppal Ring Road": {"lat": 17.3984, "lon": 78.5583, "zone_id": 4},
            "Mehdipatnam Junction": {"lat": 17.3958, "lon": 78.4312, "zone_id": 1},
            "Rajiv Gandhi Intl Airport (Shamshabad)": {"lat": 17.2403, "lon": 78.4294, "zone_id": 4},
        },
        "bypasses": [(17.4126, 78.4392), (17.4650, 78.4200), (17.3958, 78.4312), (17.3750, 78.5100)],
    },
    "Mumbai": {
        "city_id": 2,
        "center": [19.0760, 72.8777],
        "locations": {
            "Chhatrapati Shivaji Maharaj Terminus (CSMT)": {"lat": 18.9398, "lon": 72.8355, "zone_id": 1},
            "Bandra-Kurla Complex (BKC)": {"lat": 19.0674, "lon": 72.8679, "zone_id": 3},
            "Andheri Station / WEH": {"lat": 19.1197, "lon": 72.8464, "zone_id": 0},
            "Lower Parel / Phoenix Palladium": {"lat": 18.9953, "lon": 72.8258, "zone_id": 2},
            "Dadar TT Circle": {"lat": 19.0178, "lon": 72.8478, "zone_id": 0},
            "Powai Hiranandani Gardens": {"lat": 19.1176, "lon": 72.9060, "zone_id": 3},
            "Bandra Linking Road": {"lat": 19.0644, "lon": 72.8362, "zone_id": 2},
            "Worli Sea Face": {"lat": 19.0096, "lon": 72.8156, "zone_id": 1},
            "Nariman Point / Marine Drive": {"lat": 18.9256, "lon": 72.8242, "zone_id": 1},
            "Saki Naka Junction": {"lat": 19.1036, "lon": 72.8874, "zone_id": 0},
            "Goregaon Mindspace / Malad": {"lat": 19.1760, "lon": 72.8358, "zone_id": 3},
            "Thane Ghodbunder Road": {"lat": 19.2183, "lon": 72.9781, "zone_id": 4},
            "Navi Mumbai Vashi Plaza": {"lat": 19.0745, "lon": 72.9978, "zone_id": 4},
            "Chembur Eastern Freeway": {"lat": 19.0522, "lon": 72.9005, "zone_id": 4},
            "Chhatrapati Shivaji Intl Airport (T2)": {"lat": 19.0974, "lon": 72.8745, "zone_id": 4},
        },
        "bypasses": [(19.0522, 72.9005), (19.0096, 72.8156), (19.1176, 72.9060), (19.0674, 72.8679)],
    },
    "Chennai": {
        "city_id": 3,
        "center": [13.0827, 80.2707],
        "locations": {
            "Chennai Central Railway Station": {"lat": 13.0827, "lon": 80.2755, "zone_id": 1},
            "OMR Tidel Park / Thoraipakkam": {"lat": 12.9896, "lon": 80.2496, "zone_id": 3},
            "Guindy Kathipara Junction": {"lat": 13.0067, "lon": 80.2031, "zone_id": 0},
            "T. Nagar / Pondy Bazaar": {"lat": 13.0418, "lon": 80.2341, "zone_id": 2},
            "Anna Nagar Roundtana": {"lat": 13.0850, "lon": 80.2101, "zone_id": 2},
            "Velachery Vijayanagar Junction": {"lat": 12.9755, "lon": 80.2207, "zone_id": 0},
            "Sholinganallur IT Corridor": {"lat": 12.9010, "lon": 80.2279, "zone_id": 3},
            "Adyar Signal / LB Road": {"lat": 13.0012, "lon": 80.2565, "zone_id": 4},
            "Nungambakkam High Road": {"lat": 13.0569, "lon": 80.2425, "zone_id": 1},
            "Mount Road / Anna Salai": {"lat": 13.0604, "lon": 80.2640, "zone_id": 1},
            "Porur Junction": {"lat": 13.0382, "lon": 80.1565, "zone_id": 0},
            "Tambaram Bus Terminus": {"lat": 12.9249, "lon": 80.1000, "zone_id": 4},
            "Koyambedu CMBT": {"lat": 13.0694, "lon": 80.2054, "zone_id": 4},
            "DLF Cybercity Manapakkam": {"lat": 13.0213, "lon": 80.1753, "zone_id": 3},
            "Chennai International Airport (Meenambakkam)": {"lat": 12.9941, "lon": 80.1709, "zone_id": 4},
        },
        "bypasses": [(13.0067, 80.2031), (13.0569, 80.2425), (13.0694, 80.2054), (13.0012, 80.2565)],
    },
    "Delhi": {
        "city_id": 4,
        "center": [28.6139, 77.2090],
        "locations": {
            "New Delhi Railway Station (NDLS)": {"lat": 28.6429, "lon": 77.2191, "zone_id": 1},
            "Connaught Place (Rajiv Chowk)": {"lat": 28.6315, "lon": 77.2167, "zone_id": 1},
            "Cyber Hub / DLF Phase 2 (Gurugram)": {"lat": 28.4950, "lon": 77.0895, "zone_id": 3},
            "Noida Sector 18 / Atta Market": {"lat": 28.5708, "lon": 77.3261, "zone_id": 2},
            "Noida Sector 62 Electronic City": {"lat": 28.6271, "lon": 77.3649, "zone_id": 3},
            "Dhaula Kuan Junction": {"lat": 28.5921, "lon": 77.1615, "zone_id": 0},
            "ITO Crossing": {"lat": 28.6289, "lon": 77.2414, "zone_id": 0},
            "Saket Select Citywalk": {"lat": 28.5286, "lon": 77.2193, "zone_id": 2},
            "Karol Bagh Market": {"lat": 28.6514, "lon": 77.1907, "zone_id": 2},
            "Lajpat Nagar Central Market": {"lat": 28.5677, "lon": 77.2433, "zone_id": 2},
            "Ashram Chowk": {"lat": 28.5700, "lon": 77.2591, "zone_id": 0},
            "Nehru Place IT Hub": {"lat": 28.5494, "lon": 77.2517, "zone_id": 3},
            "Rohini Sector 10 / Rithala": {"lat": 28.7161, "lon": 77.1147, "zone_id": 4},
            "Dwarka Sector 21": {"lat": 28.5522, "lon": 77.0583, "zone_id": 4},
            "Indira Gandhi Intl Airport (T3)": {"lat": 28.5562, "lon": 77.1000, "zone_id": 4},
        },
        "bypasses": [(28.5921, 77.1615), (28.6289, 77.2414), (28.5494, 77.2517), (28.6514, 77.1907)],
    },
    "Kolkata": {
        "city_id": 5,
        "center": [22.5726, 88.3639],
        "locations": {
            "Howrah Junction Railway Station": {"lat": 22.5839, "lon": 88.3428, "zone_id": 0},
            "Salt Lake Sector V (IT Hub)": {"lat": 22.5735, "lon": 88.4331, "zone_id": 3},
            "New Town / Eco Park": {"lat": 22.5965, "lon": 88.4748, "zone_id": 3},
            "Park Street / Esplanade": {"lat": 22.5551, "lon": 88.3515, "zone_id": 1},
            "Science City / EM Bypass": {"lat": 22.5393, "lon": 88.3960, "zone_id": 0},
            "Gariahat Junction": {"lat": 22.5176, "lon": 88.3665, "zone_id": 2},
            "Ultadanga / Hudco Crossing": {"lat": 22.5937, "lon": 88.3879, "zone_id": 0},
            "BBD Bagh / Dalhousie": {"lat": 22.5714, "lon": 88.3475, "zone_id": 1},
            "South City Mall / Jadavpur": {"lat": 22.5015, "lon": 88.3614, "zone_id": 2},
            "Sealdah Station": {"lat": 22.5663, "lon": 88.3702, "zone_id": 1},
            "Ruby General Hospital Crossing": {"lat": 22.5144, "lon": 88.4017, "zone_id": 4},
            "Behala Chowrasta": {"lat": 22.4875, "lon": 88.3106, "zone_id": 4},
            "Dakshineswar / Dunlop": {"lat": 22.6548, "lon": 88.3575, "zone_id": 4},
            "New Market / Lindsay Street": {"lat": 22.5601, "lon": 88.3526, "zone_id": 2},
            "Netaji Subhas Chandra Bose Intl Airport": {"lat": 22.6520, "lon": 88.4463, "zone_id": 4},
        },
        "bypasses": [(22.5393, 88.3960), (22.5937, 88.3879), (22.5551, 88.3515), (22.5144, 88.4017)],
    },
}

WEATHER_LABELS = {0: "Clear Sky", 1: "Cloudy / Humid", 2: "Heavy Monsoon Rain"}
STATUS_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

road_path = "models/road_congestion_model.pkl"
parking_path = "models/parking_model.pkl"
if os.path.exists(road_path) and os.path.exists(parking_path):
    road_model = joblib.load(road_path)
    parking_model = joblib.load(parking_path)
    DEMO_MODE = False
else:
    road_model, parking_model = None, None
    DEMO_MODE = True


def geocode_place(place_query: str, city_name: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{place_query}, {city_name}, India", "format": "json", "limit": 1}
        headers = {"User-Agent": "SmartCongestionStudentProject/2.0"}
        res = requests.get(url, params=params, headers=headers, timeout=5).json()
        if res:
            return float(res[0]["lat"]), float(res[0]["lon"])
    except Exception:
        pass
    return CITY_DATA[city_name]["center"]


def run_ml_prediction(city_id: int, zone_id: int, hour: int, day_of_week: int, is_weekend: int, weather_code: int):
    features = pd.DataFrame([{
        "city_id": city_id,
        "zone_id": zone_id,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "weather_code": weather_code,
    }])
    if not DEMO_MODE:
        try:
            road_level = int(road_model.predict(features)[0])
            parking_occ = float(np.clip(parking_model.predict(features)[0], 0, 100))
            return road_level, round(parking_occ, 1)
        except Exception:
            pass

    rush = 1 if (8 <= hour <= 11 or 17 <= hour <= 21) else 0
    road_level = 2 if (rush and weather_code >= 1) else (1 if rush else 0)
    parking_occ = float(np.clip(35 + (rush * 42) + (zone_id * 4) + (weather_code * 6), 5, 99))
    return road_level, round(parking_occ, 1)


def get_routes(start_lat, start_lon, end_lat, end_lon, bypass_waypoints):
    parsed_routes = []
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives=3"
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
        for wp_lat, wp_lon in bypass_waypoints:
            wp_url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{start_lon},{start_lat};{wp_lon},{wp_lat};{end_lon},{end_lat}"
                f"?overview=full&geometries=geojson"
            )
            try:
                wp_res = requests.get(wp_url, timeout=5).json()
                if wp_res.get("code") == "Ok" and wp_res["routes"]:
                    r = wp_res["routes"][0]
                    dist_km = round(r["distance"] / 1000.0, 1)
                    if abs(dist_km - primary_dist) >= 1.2 and dist_km <= primary_dist * 1.45:
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
        mid_lat = (start_lat + end_lat) / 2 + 0.012
        mid_lon = (start_lon + end_lon) / 2 - 0.012
        parsed_routes = [
            {"coords": [[start_lat, start_lon], [end_lat, end_lon]], "distance_km": 12.0, "duration_min": 28},
            {"coords": [[start_lat, start_lon], [mid_lat, mid_lon], [end_lat, end_lon]], "distance_km": 14.1, "duration_min": 32},
        ]
    return parsed_routes


# ---------------------------------------------------------
# 3. AUTHENTICATION & CITY SELECTION ROUTES
# ---------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if "user_id" in session:
        return redirect(url_for("index"))

    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action")
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        selected_city = request.form.get("city", "Bengaluru")
        if selected_city not in CITY_DATA:
            selected_city = "Bengaluru"

        conn = get_db()
        cursor = conn.cursor()

        if action == "register":
            full_name = request.form.get("full_name", "").strip()
            if not full_name or not email or not password:
                error = "All fields are required to create an account."
            else:
                try:
                    cursor.execute(
                        "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                        (full_name, email, generate_password_hash(password)),
                    )
                    conn.commit()
                    success = "Account created successfully! Please sign in."
                except sqlite3.IntegrityError:
                    error = "An account with this email already exists."
        else:
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user["password_hash"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["full_name"]
                session["user_email"] = user["email"]
                session["city"] = selected_city
                conn.close()
                return redirect(url_for("index"))
            else:
                error = "Invalid email or password."

        conn.close()

    return render_template("login.html", cities=list(CITY_DATA.keys()), error=error, success=success)


@app.route("/set-city/<city_name>")
@login_required
def set_city(city_name):
    if city_name in CITY_DATA:
        session["city"] = city_name
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ---------------------------------------------------------
# 4. MAIN DASHBOARD & SEARCH HISTORY PAGE
# ---------------------------------------------------------
@app.route("/")
@login_required
def index():
    active_city = session.get("city", "Bengaluru")
    if active_city not in CITY_DATA:
        active_city = "Bengaluru"
        session["city"] = active_city

    city_locations = list(CITY_DATA[active_city]["locations"].keys())

    return render_template(
        "index.html",
        active_city=active_city,
        cities=list(CITY_DATA.keys()),
        origins=city_locations,
        destinations=city_locations,
        city_center=CITY_DATA[active_city]["center"],
        demo_mode=DEMO_MODE,
        user_name=session.get("user_name", "User"),
        user_email=session.get("user_email", ""),
    )


@app.route("/history")
@login_required
def history_page():
    active_city = session.get("city", "Bengaluru")
    return render_template(
        "history.html",
        active_city=active_city,
        cities=list(CITY_DATA.keys()),
        user_name=session.get("user_name", "User"),
        user_email=session.get("user_email", ""),
    )


# ---------------------------------------------------------
# 5. API ENDPOINTS (PREDICT & HISTORY)
# ---------------------------------------------------------
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
    est_delay = [5, 18, 42][road_level] + (weather_code * 6)

    routes = get_routes(o_lat, o_lon, d_lat, d_lon, city_cfg["bypasses"])
    primary = routes[0]
    alt = routes[1] if len(routes) > 1 else None
    total_eta = primary["duration_min"] + est_delay

    if save_history:
        now_dt = datetime.now()
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
                dest_name,
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

    return jsonify({
        "city": active_city,
        "origin": {"name": origin_name, "lat": o_lat, "lon": o_lon},
        "destination": {"name": dest_name, "lat": d_lat, "lon": d_lon},
        "road_level": road_level,
        "parking_occ": parking_occ,
        "available_pct": round(100 - parking_occ, 1),
        "est_delay": est_delay,
        "primary_route": primary,
        "alt_route": alt,
        "hourly_parking": hourly_parking,
        "hourly_road": hourly_road,
    })


@app.route("/api/history", methods=["GET"])
@login_required
def get_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM search_history WHERE user_id = ? ORDER BY id DESC",
        (session["user_id"],),
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history/clear", methods=["POST"])
@login_required
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM search_history WHERE user_id = ?", (session["user_id"],))
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)