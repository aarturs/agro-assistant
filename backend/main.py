import os, asyncio, httpx
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv
import ee

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

try:
    import json
    from google.oauth2.credentials import Credentials
    cred_path = "/etc/secrets/earthengine_credentials"
    GEE_PROJECT = os.environ.get("GEE_PROJECT", "")
    if os.path.exists(cred_path):
        with open(cred_path) as f:
            cred_data = json.load(f)
        credentials = Credentials(
            token=None,
            refresh_token=cred_data["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id="517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com",
            client_secret="",
            scopes=cred_data["scopes"]
        )
        ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    else:
        ee.Initialize(project=GEE_PROJECT)
    GEE_AVAILABLE = True
except Exception as e:
    print(f"GEE not available: {e}")
    GEE_AVAILABLE = False


class ChatRequest(BaseModel):
    message: str
    lat: float
    lon: float
    conversation_history: list[dict] = []


import time
_weather_cache = {}

async def fetch_open_meteo(lat, lon):
    cache_key = f"{round(lat,2)}_{round(lon,2)}"
    now = time.time()
    if cache_key in _weather_cache:
        cached_time, cached_data = _weather_cache[cache_key]
        if now - cached_time < 3600:
            return cached_data
    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "weather_code"],
        "hourly": ["temperature_2m", "precipitation", "precipitation_probability", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "Europe/Riga", "forecast_days": 3,
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://api.open-meteo.com/v1/forecast", params=params)
        data = r.json()
        if not data.get("error"):
            _weather_cache[cache_key] = (now, data)
        return data


def fetch_ndvi(lat, lon):
    if not GEE_AVAILABLE:
        return None
    try:
        pt = ee.Geometry.Point([lon, lat])
        today = datetime.utcnow()
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(pt)
            .filterDate((today - timedelta(days=45)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
            .sort("system:time_start", False)
        )
        img = col.first()
        res = img.normalizedDifference(["B8", "B4"]).rename("NDVI").reduceRegion(
            ee.Reducer.mean(), pt.buffer(100), 10
        ).getInfo()
        return round(res.get("NDVI", 0), 3) if res else None
    except Exception as e:
        print(f"NDVI error: {e}")
        return None


async def get_field_data(lat, lon):
    try:
        r = supabase.rpc("find_field_at_point", {"lng": lon, "lat_val": lat}).execute()
        return r.data[0] if r.data else {"block_id": "Nav", "crop_code": "Nezinama", "area_ha": 0}
    except Exception as e:
        print(f"LAD error: {e}")
        return {"block_id": "Nav", "crop_code": "Nezinama", "area_ha": 0}


async def get_vaad(question, crop=""):
    try:
        r = supabase.rpc("match_vaad_products", {"query_text": f"{question} {crop}".strip(), "match_count": 3}).execute()
        return r.data or []
    except Exception as e:
        print(f"VAAD error: {e}")
        return []


@app.post("/chat")
async def chat(req: ChatRequest):
    weather, field = await asyncio.gather(
        fetch_open_meteo(req.lat, req.lon),
        get_field_data(req.lat, req.lon)
    )
    vaad = await get_vaad(req.message, field.get("crop_code", ""))
    ndvi = await asyncio.get_event_loop().run_in_executor(None, fetch_ndvi, req.lat, req.lon)

    cur = weather.get("current", {})
    h = weather.get("hourly", {})
    d = weather.get("daily", {})
    precip24 = round(sum(h.get("precipitation", [0])[:24]), 1)
    ndvi_txt = f"NDVI {ndvi}" if ndvi else "Nav satelitdatu"
    vaad_txt = "\n".join([p.get("content", "") for p in vaad]) if vaad else "Nav preparatu datu."

    system = (
        "Tu esi lauksaimniecibas konsultants Latvija ar 20 gadu pieredzi. Atbildi TIKAI latviski.\n\n"
        f"LAUKS: {field.get('block_id')} | {field.get('crop_code')} | {field.get('area_ha')} ha\n"
        f"LAIKS: {cur.get('temperature_2m')}C, {cur.get('relative_humidity_2m')}% mitrums, vejs {cur.get('wind_speed_10m')} km/h\n"
        f"Nokrisni 24h: {precip24}mm | Augsne: {h.get('soil_temperature_0cm', [0])[0]}C\n"
        f"Prognoze 3d: max {d.get('temperature_2m_max', ['?'])[0]}C, nokrisni {round(sum(d.get('precipitation_sum', [0])), 1)}mm\n"
        f"NDVI: {ndvi_txt}\n"
        f"VAAD: {vaad_txt}\n\n"
        "Sniedz: 1) TULIITEJA RICIBA 2) NEDELAS PLANS 3) BRIDINAJUMI"
    )

    msgs = req.conversation_history + [{"role": "user", "content": req.message}]

    def generate():
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}] + msgs,
                max_tokens=900,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"Kluda: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
async def health():
    try:
        supabase.table("vaad_products").select("id").limit(1).execute()
        sb = True
    except Exception:
        sb = False
    return {"status": "ok", "gee": GEE_AVAILABLE, "supabase": sb}


@app.get("/weather")
async def weather_endpoint(lat: float, lon: float):
    return await fetch_open_meteo(lat, lon)