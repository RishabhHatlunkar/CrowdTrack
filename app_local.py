import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import sqlite3
import threading
import queue
import platform
import pyttsx3
from ultralytics import YOLO

# Custom utils
from utils import detect_people, get_zone_id, draw_zone_grid

# ---------------- CONFIG ----------------
GRID_ROWS, GRID_COLS = 3, 3
FIXED_CONFIDENCE = 0.45
MODEL_PATH = "yolov8n.pt"
DB_NAME = "crowd_guard.db"
FRAME_SKIP = 3  # Run AI every 3rd frame for speed

# ---------------- ASYNC DATABASE & AUDIO WORKER ----------------
db_queue = queue.Queue()

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    # Log table (all zones)
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (timestamp TEXT, zone_0 INTEGER, zone_1 INTEGER, zone_2 INTEGER, 
                  zone_3 INTEGER, zone_4 INTEGER, zone_5 INTEGER, 
                  zone_6 INTEGER, zone_7 INTEGER, zone_8 INTEGER)''')
    # Alert table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts 
                 (timestamp TEXT, zone_id INTEGER, count INTEGER, message TEXT)''')
    conn.commit()
    conn.close()

def db_worker():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    while True:
        try:
            task = db_queue.get()
            if task is None: break
            query, params = task
            c.execute(query, params)
            conn.commit()
            db_queue.task_done()
        except Exception:
            pass
    conn.close()

# Start background thread
init_db()
threading.Thread(target=db_worker, daemon=True).start()

def async_save_alert(zone_id, count, msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    db_queue.put(("INSERT INTO alerts VALUES (?, ?, ?, ?)", (ts, zone_id, count, msg)))

def async_save_log(zone_counts):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    db_queue.put(("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (ts, *zone_counts)))

def get_recent_alerts(limit=10):
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT * FROM alerts ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ---------------- HELPER: ASYNC SPEECH ----------------
def speak(text):
    def _speak():
        if platform.system() == "Windows":
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.say(text)
                engine.runAndWait()
            except: pass
        else:
            print(f"🔊 {text}")
    threading.Thread(target=_speak, daemon=True).start()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- UI LAYOUT ----------------
st.set_page_config(page_title="CrowdGuardAI", layout="wide", page_icon="🛡️")
st.title("🛡️ CrowdGuardAI - Monitor")

# Session State
if "source_mode" not in st.session_state: st.session_state.source_mode = None
if "peak_count" not in st.session_state: st.session_state.peak_count = 0
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "zone_beep_timers" not in st.session_state: st.session_state.zone_beep_timers = {}
if "last_detections" not in st.session_state: st.session_state.last_detections = [] 

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Threshold Settings")

# ✅ SINGLE GLOBAL THRESHOLD (Applied to every zone individually)
global_threshold = st.sidebar.slider(
    "⚠️ Max People per Zone", 
    min_value=1, max_value=50, value=5,
    help="If ANY single zone exceeds this number, an alert is triggered for that zone."
)

st.sidebar.divider()
st.sidebar.subheader("📡 Source")
if st.sidebar.button("▶️ Start Webcam"): st.session_state.source_mode = "webcam"
if st.sidebar.button("⏹️ Stop Webcam"): st.session_state.source_mode = None
if st.sidebar.button("📁 Upload Video"): st.session_state.source_mode = "video"
if st.sidebar.button("💾 Database Logs"): st.session_state.source_mode = "export"

# ---------------- PROCESS FRAME ----------------
def process_frame(frame, frame_count):
    frame_h, frame_w = frame.shape[:2]
    
    # 1. SKIP FRAMES FOR SPEED (Run AI every 3rd frame)
    if frame_count % FRAME_SKIP == 0:
        small_frame = cv2.resize(frame, (640, 480))
        results = detect_people(model, small_frame, conf=FIXED_CONFIDENCE)
        
        # Scale back
        scale_x, scale_y = frame_w / 640, frame_h / 480
        scaled_dets = []
        for x1, y1, x2, y2, conf in results:
            scaled_dets.append((int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), conf))
        st.session_state.last_detections = scaled_dets
    
    detections = st.session_state.last_detections
    
    # 2. Draw Grid & Calculate
    draw_zone_grid(frame, GRID_ROWS, GRID_COLS)
    zone_counts = [0] * (GRID_ROWS * GRID_COLS)
    
    for x1, y1, x2, y2, _ in detections:
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= xc < frame_w and 0 <= yc < frame_h:
            z_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
            zone_counts[z_id] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (xc, yc), 3, (0, 255, 0), -1)

    # 3. CHECK ALERTS (Global Threshold applied per Zone)
    for i, count in enumerate(zone_counts):
        if count >= global_threshold:
            # Visual Alert on Grid
            r, c = i // GRID_COLS, i % GRID_COLS
            zw, zh = frame_w // GRID_COLS, frame_h // GRID_ROWS
            cv2.rectangle(frame, (c*zw, r*zh), ((c+1)*zw, (r+1)*zh), (0, 0, 255), 4)
            cv2.putText(frame, f"ZONE {i} FULL!", (c*zw+10, r*zh+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Audio/DB Alert (Throttled 5s)
            if time.time() - st.session_state.zone_beep_timers.get(i, 0) > 5:
                msg = f"Zone {i} Critical: {count} People"
                speak(f"Alert Zone {i}")
                async_save_alert(i, count, msg)
                st.session_state.zone_beep_timers[i] = time.time()

    # 4. Log Data (Throttled)
    if frame_count % FRAME_SKIP == 0:
        async_save_log(zone_counts)

    return frame, sum(zone_counts)

# ---------------- MAIN LOOP ----------------
if st.session_state.source_mode in ["webcam", "video"]:
    c_vid, c_alert = st.columns([0.75, 0.25])
    with c_vid: 
        FRAME_WINDOW = st.empty()
        METRICS = st.empty()
    with c_alert: 
        st.subheader("🚨 Live Alerts")
        ALERTS = st.empty()

    cap = cv2.VideoCapture(0) if st.session_state.source_mode == "webcam" else None
    if st.session_state.source_mode == "video":
        f = st.file_uploader("Upload Video", type=["mp4"])
        if f: 
            t = tempfile.NamedTemporaryFile(delete=False)
            t.write(f.read())
            cap = cv2.VideoCapture(t.name)

    if cap and cap.isOpened():
        cnt = 0
        while cap.isOpened() and st.session_state.source_mode:
            ret, frame = cap.read()
            if not ret: break
            if st.session_state.source_mode == "webcam": frame = cv2.flip(frame, 1)

            frame, total = process_frame(frame, cnt)
            cnt += 1

            # Update UI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB", use_container_width=True)
            
            if cnt % 5 == 0: # Update text less often
                METRICS.metric("👥 Total Count", total)
                recent = get_recent_alerts()
                if not recent.empty:
                    html = "".join([f"<div style='border-left:4px solid red; padding:5px; margin-bottom:5px; font-size:12px'><b>{r['timestamp'].split()[-1]}</b>: {r['message']}</div>" for _, r in recent.iterrows()])
                    ALERTS.markdown(html, unsafe_allow_html=True)
                else:
                    ALERTS.info("All zones normal.")
            time.sleep(0.001)
        cap.release()

elif st.session_state.source_mode == "export":
    st.markdown("### 💾 Database Logs")
    conn = sqlite3.connect(DB_NAME)
    st.dataframe(pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 500", conn), use_container_width=True)
    conn.close()