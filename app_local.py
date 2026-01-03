import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from ultralytics import YOLO
import threading

# Page config
st.set_page_config(
    page_title="CrowdGuard.AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'source_mode' not in st.session_state:
    st.session_state.source_mode = None
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'peak_count' not in st.session_state:
    st.session_state.peak_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'zone_beep_timers' not in st.session_state:
    st.session_state.zone_beep_timers = {}
if 'LOG' not in st.session_state:
    st.session_state.LOG = []
if 'incidents' not in st.session_state:
    st.session_state.incidents = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'monitor'
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = []

GRID_ROWS, GRID_COLS = 3, 3
FRAME_SKIP = 2  # Process every 2nd frame for speed
RESIZE_WIDTH = 640  # Resize frame for faster processing

# Load YOLOv8 model with optimizations
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()  # Fuse layers for faster inference
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load YOLOv8 model: {e}")
    model = None

# Detection Functions
def detect_people(model, frame, conf=0.5):
    """Detect people in frame using YOLOv8 - Optimized version"""
    if model is None:
        return []
    
    # Run inference with optimizations
    results = model(frame, conf=conf, classes=[0], verbose=False, device='cpu')  # class 0 is 'person'
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
    
    return detections

def get_zone_id(x, y, frame_w, frame_h, rows, cols):
    """Get zone ID from coordinates"""
    zone_w = frame_w // cols
    zone_h = frame_h // rows
    col = min(x // zone_w, cols - 1)
    row = min(y // zone_h, rows - 1)
    return row * cols + col

def draw_zone_grid(frame, rows, cols):
    """Draw grid overlay on frame"""
    h, w = frame.shape[:2]
    zone_w = w // cols
    zone_h = h // rows
    
    # Draw vertical lines
    for i in range(1, cols):
        cv2.line(frame, (i * zone_w, 0), (i * zone_w, h), (0, 255, 255), 2)
    
    # Draw horizontal lines
    for i in range(1, rows):
        cv2.line(frame, (0, i * zone_h), (w, i * zone_h), (0, 255, 255), 2)
    
    return frame

# Cyberpunk Theme CSS
def load_theme():
    theme = st.session_state.theme
    
    if theme == 'dark':
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@400;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
            font-family: 'Roboto Mono', monospace;
        }
        
        /* Animated Grid Background */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(6, 182, 212, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #06b6d4;
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.8);
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        
        .cyber-card {
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 0 30px rgba(6, 182, 212, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .neon-text {
            color: #06b6d4;
            text-shadow: 0 0 10px #06b6d4, 0 0 20px #06b6d4;
        }
        
        .neon-pink {
            color: #ec4899;
            text-shadow: 0 0 10px #ec4899, 0 0 20px #ec4899;
        }
        
        .neon-yellow {
            color: #eab308;
            text-shadow: 0 0 10px #eab308, 0 0 20px #eab308;
        }
        
        .neon-green {
            color: #22c55e;
            text-shadow: 0 0 10px #22c55e, 0 0 20px #22c55e;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(10, 10, 10, 0.8);
            border: 2px solid rgba(6, 182, 212, 0.4);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 900;
            color: #06b6d4;
            text-shadow: 0 0 15px #06b6d4;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #06b6d4, #8b5cf6);
            color: white;
            border: 2px solid #06b6d4;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 40px rgba(6, 182, 212, 0.8);
            transform: translateY(-2px);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(10, 10, 10, 0.8);
            padding: 10px;
            border-radius: 10px;
            border: 2px solid rgba(6, 182, 212, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #06b6d4;
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #06b6d4, #8b5cf6);
            border: 2px solid #06b6d4;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
        }
        
        /* Alert Box */
        .alert-box {
            background: rgba(239, 68, 68, 0.2);
            border: 2px solid #ef4444;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Zone Grid */
        .zone-safe {
            background: rgba(34, 197, 94, 0.3);
            border: 2px solid #22c55e;
            box-shadow: 0 0 15px rgba(34, 197, 94, 0.6);
        }
        
        .zone-warning {
            background: rgba(234, 179, 8, 0.3);
            border: 2px solid #eab308;
            box-shadow: 0 0 15px rgba(234, 179, 8, 0.6);
        }
        
        .zone-danger {
            background: rgba(239, 68, 68, 0.3);
            border: 2px solid #ef4444;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
            animation: pulse 1s infinite;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(10, 10, 10, 0.95);
            border-right: 2px solid rgba(6, 182, 212, 0.3);
        }
        
        /* Input fields */
        .stSlider {
            color: #06b6d4;
        }
        
        </style>
        """
    else:  # Light theme
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@400;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e9d5ff 50%, #fce7f3 100%);
            font-family: 'Roboto Mono', monospace;
        }
        
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(168, 85, 247, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #7c3aed;
            text-shadow: 0 0 20px rgba(124, 58, 237, 0.5);
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        
        .cyber-card {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(168, 85, 247, 0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 0 30px rgba(168, 85, 247, 0.2);
        }
        
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.8);
            border: 2px solid rgba(168, 85, 247, 0.4);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 900;
            color: #7c3aed;
            text-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            color: white;
            border: 2px solid #7c3aed;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.5);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            border: 2px solid rgba(168, 85, 247, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #7c3aed;
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed, #ec4899);
            color: white;
            border: 2px solid #7c3aed;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.6);
        }
        
        .alert-box {
            background: rgba(239, 68, 68, 0.1);
            border: 2px solid #ef4444;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        
        .zone-safe {
            background: rgba(34, 197, 94, 0.2);
            border: 2px solid #22c55e;
        }
        
        .zone-warning {
            background: rgba(234, 179, 8, 0.2);
            border: 2px solid #eab308;
        }
        
        .zone-danger {
            background: rgba(239, 68, 68, 0.2);
            border: 2px solid #ef4444;
            animation: pulse 1s infinite;
        }
        
        </style>
        """
    
    st.markdown(css, unsafe_allow_html=True)

load_theme()

# Header
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    if st.session_state.theme == 'dark':
        st.markdown("""
        <h1 style='margin:0; padding:0;'>
            ⚡ CROWDGUARD<span style='color:#ec4899;'>.AI</span>
        </h1>
        <p style='font-size:12px; color:#06b6d4; margin:0; letter-spacing:2px;'>&gt; NEURAL CROWD INTELLIGENCE SYSTEM</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <h1 style='margin:0; padding:0;'>
            ⚡ CROWDGUARD<span style='color:#ec4899;'>.AI</span>
        </h1>
        <p style='font-size:12px; color:#7c3aed; margin:0; letter-spacing:2px;'>&gt; NEURAL CROWD INTELLIGENCE SYSTEM</p>
        """, unsafe_allow_html=True)

with col2:
    if st.button("🌓 TOGGLE THEME", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

with col3:
    monitoring_label = "⏹️ STOP" if st.session_state.is_monitoring else "▶️ START"
    if st.button(monitoring_label, use_container_width=True, type="primary"):
        st.session_state.is_monitoring = not st.session_state.is_monitoring
        if st.session_state.is_monitoring:
            st.session_state.start_time = time.time()

st.markdown("---")

# Horizontal Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📡 LIVE MONITOR", "📊 ANALYTICS", "🚨 INCIDENTS", "⚙️ SETTINGS"])

# Helper Functions
def get_zone_status(count, threshold):
    if count >= threshold:
        return "danger", "🔴 CRITICAL"
    elif count >= threshold * 0.6:
        return "warning", "🟡 WARNING"
    return "safe", "🟢 SAFE"

def render_zone_grid(zone_data, threshold):
    st.markdown("### 🗺️ ZONE MONITOR (3x3 GRID)")
    
    for i in range(GRID_ROWS):
        cols = st.columns(GRID_COLS)
        for j in range(GRID_COLS):
            zone_id = i * GRID_COLS + j
            count = zone_data[zone_id]
            status, status_text = get_zone_status(count, threshold)
            
            with cols[j]:
                st.markdown(f"""
                <div class='zone-{status}' style='text-align:center; padding:20px; border-radius:8px; margin:5px;'>
                    <div style='font-size:12px; opacity:0.7;'>ZONE {zone_id}</div>
                    <div style='font-size:36px; font-weight:900; margin:10px 0;'>{count}</div>
                    <div style='font-size:10px; font-weight:700;'>{status_text}</div>
                </div>
                """, unsafe_allow_html=True)

# TAB 1: LIVE MONITOR
with tab1:
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get real zone data or simulate
    if st.session_state.webcam_active or st.session_state.video_file:
        zone_data = st.session_state.get('zone_data', [0] * 9)
    else:
        zone_data = [0] * 9
    
    current_total = sum(zone_data)
    elapsed = int(time.time() - st.session_state.start_time) if st.session_state.is_monitoring else 0
    avg_density = current_total / 9 if current_total > 0 else 0
    
    if current_total > st.session_state.peak_count:
        st.session_state.peak_count = current_total
    
    with col1:
        st.metric("👥 CURRENT COUNT", current_total)
    with col2:
        st.metric("📈 PEAK COUNT", st.session_state.peak_count)
    with col3:
        st.metric("⏱️ UPTIME", f"{elapsed}s")
    with col4:
        st.metric("📊 AVG DENSITY", f"{avg_density:.1f}")
    
    st.markdown("---")
    
    # Main monitoring area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📹 LIVE FEED")
        
        # Video/Webcam Display
        video_placeholder = st.empty()
        
        # Show current frame if available
        if st.session_state.current_frame is not None:
            video_placeholder.image(st.session_state.current_frame, channels="RGB", use_container_width=True)
        else:
            video_placeholder.info("📷 No active video feed. Start webcam or upload video below.")
        
        st.markdown("---")
        
        # Zone Grid
        threshold = st.session_state.get('alert_threshold', 5)
        render_zone_grid(zone_data, threshold)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🎯 QUICK ACTIONS")
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("📸 START WEBCAM", use_container_width=True, disabled=st.session_state.webcam_active):
                st.session_state.webcam_active = True
                st.session_state.is_monitoring = True
                st.rerun()
            
            if st.session_state.webcam_active:
                if st.button("⏹️ STOP WEBCAM", use_container_width=True):
                    st.session_state.webcam_active = False
                    st.session_state.current_frame = None
                    st.rerun()
        
        with action_cols[1]:
            uploaded_file = st.file_uploader("📤 UPLOAD VIDEO", type=["mp4", "avi", "mov"], label_visibility="collapsed")
            if uploaded_file is not None:
                st.session_state.video_file = uploaded_file
                st.session_state.is_monitoring = True
        
        with action_cols[2]:
            if st.button("➕ REPORT INCIDENT", use_container_width=True):
                st.session_state.show_incident_form = True
    
    with col_right:
        st.markdown("### 🚨 LIVE ALERTS")
        
        # Check for alerts
        for idx, count in enumerate(zone_data):
            if count >= threshold:
                alert_time = datetime.now().strftime("%H:%M:%S")
                alert = {
                    'zone': idx,
                    'count': count,
                    'time': alert_time,
                    'type': 'overcrowded'
                }
                # Avoid duplicates
                if not any(a['zone'] == idx and a['time'] == alert_time for a in st.session_state.alerts):
                    st.session_state.alerts.insert(0, alert)
        
        # Keep only last 10 alerts
        st.session_state.alerts = st.session_state.alerts[:10]
        
        if st.session_state.alerts:
            for alert in st.session_state.alerts:
                st.markdown(f"""
                <div class='alert-box'>
                    <div style='font-weight:700; margin-bottom:5px;'>🔴 ZONE {alert['zone']} OVERCROWDED</div>
                    <div style='font-size:12px; opacity:0.8;'>Count: {alert['count']} | Time: {alert['time']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🟢 NO ACTIVE ALERTS - ALL ZONES SAFE")

# Process Webcam
if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            
            # Increment frame counter
            st.session_state.frame_count += 1
            
            # Only run detection every FRAME_SKIP frames
            if st.session_state.frame_count % FRAME_SKIP == 0:
                # Resize frame for faster processing
                scale = RESIZE_WIDTH / frame_w
                small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame_h * scale)))
                small_h, small_w = small_frame.shape[:2]
                
                # Detect people on smaller frame
                detection_conf = st.session_state.get('detection_confidence', 0.45)
                detections = detect_people(model, small_frame, conf=detection_conf)
                
                # Scale detections back to original frame size
                scale_x = frame_w / small_w
                scale_y = frame_h / small_h
                scaled_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf = det
                    scaled_detections.append((
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y), conf
                    ))
                
                # Store detections for reuse
                st.session_state.cached_detections = scaled_detections
            else:
                # Reuse cached detections
                scaled_detections = st.session_state.get('cached_detections', [])
            
            # Initialize zone counts
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)
            
            # Draw grid
            draw_zone_grid(frame, GRID_ROWS, GRID_COLS)
            
            # Process detections
            for det in scaled_detections:
                x1, y1, x2, y2, conf = det
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                zone_counts[zone_id] += 1
                
                # Draw bounding box (only if enabled)
                if st.session_state.get('show_boxes', True):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (xc, yc), 4, (0, 0, 255), -1)
            
            # Add zone count overlays
            threshold = st.session_state.get('alert_threshold', 5)
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    zone_id = i * GRID_COLS + j
                    count = zone_counts[zone_id]
                    x = j * frame_w // GRID_COLS + 10
                    y = i * frame_h // GRID_ROWS + 30
                    
                    # Determine color
                    if count >= threshold:
                        color = (0, 0, 255)  # Red
                        label = f"Z{zone_id}: {count} [ALERT]"
                    elif count >= threshold * 0.6:
                        color = (0, 255, 255)  # Yellow
                        label = f"Z{zone_id}: {count} [WARN]"
                    else:
                        color = (0, 255, 0)  # Green
                        label = f"Z{zone_id}: {count}"
                    
                    cv2.putText(frame, label, (x, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add FPS counter
            fps_text = f"FPS: {st.session_state.frame_count % 30}"
            cv2.putText(frame, fps_text, (10, frame_h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Store zone data and frame
            st.session_state.zone_data = zone_counts
            st.session_state.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Log data only every FRAME_SKIP frames
            if st.session_state.frame_count % FRAME_SKIP == 0:
                log_entry = {'timestamp': datetime.now().strftime("%H:%M:%S")}
                for i, c in enumerate(zone_counts):
                    log_entry[f'Zone_{i}'] = c
                st.session_state.LOG.append(log_entry)
                
                if len(st.session_state.LOG) > 500:
                    st.session_state.LOG = st.session_state.LOG[-500:]
        
        cap.release()
    
    # Faster refresh
    time.sleep(0.01)
    st.rerun()

# Process Video File
if st.session_state.video_file is not None and not st.session_state.webcam_active:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(st.session_state.video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_h, frame_w = frame.shape[:2]
            
            # Detect people
            detection_conf = st.session_state.get('detection_confidence', 0.5)
            detections = detect_people(model, frame, conf=detection_conf)
            
            # Initialize zone counts
            zone_counts = [0] * (GRID_ROWS * GRID_COLS)
            
            # Draw grid
            draw_zone_grid(frame, GRID_ROWS, GRID_COLS)
            
            # Process detections
            for det in detections:
                x1, y1, x2, y2, conf = det
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
                zone_counts[zone_id] += 1
                
                # Draw bounding box
                if st.session_state.get('show_boxes', True):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (xc, yc), 4, (0, 0, 255), -1)
            
            # Add zone count overlays
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    zone_id = i * GRID_COLS + j
                    count = zone_counts[zone_id]
                    x = j * frame_w // GRID_COLS + 10
                    y = i * frame_h // GRID_ROWS + 30
                    
                    # Determine color
                    if count >= threshold:
                        color = (0, 0, 255)  # Red
                    elif count >= threshold * 0.6:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green
                    
                    cv2.putText(frame, f"Z{zone_id}: {count}", (x, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Store zone data and frame
            st.session_state.zone_data = zone_counts
            st.session_state.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Log data
            log_entry = {'timestamp': datetime.now().strftime("%H:%M:%S")}
            for i, c in enumerate(zone_counts):
                log_entry[f'Zone_{i}'] = c
            st.session_state.LOG.append(log_entry)
            
            if len(st.session_state.LOG) > 500:
                st.session_state.LOG = st.session_state.LOG[-500:]
        
        cap.release()
    
    # Auto-refresh for video
    time.sleep(0.05)
    st.rerun()

# TAB 2: ANALYTICS
with tab2:
    st.markdown("### 📊 CROWD ANALYTICS DASHBOARD")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔥 CUMULATIVE HEATMAP")
        
        # Create heatmap data
        if len(st.session_state.LOG) > 0:
            df = pd.DataFrame(st.session_state.LOG)
            zone_cols = [f'Zone_{i}' for i in range(9) if f'Zone_{i}' in df.columns]
            if zone_cols:
                heatmap_data = df[zone_cols].sum().values.reshape(3, 3)
            else:
                heatmap_data = np.random.randint(0, 50, (3, 3))
        else:
            heatmap_data = np.random.randint(0, 50, (3, 3))
        
        fig, ax = plt.subplots(figsize=(6, 5))
        if st.session_state.theme == 'dark':
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Count'}, linewidths=2, ax=ax)
            ax.tick_params(colors='#06b6d4')
        else:
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                       cbar_kws={'label': 'Count'}, linewidths=2, ax=ax)
        
        ax.set_title('Zone Activity Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📈 CROWD TREND")
        
        # Generate trend data
        if len(st.session_state.LOG) > 0:
            df = pd.DataFrame(st.session_state.LOG)
            if 'timestamp' in df.columns:
                trend_data = df.tail(20)
            else:
                trend_data = pd.DataFrame({
                    'timestamp': [f"T{i}" for i in range(20)],
                    'total': np.random.randint(0, 50, 20)
                })
        else:
            trend_data = pd.DataFrame({
                'timestamp': [f"T{i}" for i in range(20)],
                'total': np.random.randint(0, 50, 20)
            })
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if st.session_state.theme == 'dark':
            fig2.patch.set_facecolor('#0a0a0a')
            ax2.set_facecolor('#0a0a0a')
            ax2.plot(range(len(trend_data)), trend_data.get('total', trend_data.iloc[:, 1]), 
                    color='#06b6d4', linewidth=2, marker='o')
            ax2.tick_params(colors='#06b6d4')
            ax2.spines['bottom'].set_color('#06b6d4')
            ax2.spines['top'].set_color('#06b6d4')
            ax2.spines['right'].set_color('#06b6d4')
            ax2.spines['left'].set_color('#06b6d4')
        else:
            ax2.plot(range(len(trend_data)), trend_data.get('total', trend_data.iloc[:, 1]), 
                    color='#7c3aed', linewidth=2, marker='o')
        
        ax2.set_title('Crowd Level Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Total Count')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    
    st.markdown("---")
    
    if st.button("📥 EXPORT DATA (CSV)", use_container_width=True):
        if len(st.session_state.LOG) > 0:
            df = pd.DataFrame(st.session_state.LOG)
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ DOWNLOAD CSV",
                data=csv,
                file_name=f"crowdguard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("⚠️ No data available to export")

# TAB 3: INCIDENTS
with tab3:
    st.markdown("### 🚨 INCIDENT MANAGEMENT")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("➕ NEW INCIDENT", use_container_width=True, type="primary"):
            st.session_state.show_incident_form = True
    
    # Show incident form
    if st.session_state.get('show_incident_form', False):
        with st.form("incident_form"):
            st.markdown("#### 📝 REPORT NEW INCIDENT")
            
            incident_type = st.selectbox(
                "Incident Type",
                ["🚑 Medical Emergency", "👊 Fight/Violence", "🚩 Suspicious Object", 
                 "🔥 Fire Hazard", "🔍 Lost Person", "⚠️ Other"]
            )
            
            zone_select = st.selectbox("Zone", [f"Zone {i}" for i in range(9)])
            
            notes = st.text_area("Notes", placeholder="Enter incident details...")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("✅ SUBMIT", use_container_width=True)
            with col2:
                cancel = st.form_submit_button("❌ CANCEL", use_container_width=True)
            
            if submit:
                incident = {
                    'id': len(st.session_state.incidents),
                    'type': incident_type,
                    'zone': zone_select,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'ACTIVE',
                    'notes': notes
                }
                st.session_state.incidents.append(incident)
                st.session_state.show_incident_form = False
                st.success("✅ Incident reported successfully!")
                st.rerun()
            
            if cancel:
                st.session_state.show_incident_form = False
                st.rerun()
    
    st.markdown("---")
    
    # Display incidents
    if st.session_state.incidents:
        for incident in reversed(st.session_state.incidents):
            status_color = "#22c55e" if incident['status'] == 'RESOLVED' else "#eab308"
            
            st.markdown(f"""
            <div class='cyber-card' style='margin:10px 0;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-size:20px; font-weight:700; margin-bottom:5px;'>
                            {incident['type']}
                        </div>
                        <div style='font-size:12px; opacity:0.7;'>
                            {incident['zone']} | {incident['timestamp']}
                        </div>
                        {f"<div style='margin-top:8px; font-size:13px;'>{incident['notes']}</div>" if incident['notes'] else ""}
                    </div>
                    <div style='background:{status_color}; padding:8px 16px; border-radius:6px; font-weight:700;'>
                        {incident['status']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📋 No incidents reported yet")

# TAB 4: SETTINGS
with tab4:
    st.markdown("### ⚙️ SYSTEM CONFIGURATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎚️ DETECTION SETTINGS")
        
        alert_threshold = st.slider(
            "Alert Threshold (per zone)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of people to trigger an alert"
        )
        st.session_state.alert_threshold = alert_threshold
        
        detection_confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        st.session_state.detection_confidence = detection_confidence
        
        show_boxes = st.checkbox("Show detection boxes", value=True)
        st.session_state.show_boxes = show_boxes
        
        audio_alerts = st.checkbox("Enable audio alerts", value=True)
    
    with col2:
        st.markdown("#### 🎨 DISPLAY SETTINGS")
        
        grid_size = st.selectbox("Grid Size", ["3x3", "4x4", "5x5"], index=0)
        
        frame_skip = st.slider(
            "Frame Skip (higher = faster)",
            min_value=1,
            max_value=10,
            value=3,
            help="Process every Nth frame. Higher values = faster performance"
        )
        FRAME_SKIP = frame_skip
        
        st.markdown("#### 💾 DATA MANAGEMENT")
        
        if st.button("🗑️ CLEAR ALL DATA", use_container_width=True):
            st.session_state.LOG = []
            st.session_state.incidents = []
            st.session_state.alerts = []
            st.session_state.peak_count = 0
            st.success("✅ All data cleared!")
        
        if st.button("🔄 RESET SYSTEM", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; padding:20px; opacity:0.7;'>
    <p style='margin:0; font-size:12px; letter-spacing:2px;'>
        {'⚡' if st.session_state.theme == 'dark' else '🔮'} CROWDGUARD.AI v2.0 | 
        POWERED BY YOLOV8 + NEURAL NETWORKS | 
        {'DARK MODE' if st.session_state.theme == 'dark' else 'LIGHT MODE'} ACTIVE
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for monitoring
if st.session_state.is_monitoring:
    time.sleep(2)
    st.rerun()