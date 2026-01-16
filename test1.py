import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from scipy.signal import butter, lfilter
import statistics
import csv
import os

# --- CONFIGURATION ---
COM_PORT = '/dev/cu.usbmodem1101'  # ⚠️ Check this!
BAUD_RATE = 57600
WINDOW_SIZE = 1500        # 7.5 seconds
LOG_FILE = f"ecg_log_{int(time.time())}.csv"

# --- FILTER SETTINGS ---
FS = 200.0                
LOWCUT = 0.5              
HIGHCUT = 25.0            
FILTER_ORDER = 4          

# --- VARIABLES ---
data_buffer = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
filtered_buffer = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
last_beat_time = 0

# METRICS STORAGE
bpm_history = deque([60] * 10, maxlen=10) 
rr_intervals = deque([0.8] * 30, maxlen=30) # Store last 30 gaps for HRV

# --- LOGGING SETUP ---
# Creates a new file instantly
with open(LOG_FILE, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Raw_Signal", "Filtered_Signal", "BPM", "Event"])
print(f"📝 Logging data to: {LOG_FILE}")

# --- FILTER MATH ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, b, a):
    return lfilter(b, a, data)

b, a = butter_bandpass(LOWCUT, HIGHCUT, FS, FILTER_ORDER)

# --- SERIAL CONNECTION ---
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("✅ Connected! Dashboard Active.")
except:
    print("❌ Error: Port not found.")
    exit()

# --- PLOTTING SETUP ---
fig, ax = plt.subplots(figsize=(12, 7))
fig.canvas.manager.set_window_title("Advanced Medical Dashboard")
ax.set_facecolor('black')
fig.patch.set_facecolor('#222222')

# Main Signal Line
line, = ax.plot(range(WINDOW_SIZE), [0]*WINDOW_SIZE, color='#00FFFF', linewidth=2)

# TEXT LABELS
bpm_text = ax.text(0.95, 0.95, "BPM: --", transform=ax.transAxes, color='white', fontsize=20, fontweight='bold', ha='right')
hrv_text = ax.text(0.95, 0.88, "HRV (Stress): --", transform=ax.transAxes, color='yellow', fontsize=12, ha='right')
alert_text = ax.text(0.50, 0.50, "", transform=ax.transAxes, color='red', fontsize=30, fontweight='bold', ha='center', alpha=0.0)

ax.grid(True, color='#444444', linestyle='--')

def update(frame):
    global last_beat_time
    
    # 1. READ & LOG
    while ser.in_waiting > 0:
        try:
            raw_str = ser.readline().decode('utf-8').strip()
            if raw_str.lstrip('-').isdigit():
                val = int(raw_str)
                data_buffer.append(val)
        except:
            pass

    if len(data_buffer) >= WINDOW_SIZE:
        # 2. FILTER
        raw_signal = np.array(data_buffer)
        clean_signal = apply_filter(raw_signal, b, a)
        filtered_buffer.clear()
        filtered_buffer.extend(clean_signal)

        # LOGGING (Append to CSV)
        # We log the latest point. Note: File I/O in loop can be slow, 
        # usually we buffer this, but for <200Hz it's okay.
        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), data_buffer[-1], clean_signal[-1], bpm_history[-1], ""])
        
        # 3. ANALYSIS
        current_val = clean_signal[-1]
        current_time = time.time()
        p95 = np.percentile(clean_signal, 95)
        threshold = p95 * 0.65 
        
        # TIMEOUT
        if (current_time - last_beat_time) > 4.0:
            bpm_text.set_text("BPM: --")
            bpm_text.set_color('grey')
        
        # FADE ALERT TEXT
        alert_text.set_alpha(max(0, alert_text.get_alpha() - 0.05))

        # --- BEAT DETECTION ---
        if current_val > threshold and (current_time - last_beat_time) > 0.4:
            duration = current_time - last_beat_time
            instant_bpm = 60 / duration
            
            if 40 < instant_bpm < 180:
                # A. ARRHYTHMIA CHECK (The "PVC" Detector)
                # If this beat came 20% faster than the average rhythm
                avg_duration = sum(rr_intervals) / len(rr_intervals)
                if duration < (avg_duration * 0.8):
                    alert_text.set_text("⚠️ PREMATURE BEAT")
                    alert_text.set_alpha(1.0)
                    # Log the event
                    with open(LOG_FILE, "a", newline='') as f:
                        csv.writer(f).writerow([time.time(), "0", "0", instant_bpm, "PVC_DETECTED"])

                # Store Data
                rr_intervals.append(duration)
                
                # B. STABLE BPM CALC
                if abs(instant_bpm - statistics.median(bpm_history)) < 30 or len(bpm_history) < 5:
                    bpm_history.append(instant_bpm)
                    new_avg = int(statistics.median(bpm_history))
                    bpm_text.set_text(f"BPM: {new_avg}")
                    bpm_text.set_color('#FF3333')

                # C. HRV CALCULATION (Stress)
                # SDNN: Standard Deviation of beat intervals
                if len(rr_intervals) > 10:
                    sdnn = statistics.stdev(rr_intervals) * 1000 # Convert to ms
                    # Normal SDNN is 30-70ms. Stressed is <30ms. Relaxed is >50ms.
                    hrv_text.set_text(f"HRV: {int(sdnn)} ms")

            last_beat_time = current_time
        
        # Auto-Scale
        view_min = np.percentile(clean_signal, 1) 
        view_max = np.percentile(clean_signal, 99)
        margin = (view_max - view_min) * 0.1
        if margin < 100: margin = 100
        ax.set_ylim(view_min - margin, view_max + margin)
    
    line.set_ydata(filtered_buffer)
    return line, bpm_text, hrv_text, alert_text

ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
plt.show()