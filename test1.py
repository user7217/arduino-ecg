import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from scipy.fftpack import fft
import statistics
import csv

# configuration
com_port = '/dev/cu.usbmodem1101'
baud_rate = 57600
window_size = 1500        
fs = 200.0                

# lead-off detection thresholds
# increased to prevent false positives from dc drift
rail_threshold = 300000 
flatline_threshold = 0.1 

# filter settings
lowcut = 0.5
highcut = 75.0 
notch_freq = 50.0 
notch_quality = 30.0 

# data storage
data_buffer = deque(maxlen=window_size)
bpm_history = deque(maxlen=10) 
rr_history_x = deque(maxlen=50) 
rr_history_y = deque(maxlen=50) 
last_beat_time = time.time()
beat_intervals = [] 

# logging
log_file = f"ecg_data_{int(time.time())}.csv"
with open(log_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "raw", "filtered", "bpm", "status"])
print(f"logging to {log_file}...")

# signal processing
def create_filters(low, high, notch, rate, order=4):
    nyq = 0.5 * rate
    b_band, a_band = butter(order, [low/nyq, high/nyq], btype='band')
    b_notch, a_notch = iirnotch(notch, notch_quality, rate)
    return b_band, a_band, b_notch, a_notch

def apply_filters(data, bb, ab, bn, an):
    y = lfilter(bb, ab, data)
    y = lfilter(bn, an, y)
    return y

b_band, a_band, b_notch, a_notch = create_filters(lowcut, highcut, notch_freq, fs)
xf = np.linspace(0.0, fs/2.0, window_size//2) 

# serial connection
try:
    ser = serial.Serial(com_port, baud_rate, timeout=0.1)
    time.sleep(2)
    print("connected. filters active.")
except:
    print("error: port blocked or not found.")
    exit()

# plot setup
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#121212') 
gs = GridSpec(2, 2, figure=fig)

ax_wave = fig.add_subplot(gs[0, :])
ax_wave.set_facecolor('black')
ax_wave.set_title("ecg waveform", color='white')
line_wave, = ax_wave.plot([], [], color='#00FFFF', lw=1.5)
ax_wave.grid(True, color='#333333', linestyle='--')
status_text = ax_wave.text(0.98, 0.90, "init...", transform=ax_wave.transAxes, color='white', fontsize=18, fontweight='bold', ha='right')

ax_fft = fig.add_subplot(gs[1, 0])
ax_fft.set_facecolor('black')
ax_fft.set_title("frequency spectrum", color='white')
line_fft, = ax_fft.plot([], [], color='#FF00FF', lw=1.5)
ax_fft.set_xlim(0, 75) 
ax_fft.set_ylim(0, 5000) 
ax_fft.grid(True, color='#333333', linestyle='--')

ax_poincare = fig.add_subplot(gs[1, 1])
ax_poincare.set_facecolor('black')
ax_poincare.set_title("poincare plot", color='white')
scat_poincare = ax_poincare.scatter([], [], color='#00FF00', s=30, alpha=0.7)
ax_poincare.set_xlim(0.4, 1.2)
ax_poincare.set_ylim(0.4, 1.2)
ax_poincare.grid(True, color='#333333', linestyle='--')
ax_poincare.plot([0, 2], [0, 2], color='gray', linestyle=':', alpha=0.5)

def update(frame):
    global last_beat_time
    
    while ser.in_waiting:
        try:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if raw_line.lstrip('-').isdigit():
                val = int(raw_line)
                # allow larger range before clamping to catch real railing
                val = max(-500000, min(500000, val)) 
                data_buffer.append(val)
        except: pass

    if len(data_buffer) > 100:
        
        # lead-off check
        recent_data = list(data_buffer)[-50:]
        current_raw = data_buffer[-1]
        
        is_railed = abs(current_raw) > rail_threshold
        
        try:
            std_dev = statistics.stdev(recent_data)
            is_flat = std_dev < flatline_threshold
        except: 
            is_flat = False
            std_dev = 0

        if is_railed or is_flat:
            # debug info on screen
            if is_railed:
                status_text.set_text(f"LEADS OFF (RAIL: {current_raw})")
            else:
                status_text.set_text(f"LEADS OFF (FLAT: {std_dev:.2f})")
                
            status_text.set_color('yellow')
            bpm_history.clear() 
            clean = np.zeros(len(data_buffer)) 
            
            with open(log_file, "a", newline='') as f:
                csv.writer(f).writerow([time.time(), current_raw, 0, 0, "LEADS_OFF"])
        
        else:
            raw_arr = np.array(data_buffer)
            if np.isnan(raw_arr).any(): raw_arr = np.nan_to_num(raw_arr)
            
            clean = apply_filters(raw_arr, b_band, a_band, b_notch, a_notch)
            if np.isnan(clean).any(): clean = np.nan_to_num(clean)

            curr_val = clean[-1]
            curr_time = time.time()
            p95 = np.percentile(clean, 95)
            thresh = p95 * 0.70
            
            if curr_val > thresh and (curr_time - last_beat_time) > 0.4:
                duration = curr_time - last_beat_time 
                instant_bpm = 60 / duration
                
                if 0.3 < duration < 1.5: 
                    bpm_history.append(instant_bpm)
                    
                    if len(bpm_history) >= 5:
                        stable_bpm = int(statistics.median(bpm_history))
                        status_text.set_text(f"BPM: {stable_bpm}")
                        status_text.set_color('#FF3333')
                    else:
                        status_text.set_text("calc...")
                        status_text.set_color('white')
                    
                    if len(beat_intervals) > 0:
                        rr_history_x.append(beat_intervals[-1])
                        rr_history_y.append(duration)
                    beat_intervals.append(duration)
                    
                    with open(log_file, "a", newline='') as f:
                        csv.writer(f).writerow([time.time(), data_buffer[-1], clean[-1], instant_bpm, "BEAT"])
                
                last_beat_time = curr_time
            else:
                if (curr_time - last_beat_time) > 3.0:
                    if "BPM" in status_text.get_text():
                        status_text.set_color('white')

        line_wave.set_data(range(len(clean)), clean)
        
        if is_railed or is_flat:
             ax_wave.set_ylim(-1000, 1000)
        else:
            mi, ma = np.min(clean), np.max(clean)
            if ma == mi: ma = mi + 1
            margin = (ma - mi) * 0.1
            ax_wave.set_ylim(mi - margin, ma + margin)
        
        ax_wave.set_xlim(0, len(clean))

        yf = fft(clean)
        mag = 2.0/window_size * np.abs(yf[0:len(clean)//2])
        line_fft.set_data(xf[:len(mag)], mag)
        if len(mag) > 0:
            max_f = np.max(mag[1:])
            ax_fft.set_ylim(0, max_f * 1.2 if max_f > 10 else 100)

        if len(rr_history_x) > 0:
            scat_poincare.set_offsets(np.c_[rr_history_x, rr_history_y])

    return line_wave, line_fft, scat_poincare, status_text

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()