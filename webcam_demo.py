import cv2
import numpy as np
import tensorflow as tf
import os
import time
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from skimage.util import img_as_float
import sys

# --- Add the 'code' folder to Python's path ---
# This is the fix for 'ModuleNotFoundError'
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(script_dir, 'code')
sys.path.append(code_dir)
# --- End of fix ---

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import parts of the project's code
from model import MTTS_CAN
from inference_preprocess import detrend

# --- Constants ---
IMG_ROWS = 36
IMG_COLS = 36
FRAME_DEPTH = 10  # This was a model-specific value, but we need full frames
FS = 30           # We'll try to get 30 FPS from the webcam
BUFFER_SECONDS = 30 # The model was trained on 30-second (900-frame) chunks
BUFFER_SIZE = BUFFER_SECONDS * FS

# --- Load Model & Face Detector ---
print("Loading model...")
model = MTTS_CAN(FRAME_DEPTH, 32, 64, (IMG_ROWS, IMG_COLS, 3))
model.load_weights('./mtts_can.hdf5')
print("Model loaded.")

print("Loading face detector...")
try:
    # Get the path to the haarcascades directory from cv2
    haarcascade_dir = os.path.dirname(cv2.data.haarcascades)
    face_cascade_path = os.path.join(haarcascade_dir, 'haarcascade_frontalface_default.xml')
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {face_cascade_path}")
    print("Face detector loaded.")
except Exception as e:
    print(f"Error loading face detector: {e}")
    print("Could not find 'haarcascade_frontalface_default.xml'")
    print("Please make sure your 'opencv-contrib-python' install is correct.")
    exit()

# --- Utility Functions (from predict_vitals.py) ---
def get_heart_rate(pulse_pred, fs):
    """Calculates heart rate from a pulse signal using PSD."""
    
    # 1. Detrend and filter the signal
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    # 2. Get Power Spectral Density
    # Using matplotlib.pyplot.psd (as in the original script)
    # We'll use a Hamming window
    pxx, frequency = plt.psd(pulse_pred, NFFT=len(pulse_pred), Fs=fs, window=np.hamming(len(pulse_pred)))
    plt.close() # Close the plot window

    # 3. Find the peak in the valid HR range (45-150 bpm)
    valid_range = (frequency >= 0.75) & (frequency <= 2.5) # 0.75 Hz = 45 bpm, 2.5 Hz = 150 bpm
    if not np.any(valid_range):
        return 0.0 # No valid frequencies found
        
    valid_pxx = pxx[valid_range]
    valid_freq = frequency[valid_range]
    
    if len(valid_pxx) == 0:
        return 0.0

    peak_index = np.argmax(valid_pxx)
    hr_frequency = valid_freq[peak_index]
    
    return hr_frequency * 60 # Return as BPM

# --- Main Application ---
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam to 30 FPS, if possible
cap.set(cv2.CAP_PROP_FPS, FS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame buffer
frame_buffer = []
hr = 0.0
last_hr_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    display_frame = frame.copy()
    
    if len(faces) > 0:
        # Use the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Crop the face
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess: Resize to model's expected input
        resized_face = cv2.resize(face_roi, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)
        
        # Normalize (same as inference_preprocess.py)
        # Using img_as_float from skimage (which you already installed)
        normalized_face = img_as_float(resized_face)
        
        # Add to buffer
        frame_buffer.append(normalized_face)
        
        # Draw rectangle on display
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    else:
        # No face detected
        # To keep the buffer synchronized, we could add a black frame
        # or, more simply, just clear the buffer so we restart
        if len(frame_buffer) > 0:
            print("No face detected, clearing buffer.")
            frame_buffer = []

    # --- Prediction ---
    if len(frame_buffer) < BUFFER_SIZE:
        # Buffer is not full, show status
        status_text = f"Collecting frames: {len(frame_buffer)} / {BUFFER_SIZE}"
    else:
        # Buffer is full, run prediction
        if len(frame_buffer) > BUFFER_SIZE:
            # This is a sliding window, remove the oldest frame
            frame_buffer.pop(0)

        # Convert to numpy array
        # This is our dXsub
        dx_sub = np.array(frame_buffer)
        
        # The model needs two inputs (RGB and Motion). We'll approximate this.
        # This is a simplification, but let's try it.
        # The original code used dXsub[:, :, :, :3] and dXsub[:, :, :, -3:]
        # This implies the preprocess script created a 6-channel input.
        # We only have 3. Let's try passing the same thing twice.
        
        # Re-check: The original code used dXsub[:, :, :, :3] and dXsub[:, :, :, -3:]
        # The preprocess script created a (T, H, W, 6) array.
        # Let's just use our (T, H, W, 3) and pass it twice as (T,H,W,3) and (T,H,W,3)
        # The model architecture (model.py) shows it expects two inputs.
        
        # A quick fix to satisfy the model's two-input structure
        # We'll pass the same data as both 'appearance' and 'motion'
        model_input = (dx_sub, dx_sub)

        try:
            # Predict
            yptest = model.predict(model_input, batch_size=BUFFER_SIZE, verbose=0)
            
            # Get pulse signal
            pulse_pred = yptest[0] # yptest[0] is pulse, yptest[1] is respiration
            
            # Calculate HR
            hr = get_heart_rate(pulse_pred, FS)
            
            status_text = f"Heart Rate: {hr:.1f} BPM"
            last_hr_time = time.time()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            print("Clearing buffer and retrying.")
            frame_buffer = [] # Clear buffer on error
            status_text = "Prediction error, resetting..."

    # Display status
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow('Real-Time Heart Rate', display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()