This project implements a complete rPPG pipeline that takes live video from a phone or webcam, detects the face, extracts regions of interest (forehead/cheeks), and feeds them into a deep learning model to estimate the blood volume pulse and corresponding heart rate in real time. The goal is to provide a low-cost, non-contact alternative to traditional wearable sensors for telemedicine, fitness, and everyday health monitoring.​

🧠 Key Features
CNN-based rPPG using MTTS-CAN-style temporal shift and attention mechanisms for robust pulse extraction from RGB face video.​

Real-time inference pipeline built with OpenCV for video capture and TensorFlow/TensorFlow Lite for model execution.​

No wearables required: heart rate estimated purely from camera frames, eliminating the need for smartwatches, chest straps, or finger probes.​

Basic motion and illumination handling using facial landmark tracking and temporal filtering to reduce noise and artifacts in the extracted signal.​

Evaluation on public rPPG datasets (e.g., UBFC-rPPG, PURE) with ECG/PPG ground truth to benchmark accuracy.​

🏗️ Architecture
Capture video from the phone camera or webcam (30 FPS) using OpenCV.​

Detect the face and define regions of interest (forehead and cheeks) for rPPG signal extraction.​

Feed spatio-temporal patches into an MTTS-CAN-like CNN model that learns to enhance the physiological signal while suppressing motion and lighting variation.​

Post-process the predicted pulse signal with bandpass filtering and spectral analysis to estimate beats-per-minute (BPM).​

Display heart rate and signal quality indicators in real time through a simple UI or command-line visualization.​

🛠️ Tech Stack
Languages: Python

Libraries: OpenCV, TensorFlow / TensorFlow Lite, NumPy, SciPy

Modeling: MTTS-CAN-style rPPG network for camera-based vital sign estimation​

Datasets (optional for training/eval): UBFC-rPPG, PURE, and other rPPG datasets supported by rPPG-toolbox/pyVHR.​

🎯 What This Project Demonstrates
Practical implementation of camera-based heart rate estimation using deep learning and rPPG.​

Experience with real-time computer vision pipelines, temporal signal processing, and deployment-oriented optimization (e.g., moving towards mobile-ready models).​

Strong hands-on skills in Python, CNN-based architectures, and integrating research ideas like MTTS-CAN into an end-to-end application.​

Disclaimer: This project is for research and educational purposes only and is not a certified medical device. It should not be used for diagnosis or treatment decisions.
