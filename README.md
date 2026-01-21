# Gesture-Controlled Spotify Player 🎵✋
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A real-time hand gesture–controlled media system that allows users to control Spotify playback using a webcam.  
Built using computer vision and hand landmark detection, with carefully designed gesture logic for reliable and intuitive control.

---

## 🚀 Features

- ✌️ **Two-finger hold** → Continuous volume control  
- 👉 **Two-finger horizontal flick** → Next / Previous track  
- 🤏 **Thumb–index pinch** → Play / Pause  
- 🎯 Gesture gating to prevent accidental triggers  
- 🔊 Controls **Spotify only** (not system volume)  
- ⚡ Real-time performance with smooth interaction

---

## 🧠 How It Works

1. Captures live video from webcam
2. Uses MediaPipe Hands to detect 21 hand landmarks
3. Applies rule-based gesture logic:
   - Finger pose detection
   - Distance-based pinch detection
   - Temporal motion analysis for flicks
4. Maps gestures to Spotify actions using AppleScript

This project uses a **pretrained ML model for hand tracking**, combined with **custom gesture logic and real-time control systems**.

---

## 🛠 Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- AppleScript (macOS)
- Spotify Desktop App

---

## 🧪 Gestures

| Gesture | Action |
|------|------|
| ✌️ Hold index + middle finger | Volume up / down |
| 👉 Fast horizontal flick | Next / Previous track |
| 🤏 Thumb + index pinch | Play / Pause |

---

## ▶️ Getting Started

### Prerequisites
- macOS
- Spotify Desktop App
- Python 3.9+

### Installation

```bash
git clone https://github.com/yourusername/gesture-controlled-spotify.git
cd gesture-controlled-spotify
pip install -r requirements.txt
