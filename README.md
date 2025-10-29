#  Smart Doorbell with Face Recognition and Liveness Detection

An **AI-powered smart doorbell** system built on **Raspberry Pi 5** and **Orbbec Astra** depth camera.  
It performs **face detection, recognition, and liveness verification** to distinguish real humans from photos, then sends instant **Telegram alerts** to homeowners.

---

##  System Overview
The system combines **RGB + Depth** information for reliable access control.  
Faces are detected using **SCRFD**, recognized with **ArcFace embeddings**, and verified through a **depth-based GrayCNN** classifier trained on fake/real depth patterns.

---

##  Features
-  **AI Face Detection & Recognition** – SCRFD + ArcFace  
-  **Depth-based Liveness Detection** – custom CNN on Astra depth image  
-  **Access Control** – unlocks door via Arduino + servo  
-  **Telegram Notification** – sends image & timestamp when stranger detected  
-  **Multi-threaded pipeline** – live streaming, detection, and message queue  
-  **Face Database** – local embedding storage for known persons  

---

##  Hardware Setup
| Component | Description |
|------------|-------------|
| Raspberry Pi 5 | Main controller |
| Orbbec Astra | RGB-Depth camera |
| Arduino UNO | Servo control & IR sensors |
| SG90 Servo | Door lock |
| IR Sensor | Motion detection |
| Wi-Fi | Telegram alert via Bot API |

---

##  Software Stack
- Python 3.10  
- PyTorch  
- OpenNI2 (for Astra)  
- OpenCV  
- Telegram Bot API  
- SCRFD + ArcFace models  
- Custom depth CNN (`GrayCNN`)

---

##  How It Works

### 1️ Face Detection and Recognition
Detects faces using **SCRFD** and generates **ArcFace embeddings**.  
If the embedding matches the database → access granted.

---

### 2️ Depth-based Liveness Check
Extracts depth ROI and feeds it into the **GrayCNN** model to verify **real vs fake**.

<p align="center">
  <img width="200" height="200" alt="Depth Liveness Example" src="https://github.com/user-attachments/assets/62ef0b34-8e9a-4277-a4f4-c08ead1538fb" />
</p>

---

### 3️ Stranger Detection and Alert
If an **unknown person** stays longer than **2 seconds**,  
a cropped face image is automatically sent to **Telegram** with timestamp and alert message.

---

##  System Flow
<p align="center">
  <img width="700" alt="System Flow" src="https://github.com/user-attachments/assets/01d9338f-ed62-4871-a73b-3ae9da397857" />
</p>

---

## 💬 Telegram Alert Example
<p align="center">
  <img width="380" alt="Telegram Alert" src="https://github.com/user-attachments/assets/b00abb94-9630-4544-95de-97420ed711d4" />
</p>

---

##  Developed by
**Minh Tân – HCMUTE Robotics Lab**  
 Ho Chi Minh City University of Technology and Education  

---

> “Smart security powered by deep learning and depth vision.”
