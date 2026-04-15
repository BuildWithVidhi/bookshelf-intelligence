# 📚 BookScan — Reading Universe Visualizer

BookScan is an AI-powered system that scans physical books using a camera, extracts information via OCR, identifies them using Google Books API, and visualizes your personal library as an interactive 3D knowledge graph.

---

## 🚀 Features

### 📸 Smart Book Scanning

* Live camera feed with detection box
* Auto-detection when the book is stable
* Manual snap option available
* Motion-based stability detection

### 🔍 OCR + AI Identification

* Dual OCR pipeline:

  * Tesseract
  * EasyOCR (optional)
* Cleans and processes extracted text
* Matches books using Google Books API
* Confidence-based filtering system

### 📊 Intelligent Data System

* Stores scanned books with metadata:

  * Title, Author, Genre
  * Description, Publisher, Year
  * ISBN, Page count, Language
* Duplicate detection
* Persistent storage (`graphs.json`)

### 🌌 3D Visualization (Three.js)

* Cluster view (KMeans-based grouping)
* Genre orbit view
* Network graph (author/genre links)
* Timeline visualization

### 🧠 Clustering Engine

* Uses scikit-learn KMeans
* Feature space:

  * Genre encoding
  * Author encoding
  * Year + Page normalization

---

## 🏗️ Tech Stack

### Backend

* Python
* Flask
* OpenCV
* pytesseract
* EasyOCR (optional)
* scikit-learn
* NumPy, Pillow

### Frontend

* HTML/CSS
* JavaScript
* Three.js (3D rendering)

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone <your-repo-url>
cd project
```

### 2. Install dependencies

```
pip install flask flask-cors opencv-python pytesseract easyocr requests numpy Pillow scikit-learn
```

### 3. Install Tesseract OCR

* Windows: Install from official installer
* Linux:

```
sudo apt install tesseract-ocr
```

---

## ▶️ Running the App

```
python main.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 📷 Usage

1. Click **Start Camera**
2. Place a book inside the detection box
3. Hold steady → auto scan triggers
4. Confirm or reject detected book
5. Watch your **3D library grow**

---

## 🔌 API Endpoints

### Camera

* `POST /api/camera/start`
* `POST /api/camera/stop`

### Scan

* `GET /api/scan-result`
* `POST /api/snap`

### Library

* `GET /api/library`
* `DELETE /api/library/<id>`

### Book Actions

* `POST /api/confirm`
* `POST /api/reject`

### Visualization

* `GET /api/graph-data`
* `GET /api/cluster`

---

## 🧠 How It Works

1. Camera captures frames
2. Stability detection triggers capture
3. OCR extracts text
4. Text cleaned → query generated
5. Google Books API fetches metadata
6. Confidence scoring validates result
7. Book stored + visualized

---

## ⚠️ Known Limitations

* OCR accuracy depends on lighting and book quality
* Camera access may fail on some systems
* Google Books API may not always return results
* Development server not suitable for production

---

## 🚀 Future Improvements

* Mobile app version
* Barcode scanning integration
* Better OCR (deep learning models)
* Cloud sync for library
* Recommendation system

---

## 🛠️ Production Deployment

Use a WSGI server instead of Flask dev server:

```
pip install gunicorn
gunicorn main:app
```

---

## 👤 Author

Developed as part of an AI + Computer Vision project
Focused on merging physical books with digital intelligence.

---

## ⭐ Final Note

This is not just a scanner — it's a **personal knowledge universe builder**.

Scan. Understand. Visualize.

---
