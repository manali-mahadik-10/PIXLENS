# 🔬 PIXLENS
### Pixel Intelligence & Learning Enhancement System
**NEP 2020 Initiative | Digital Image & Video Processing (DIVP)**

A full-stack image processing toolkit built as a NEP 2020 initiative project.
50 DIVP filters implemented from scratch in pure NumPy/SciPy, served via a
Flask REST API, with a live browser frontend and SQLite history database.

## Stack
- **Backend:** Python · Flask · NumPy · SciPy · OpenCV · SQLite
- **Frontend:** Vanilla HTML/CSS/JS (single file, zero dependencies)
- **Filters:** 50 filters across 11 DIVP categories

## How to Run
Terminal 1 — Backend:
cd backend
pip install -r requirements.txt
python app.py

Terminal 2 — Frontend:
cd frontend
python -m http.server 8080

Then open: http://localhost:8080