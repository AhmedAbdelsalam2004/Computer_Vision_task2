# VisionLab 👁️

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)

VisionLab is a full-stack web application built to implement, visualize, and interact with core Computer Vision algorithms. Designed with a focus on mathematical transparency and performance, the core processing algorithms—including Canny Edge Detection, Hough Transforms, and Active Contours—are implemented entirely **from scratch** using highly optimized NumPy matrix operations, bypassing standard pre-packaged OpenCV solver functions.

## ✨ Key Features

### Shape & Edge Detection

* **Canny Edge Detector:** Fully vectorized implementation featuring custom Gaussian blurring, Sobel filtering, Non-Maximum Suppression, and Hysteresis thresholding.
* **Hough Line Transform:** Custom accumulator matrix voting to detect straight lines based on dynamic, user-defined voting thresholds.
* **Hough Circle Transform:** Pure NumPy-based circle detection utilizing 3D accumulators and customizable radius ranges.
* **Ellipse Detection:** Algebraic least-squares conic fitting in normalized space to accurately detect and draw ellipses.

### Active Contour Model (Snakes)

* **Physics-Based Evolution:** Simulates elasticity (tension) and stiffness (rigidity) forces to iteratively snap a flexible curve to image boundaries.
* **Implicit Euler Integration:** Utilizes an inverted pentadiagonal cyclic band matrix to solve the entire system of equations instantly, allowing for smooth, vibration-free curve evolution.
* **Automated Shape Metrics:** Calculates the Freeman Chain Code (8-connectivity), perimeter (Euclidean distance), and area (vectorized Shoelace formula) of the final converged contour.

### Interactive UI & Session Management

* **Real-time Parameter Tuning:** Adjust hysteresis thresholds, voting limits, and snake physics with immediate visual feedback.
* **Robust Session History:** The Django backend strictly tracks image states, historical slider values, and action logs, enabling flawless **Undo** and **Reset** functionality.
* **Pristine Source Processing:** Algorithms are strictly applied to the original uploaded image to prevent recursive filter degradation during rapid parameter testing.

## 🛠️ Tech Stack

* **Frontend:** React.js, Vite, Axios, CSS3
* **Backend:** Django, Django REST Framework (DRF)
* **Computer Vision:** NumPy (Matrix Operations), OpenCV (Basic I/O & Filtering), Pillow (Image processing)

## 🚀 Installation & Local Setup

Ensure you have Python 3.8+ and Node.js installed on your machine.

### 1. Backend Setup (Django)

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required dependencies
pip install django djangorestframework django-cors-headers numpy opencv-python pillow

# Run database migrations for session management
python manage.py migrate

# Start the Django development server (runs on port 8000)
python manage.py runserver
```

### 2. Frontend Setup (React/Vite)

```bash
# Navigate to the frontend directory
cd frontend

# Install Node modules
npm install

# Start the Vite development server (runs on port 5173)
npm run dev
```

### 3. Usage

Open your browser and navigate to the Vite local URL (usually <http://localhost:5173>).

Upload an image via the sidebar workspace.

Toggle between Shape Detection and Active Contour modes using the top navigation bar.

Adjust the sliders to see how the algorithms react to different mathematical thresholds.

## 📄 License

This project is open-source and available under the MIT License.
