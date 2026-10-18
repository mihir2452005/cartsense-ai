# 🧠 CartSense AI — Online Shopper Purchase Intent Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/Deployed-AWS%20EC2-FF9900?logo=amazonaws)](http://YOUR_EC2_PUBLIC_IP:8080)

> **Live Demo:** [http://YOUR_EC2_PUBLIC_IP:8080](http://YOUR_EC2_PUBLIC_IP:8080)  
> **GitHub:** [https://github.com/mihir2452005/cartsense-ai](https://github.com/mihir2452005/cartsense-ai)

---

## 📌 Overview

**CartSense AI** is a machine learning web application that predicts whether an online shopper will complete a purchase based on their browsing session behaviour. It uses **Logistic Regression** trained on 2,000 simulated shopper sessions and displays the purchase probability with an animated gauge bar.

### Key highlights
- **Model:** Logistic Regression with StandardScaler pipeline
- **Accuracy:** ~79.5% on held-out test set
- **UI:** Professional glassmorphism dark-themed web interface
- **API:** REST endpoint returning JSON prediction + probability
- **Deployment:** Dockerized and deployed on AWS EC2

---

## 🎯 Input Features

| Feature | Description | Example |
|---|---|---|
| `age` | Customer age (years) | 28 |
| `time_on_site_min` | Time spent on site (minutes) | 15.0 |
| `pages_viewed` | Number of pages viewed | 20 |
| `cart_items` | Items added to cart | 3 |
| `previous_purchases` | Number of previous purchases | 5 |
| `discount_offered` | Discount offered (1 = yes, 0 = no) | 1 |

---

## 🗂️ Project Structure

```
cartsense-ai/
├── train.py            # Dataset generation + model training
├── app.py              # Flask web app (UI + REST API)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build instructions
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/mihir2452005/cartsense-ai.git
cd cartsense-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```
Expected output:
```
Accuracy: 0.795
              precision    recall  f1-score
 No Purchase       0.84      0.66      0.74
    Purchase       0.77      0.90      0.83
    accuracy                           0.80
```

**4. Start the app**
```bash
python app.py
```
Open [http://localhost:8080](http://localhost:8080)

---

## 🐳 Docker

**Build and run with Docker:**
```bash
docker build -t cartsense-ai:1.0 .
docker run -d --name cartsense-ai-app -p 8080:8080 cartsense-ai:1.0
```
Open [http://localhost:8080](http://localhost:8080)

**Stop and remove:**
```bash
docker stop cartsense-ai-app
docker rm cartsense-ai-app
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` / `POST` | `/` | Web UI — form-based prediction |
| `POST` | `/predict` | REST API — JSON prediction |
| `GET` | `/health` | Health check |

### POST `/predict` — Example

**Request:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "time_on_site_min": 15,
    "pages_viewed": 20,
    "cart_items": 3,
    "previous_purchases": 5,
    "discount_offered": 1
  }'
```

**Response:**
```json
{
  "prediction": "Likely to PURCHASE",
  "purchase_probability": 74.28
}
```

### GET `/health` — Example

```bash
curl http://localhost:8080/health
```
```json
{"status": "ok"}
```

---

## ☁️ AWS EC2 Deployment

The application is deployed on an **AWS EC2 Ubuntu instance** running inside a Docker container.

| Detail | Value |
|---|---|
| Cloud Provider | Amazon Web Services (AWS) |
| Service | EC2 (Elastic Compute Cloud) |
| OS | Ubuntu Server LTS |
| Container Runtime | Docker |
| Port | 8080 |
| Live URL | [http://YOUR_EC2_PUBLIC_IP:8080](http://YOUR_EC2_PUBLIC_IP:8080) |

### Deployment steps (summary)
```bash
# On EC2 after SSH in
cd /home/ubuntu/cartsense-ai
sudo docker build -t cartsense-ai:1.0 .
sudo docker run -d --name cartsense-ai-app --restart unless-stopped -p 8080:8080 cartsense-ai:1.0
```

---

## 📊 Dataset

`train.py` programmatically generates a realistic 2,000-session shopper dataset using `numpy`. The purchase probability is derived from a logistic function weighted by time on site, cart items, previous purchases, and discount availability. The dataset is saved to `data/shoppers.csv`.

No external download required — the dataset is created automatically when `train.py` runs.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn (Logistic Regression + StandardScaler) |
| Web Framework | Flask 3.x |
| Serialization | Joblib |
| Production Server | Gunicorn |
| Containerization | Docker |
| Cloud | AWS EC2 |

---

## 📁 Part of 2CEAI702 — Machine Learning Operations Practical File
