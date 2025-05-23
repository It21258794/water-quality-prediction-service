# water-quality-prediction-service

The water quality prediction module is a key part of the system, designed to forecast essential parameters such as pH, conductivity, and turbidity throughout the treatment process. It leverages both Long Short-Term Memory (LSTM) networks and Graph Neural Networks (GNNs) to deliver accurate and timely predictions.

+ LSTM-Based Prediction:
A general LSTM model is trained using historical and real-time sensor data from a single treatment center. It captures time-based trends in water quality, enabling early detection of anomalies and proactive process control.

+ GNN-Based Enhancement:
To improve accuracy, a GNN model incorporates data from two nearby treatment centers around the target location. This allows the model to understand spatial relationships and environmental similarities, enhancing prediction robustness across different geographic contexts.

By combining temporal and spatial learning, this hybrid approach ensures more reliable water quality forecasting, supports better decision-making, and contributes to optimized chemical usage and improved purification efficiency.


## Architecture diagram

<img width="8060" alt="Water Quality Prediction - Conceptual Diagram (6)" src="https://github.com/user-attachments/assets/52e4afe1-3530-4eb4-af8e-8072b1344a0a" />


# ML Prediction Service API

A high-performance machine learning prediction service built with **FastAPI**, containerized with **Docker**, and powered by **PyTorch**, **TensorFlow**, and **scikit-learn**. Ideal for real-time predictions and scalable deployments.

---

##  Features

- ⚡ Asynchronous FastAPI backend
- 🧠 ML inference using PyTorch, TensorFlow, and Scikit-learn
- 🐳 Dockerized for smooth deployment
- 📊 Efficient data handling with Pandas and NumPy

---

##  Tech Stack

| Component       | Technology               |
|-----------------|--------------------------|
| Framework       | FastAPI                  |
| Language        | Python 3.10 (slim)       |
| ML Libraries    | PyTorch, TensorFlow, scikit-learn |
| API Server      | Uvicorn                  |
| Containerization| Docker                   |

---


### requirements

- fastapi==0.110.1     
- uvicorn[standard]==0.29.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.4.2
- torch-geometric==2.5.3
- tensorflow==2.16.2

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/prediction-service.git
cd prediction-service

```

### Build the Docker Image

```bash
docker build -t prediction-service .
```
###  Run the Container
```bash
docker run -d -p 8000:8000 prediction-service
```


