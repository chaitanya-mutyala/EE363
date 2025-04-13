# 🔋 Battery SOH & RUL Prediction using ML Models

🔬 This project focuses on accurate battery degradation forecasting using deep learning, enabling proactive maintenance and management of lithium-ion batteries.

This repository presents machine learning-based approaches to predict the **State of Health (SOH)** and **Remaining Useful Life (RUL)** of lithium-ion batteries using real-world datasets from NASA.  
Trained on battery charge cycle data, the models are visualized and evaluated through an interactive web interface.

---

## 🧠 Core Machine Learning Models

This project emphasizes three key deep learning architectures for time series regression:

### 🔸 1. MLP (Multi-Layer Perceptron)

- A fully connected feedforward neural network.
- Trained on 5-length historical SOH (%) sequences.
- Suitable for basic pattern recognition and fast training.
- Acts as a **baseline model** for comparison with sequence-aware networks.

### 🔸 2. LSTM (Long Short-Term Memory)

- Recurrent neural network architecture ideal for sequential data.
- Learns long-term dependencies in battery degradation patterns.
- Outperforms MLP in capturing time-dependent trends in SOH prediction.
- Trained using sliding window sequences of battery cycles.

### 🔸 3. Transformer

- State-of-the-art architecture used in natural language processing and time series tasks.
- Uses **self-attention** to capture global dependencies across all sequence points.
- Provides better generalization on longer sequences compared to RNNs.
- Particularly effective in modeling capacity degradation without recurrence.

### 📈 Common Features Across Models

- Input: Sequences of 5 previous SOH values.
- Output: Next SOH prediction and Remaining Useful Life (RUL) estimation.
- Metrics: MAE, RMSE score for evaluation.

---

## 📁 Dataset

We use the publicly available `.mat` datasets from NASA’s Prognostics Data Repository:
- `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0008.mat`
- Extracted charge cycles and integrated current to calculate **capacity (Ah)** and derive **SOH (%)**.
- The dataset was preprocessed to compute capacity from voltage, current, and temperature measurements using standard equations, and SOH was computed as : SOH = 𝐶𝑡 / 𝐶initial × 100

---

## 📂 Project Structure
Code/ 
   ├── assests/ # first slide of ppt
   |    ├── *.pdf
   ├── models/ # Jupyter notebooks for experimentation 
   |    ├── *.ipynb
   ├── datasets/ # NASA .mat battery datasets 
   |    ├── *.mat
   ├── backend/ # FastAPI backend - model execution logic 
   |    ├── models/  #python code of models
   |    ├── outputs/ # Generated plots (SOH, RUL) 
   |    ├── app.py
   |    ├── data_utiles.py
   |    ├── model_runner.py
   ├── frontend/ # Streamlit-based frontend UI  
   |    ├── app.py
   ├── uploads/ #if any dataset upload by user it stores here
   |    ├──  *.mat
   ├── data/
   |    ├── download.sh
   ├── requirements.txt 
   └── README.md


---

## 🚀 Features

- Upload your own `.mat` dataset or choose from preloaded NASA datasets.
- Choose one or more ML models: `MLP`, `LSTM`, `Transformer`.
- View:
  - Predicted vs Actual SOH plot
  - RUL over cycles plot
  - Performance metrics (MAE, RMSE)
  - Cycle-wise prediction table

---

## 🔧 How to Run

### 1. Clone this repository

```bash
git clone https://github.com/spkkarri/EE363/tree/main/25/08_209/Code.git
cd sohprediction 
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the backend (FastAPI)

```bash 
uvicorn backend.app:app --reload
```
 
### 4. Run the frontend (Streamlit) 
in another terminal

```bash
streamlit run frontend/app.py
```

---

## 📽️ YouTube Demo

📺 Watch Full Demo Here
<!-- Replace with your actual YouTube demo link -->

---

## 👨‍💻 Team Information

Team No: 3
Team Lead Roll Number (Last 3 Digits): 209
Course: EE363 – Machine Learning for Engineers
Team Members:
522209 : M.Ch.N.S.S.Rama Krishna
522142 : Kandula Sai Vardhan
522113 : B.S.V.Lokesh
522109 : B.Pavan Kumar
522147 : K.Mani Kumar Abhi
522139 : K.Bhargav Sasi Keerthan



---

