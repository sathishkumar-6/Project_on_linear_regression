# Linear Regression Visualizer

An interactive, full-stack machine learning application built to visualize and analyze Linear Regression algorithms from scratch. This project provides real-time step-by-step visualization of gradient descent optimization, cost function reduction, and dynamic prediction models.

---

## 🚀 Key Features

- **Algorithms Built From Scratch**: Pure mathematical implementation of Gradient Descent using Python & NumPy (no Scikit-Learn).
- **Optimization Variants**:
  - **Batch Gradient Descent**: Computes gradients over the full dataset for smooth, stable updates.
  - **Stochastic Gradient Descent (SGD)**: Updates parameters per data point for fast convergence.
  - **Mini-Batch Gradient Descent**: Balances stability and efficiency across sub-batches.
- **Interactive Visualizations**:
  - Live animated regression line fitting the dataset.
  - Real-time Cost (MSE) curve plotting iterations vs error.
  - Interactive learning rate comparisons to observe convergence vs divergence.
- **Real-Time Prediction Engine**: Input arbitrary features to forecast outcomes with automatic feature scaling and inverse transformations.

---

## 🛠️ Tech Stack

- **Frontend**: React 18, Vite, Recharts, Framer Motion, Lucide Icons
- **Backend**: FastAPI, Python 3.10+, NumPy, Uvicorn, Pydantic

---

## 📊 Mathematical Foundation

### Mean Squared Error (MSE) Cost Function
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

### Gradient Descent Parameter Updates
$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

### Partial Derivatives
$$\frac{\partial J}{\partial w} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i$$
$$\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

> **Note**: All input features are normalized using Standard Scaling (Zero-Mean, Unit-Variance) before training to guarantee optimal numerical stability.

---

## ⚡ Quick Start

### 1. Backend Setup (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 📁 Project Structure

```
Linear_Regression_project/
├── backend/
│   ├── main.py          # FastAPI server, ML mathematical algorithms & endpoints
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/  # Interactive UI elements & charts
│   │   ├── utils/       # API integration service
│   │   ├── App.jsx      # Main dashboard layout
│   │   └── index.css    # Modern CSS design tokens
│   ├── index.html
│   └── vite.config.js
├── .gitignore           # Clean repository exclusions
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

---

## 📄 License

Distributed under the MIT License.
