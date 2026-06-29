# OptiFit — Interactive ML Optimization & Regression Studio

OptiFit is a full-stack interactive machine learning platform engineered to visualize, analyze, and benchmark numerical optimization algorithms (Gradient Descent variants) from first principles. Built without high-level ML libraries (e.g., Scikit-Learn), OptiFit demonstrates exact mathematical convergence, error reduction trajectories, and real-time inference across diverse benchmarks.

---

## 🌟 Key Engineering Highlights

- **Pure Mathematical Optimization Core**: Vectorized implementation of Gradient Descent variants built entirely in Python using NumPy.
- **Optimization Algorithms**:
  - **Batch Gradient Descent**: Computes exact loss gradients over full benchmark corpora for deterministic convergence paths.
  - **Stochastic Gradient Descent (SGD)**: Implements per-sample stochastic parameter updates to escape saddle points.
  - **Mini-Batch Gradient Descent**: Vectorized mini-batch sampling balancing convergence velocity with computational throughput.
- **Real-Time Evaluation Metrics**:
  - Automated calculation of **Coefficient of Determination ($R^2$ Score)** and **Mean Absolute Error (MAE)**.
  - Interactive Loss vs. Epoch trajectories.
  - Animated regression curve fitting across real-world benchmark datasets.
- **Dynamic Feature Inference Engine**: Real-time prediction suite featuring automated Z-score standardization ($\mu=0, \sigma=1$) and domain extrapolation verification.

---

## 💻 Tech Stack

- **Frontend**: React 18, Vite, Recharts, Framer Motion, Lucide Icons, Modern CSS Design System
- **Backend**: FastAPI, Python 3.10+, NumPy, Uvicorn, Pydantic v2

---

## 🧮 Mathematical Foundation

### 1. Mean Squared Error (MSE) Cost Function
$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

### 2. Parameter Updating Rules
$$w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

### 3. Partial Derivatives
$$\frac{\partial J}{\partial w} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i, \quad \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

### 4. Model Evaluation Metrics
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}, \quad \text{MAE} = \frac{1}{m} \sum |y_i - \hat{y}_i|$$

---

## ⚡ Quick Start

### 1. Backend Service (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Dashboard (React + Vite)
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
│   ├── main.py          # FastAPI application, numerical optimization math & API routes
│   └── requirements.txt # Python dependency specification
├── frontend/
│   ├── src/
│   │   ├── components/  # Chart visualizers, control panels & metric cards
│   │   ├── utils/       # REST API integration layer
│   │   ├── App.jsx      # Core dashboard layout & state engine
│   │   └── index.css    # Design tokens & styling framework
│   ├── index.html
│   └── package.json
├── .gitignore           # Repository exclusion definitions
└── README.md            # Architecture overview & documentation
```

--
