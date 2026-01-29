# 📞 Best Time to Call

Predict optimal call times for maximum conversion using machine learning.

## Quick Start

### 1. Clone/Download this project

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add training data
Place `carInsurance_train.csv` in the `data/` folder:
```
best-time-to-call/
├── data/
│   └── carInsurance_train.csv  ← Add here
```

### 4. Train the model
```bash
python train_model.py
```

### 5. Run the app
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Deploy to Streamlit Cloud (Free)

1. Push this project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → Select your repo → Deploy

---

## Features

| Feature | Description |
|---------|-------------|
| Single Lead | Enter lead details, get best call time |
| CSV Upload | Batch predictions for multiple leads |
| Dashboard | View conversion patterns and insights |
| Technical Details | Toggle to see model internals |

---

## Project Structure

```
best-time-to-call/
├── app.py              # Streamlit app
├── train_model.py      # Model training
├── requirements.txt    # Dependencies
├── README.md           # This file
├── data/
│   └── carInsurance_train.csv
└── model/              # Generated after training
    ├── xgb_model.joblib
    ├── label_encoders.joblib
    ├── feature_cols.joblib
    └── categories.joblib
```
