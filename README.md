# Best Time to Call

Sales teams often use fixed-interval dialing—calling every lead at the same times regardless of who they are. This wastes agent time and annoys customers. This tool uses machine learning to predict the optimal hour to call each lead based on their profile (age, job, location, etc.), maximizing the chance of conversion.

The model learns patterns from historical call data: retired customers convert better in the morning, working professionals in the evening, and so on. Instead of guessing, sales reps get data-driven recommendations for when to reach out.

**Live Demo:** [best-time-to-call.streamlit.app](https://best-time-to-call.streamlit.app/)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
