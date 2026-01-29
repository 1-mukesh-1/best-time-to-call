"""Best Time to Call - Streamlit App."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Best Time to Call", page_icon="phone", layout="wide")

CAT_FEATURES = ['Job', 'Marital', 'Education', 'Communication', 'LastContactMonth', 'Outcome']
NUM_FEATURES = ['Age', 'Balance', 'HHInsurance', 'CarLoan', 'Default', 
                'LastContactDay', 'NoOfContacts', 'DaysPassed', 'PrevAttempts', 'CallHour']


@st.cache_resource
def load_model():
    """Load model artifacts."""
    model = joblib.load("model/xgb_model.joblib")
    label_encoders = joblib.load("model/label_encoders.joblib")
    feature_cols = joblib.load("model/feature_cols.joblib")
    categories = joblib.load("model/categories.joblib")
    return model, label_encoders, feature_cols, categories


def format_hour(hour):
    """Format hour as AM/PM."""
    if hour < 12:
        return f"{hour} AM"
    elif hour == 12:
        return "12 PM"
    return f"{hour-12} PM"


def encode_lead(lead_data, label_encoders):
    """Encode a single lead's categorical features."""
    encoded = lead_data.copy()
    for col in CAT_FEATURES:
        le = label_encoders[col]
        val = encoded.get(col, 'Unknown')
        encoded[col + '_encoded'] = le.transform([val])[0] if val in le.classes_ else -1
    return encoded


def predict_single_lead(model, label_encoders, feature_cols, lead_data):
    """Predict conversion probability for each hour (vectorized)."""
    hours = list(range(9, 18))
    rows = []
    
    encoded = encode_lead(lead_data, label_encoders)
    for hour in hours:
        row = encoded.copy()
        row['CallHour'] = hour
        rows.append(row)
    
    X = pd.DataFrame(rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    
    return pd.DataFrame({'Hour': hours, 'Probability': probs})


def predict_batch(model, label_encoders, feature_cols, df):
    """Predict best hour for multiple leads (fully vectorized)."""
    hours = list(range(9, 18))
    all_rows = []
    
    for idx, row in df.iterrows():
        encoded = encode_lead(row.to_dict(), label_encoders)
        for hour in hours:
            r = encoded.copy()
            r['CallHour'] = hour
            r['_lead_idx'] = idx
            r['_hour'] = hour
            all_rows.append(r)
    
    batch_df = pd.DataFrame(all_rows)
    X = batch_df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    
    batch_df['Probability'] = probs
    
    results = []
    for lead_idx in df.index:
        lead_rows = batch_df[batch_df['_lead_idx'] == lead_idx]
        best_row = lead_rows.loc[lead_rows['Probability'].idxmax()]
        results.append({
            'Lead_Index': lead_idx,
            'Best_Hour': format_hour(int(best_row['_hour'])),
            'Conversion_Probability': f"{best_row['Probability']*100:.1f}%",
            'Probability_Raw': best_row['Probability']
        })
    
    return pd.DataFrame(results)


def single_lead_mode(model, label_encoders, feature_cols, categories):
    """Single lead input form."""
    st.header("Enter Lead Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", options=categories['Job'])
        marital = st.selectbox("Marital Status", options=categories['Marital'])
        education = st.selectbox("Education", options=categories['Education'])
    
    with col2:
        balance = st.number_input("Account Balance ($)", min_value=-10000, max_value=100000, value=1000)
        hh_insurance = st.selectbox("Household Insurance", options=["No", "Yes"])
        car_loan = st.selectbox("Has Car Loan", options=["No", "Yes"])
        default = st.selectbox("Has Default", options=["No", "Yes"])
    
    with col3:
        communication = st.selectbox("Communication Type", options=categories['Communication'])
        last_contact_month = st.selectbox("Last Contact Month", options=categories['LastContactMonth'])
        last_contact_day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
        no_of_contacts = st.number_input("No. of Contacts", min_value=1, max_value=50, value=1)
        days_passed = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=-1)
        prev_attempts = st.number_input("Previous Attempts", min_value=0, max_value=50, value=0)
        outcome = st.selectbox("Previous Outcome", options=categories['Outcome'])
    
    if st.button("Predict Best Time", type="primary"):
        lead_data = {
            'Age': age, 'Job': job, 'Marital': marital, 'Education': education,
            'Balance': balance, 'HHInsurance': 1 if hh_insurance == "Yes" else 0,
            'CarLoan': 1 if car_loan == "Yes" else 0, 'Default': 1 if default == "Yes" else 0,
            'Communication': communication, 'LastContactMonth': last_contact_month,
            'LastContactDay': last_contact_day, 'NoOfContacts': no_of_contacts,
            'DaysPassed': days_passed, 'PrevAttempts': prev_attempts, 'Outcome': outcome
        }
        
        results = predict_single_lead(model, label_encoders, feature_cols, lead_data)
        best_idx = results['Probability'].idxmax()
        best_hour = results.loc[best_idx, 'Hour']
        best_prob = results.loc[best_idx, 'Probability']
        
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Recommendation")
            st.metric("Best Time to Call", format_hour(best_hour), f"{best_prob*100:.1f}% conversion chance")
            
            st.markdown("**Top 3 Time Slots:**")
            for _, row in results.nlargest(3, 'Probability').iterrows():
                st.write(f"{format_hour(int(row['Hour']))}: {row['Probability']*100:.1f}%")
        
        with col2:
            st.subheader("Hourly Breakdown")
            results['Hour_Label'] = results['Hour'].apply(format_hour)
            results['Probability_Pct'] = results['Probability'] * 100
            
            fig = px.bar(results, x='Hour_Label', y='Probability_Pct', color='Probability_Pct',
                        color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcb77'],
                        labels={'Hour_Label': 'Hour', 'Probability_Pct': 'Conversion Probability (%)'})
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)


def csv_upload_mode(model, label_encoders, feature_cols, categories):
    """Batch CSV prediction."""
    st.header("Batch Prediction")
    
    template_df = pd.DataFrame({
        'Age': [35, 55], 'Job': ['management', 'retired'], 'Marital': ['married', 'single'],
        'Education': ['tertiary', 'secondary'], 'Balance': [1000, 5000],
        'HHInsurance': [1, 0], 'CarLoan': [0, 1], 'Default': [0, 0],
        'Communication': ['cellular', 'telephone'], 'LastContactMonth': ['jan', 'feb'],
        'LastContactDay': [15, 20], 'NoOfContacts': [1, 2], 'DaysPassed': [-1, 30],
        'PrevAttempts': [0, 1], 'Outcome': ['NA', 'success']
    })
    
    st.download_button("Download Template CSV", template_df.to_csv(index=False), 
                       "lead_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} leads")
        st.dataframe(df.head())
        
        if st.button("Predict All", type="primary"):
            with st.spinner("Processing..."):
                results_df = predict_batch(model, label_encoders, feature_cols, df)
                
                st.dataframe(results_df[['Lead_Index', 'Best_Hour', 'Conversion_Probability']])
                
                output_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                st.download_button("Download Results", output_df.to_csv(index=False),
                                  "predictions.csv", "text/csv")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Leads", len(results_df))
                col2.metric("Avg Conversion", f"{results_df['Probability_Raw'].mean()*100:.1f}%")
                col3.metric("Most Common Best Hour", results_df['Best_Hour'].mode()[0])


def dashboard_mode():
    """Insights dashboard."""
    st.header("Insights Dashboard")
    
    if not os.path.exists("data/carInsurance_train.csv"):
        st.warning("Training data not found.")
        return
    
    df = pd.read_csv("data/carInsurance_train.csv")
    df['CallHour'] = pd.to_datetime(df['CallStart'], format='%H:%M:%S').dt.hour
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Overall Conversion", f"{df['CarInsurance'].mean()*100:.1f}%")
    col3.metric("Best Hour", format_hour(df.groupby('CallHour')['CarInsurance'].mean().idxmax()))
    col4.metric("Worst Hour", format_hour(df.groupby('CallHour')['CarInsurance'].mean().idxmin()))
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Hour")
        hourly = df.groupby('CallHour')['CarInsurance'].mean().reset_index()
        hourly.columns = ['Hour', 'Conversion']
        hourly['Conversion'] *= 100
        hourly['Hour_Label'] = hourly['Hour'].apply(format_hour)
        
        fig = px.bar(hourly, x='Hour_Label', y='Conversion', color='Conversion',
                    color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcb77'])
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Conversion by Job")
        job_conv = df.groupby('Job')['CarInsurance'].mean().sort_values(ascending=True).reset_index()
        job_conv.columns = ['Job', 'Conversion']
        job_conv['Conversion'] *= 100
        
        fig = px.bar(job_conv, x='Conversion', y='Job', color='Conversion',
                     color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcb77'],
                     orientation='h')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Smart vs Random Calling")
    
    random_rate = df['CarInsurance'].mean()
    best_hour = df.groupby('CallHour')['CarInsurance'].mean().idxmax()
    smart_rate = df[df['CallHour'] == best_hour]['CarInsurance'].mean()
    improvement = ((smart_rate - random_rate) / random_rate) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Random Calling", f"{random_rate*100:.1f}%")
    col2.metric("Smart Calling", f"{smart_rate*100:.1f}%", f"+{improvement:.1f}%")


def main():
    st.title("Best Time to Call")
    st.caption("Predict optimal call times for maximum conversion")
    
    if not os.path.exists("model/xgb_model.joblib"):
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    
    model, label_encoders, feature_cols, categories = load_model()
    
    st.sidebar.title("Mode")
    mode = st.sidebar.radio("Select:", ["Single Lead", "CSV Upload", "Dashboard"])
    
    if mode == "Single Lead":
        single_lead_mode(model, label_encoders, feature_cols, categories)
    elif mode == "CSV Upload":
        csv_upload_mode(model, label_encoders, feature_cols, categories)
    else:
        dashboard_mode()


if __name__ == "__main__":
    main()
