import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from cleaning import run_icu_processing

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR.parent

@st.cache_resource
def load_assets():
    assets = {
        'cb': joblib.load(path / 'best_cb.pkl'),
        'lgb': joblib.load(path / 'best_lgb.pkl'),
        'lr': joblib.load(path / 'best_lr.pkl'),
        'imputer': joblib.load(path / 'median_imputer.pkl'),
        'scaler': joblib.load(path / 'standard_scalar.pkl'),
        'feature_cb': joblib.load(path / 'feature_cb.pkl'),
        'feature_lgb': joblib.load(path / 'feature_lgb.pkl'),
        'feature_lr': joblib.load(path / 'feature_lr.pkl'),
        'feature_cols': joblib.load(path / 'feature_columns.pkl')
    }
    return assets

assets = load_assets()

COL_FEATURES = assets['feature_cols']


st.set_page_config(page_title='ICU Mortality Predictor', layout='wide')
st.title('ICU Mortality Ensemble Dashboard')
st.markdown("Predicting clinical risk based on 48-hour ICU physiological trends.")


st.sidebar.header('🔧 Ensemble Configuration')
w_cb_raw = st.sidebar.slider('CatBoost Weight', 0.0, 1.0, 0.5)
w_lgb_raw = st.sidebar.slider('LightGBM Weight', 0.0, 1.0, 0.4)
w_lr_raw = st.sidebar.slider('Logistic Weight', 0.0, 1.0, 0.1)

total_weight = w_cb_raw + w_lgb_raw + w_lr_raw
if total_weight > 0:
    w_cb, w_lgb, w_lr = w_cb_raw/total_weight, w_lgb_raw/total_weight, w_lr_raw/total_weight
else:
    w_cb, w_lgb, w_lr = 0.33, 0.33, 0.34

st.sidebar.info(f"**Effective Weights:**\nCB: {w_cb:.2f} | LGB: {w_lgb:.2f} | LR: {w_lr:.2f}")

threshold = st.sidebar.number_input('Risk Threshold', value=0.35, step=0.05)


if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'X_linear' not in st.session_state:
    st.session_state.X_linear = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None


tab1, tab2 = st.tabs(['📊 Batch Processing', '👤 Single Patient Analysis'])

with tab1:
    st.header('Step 1: Patient Data Processor')
    
    ts_files = st.file_uploader("Upload Patient Records (.txt)", type="txt", accept_multiple_files=True)

    if ts_files:
        all_patients = []
        with st.spinner(f"Processing {len(ts_files)} records..."):
            for ts_file in ts_files:
                df = run_icu_processing(ts_file)
                                
                col_miss = [c for c in df.columns if c.endswith('_Miss')]
                df['Total_Missing_Sum'] = df[col_miss].sum(axis=1)
                
                col_count = [c for c in df.columns if c.endswith('_Count')]
                df[col_count] = df[col_count].apply(np.log1p)
                df = df.rename(columns={c: c.replace('_Count', '_Log_Count') for c in col_count})

                col_last_minute = [c for c in df.columns if c.endswith('_Last_Minute')]
                for col in col_last_minute:
                    gap_name = col.replace('_Last_Minute', '_Last_Gap')
                    df[gap_name] = 2880 - df[col]
                
                col_gap = [c for c in df.columns if c.endswith('_Last_Gap')]
                df['Global_Last_Gap'] = df[col_gap].min(axis=1)
                df = df.drop(columns=col_last_minute)

                df['Gender'] = df['Gender'].fillna(-1)

                for i in [1.0, 2.0, 3.0, 4.0]:
                    col_name = f'ICUType_{i}'
                    df[col_name] = (df['ICUType'] == i).astype(int) if 'ICUType' in df.columns else 0

                df['Shock_Index'] = df['HR_Final'] / df['UniSysABP_Final']
                df['GCS_MAP_Product'] = df['GCS_Final'] * df['UniMAP_Final']
                df['Anion_Gap'] = (df['Na_Final'] + df['K_Final']) - df['HCO3_Final']

                current_id = df['RecordID'].values[0] if 'RecordID' in df.columns else ts_file.name
                
                df_inference = df.reindex(columns=COL_FEATURES)
                df_inference.index = [current_id]
                all_patients.append(df_inference)

            X_input = pd.concat(all_patients).replace([np.inf, -np.inf], np.nan)
            st.session_state.df_processed = X_input

            with st.spinner("Calculating Ensemble Risk..."):
                X_imputed = pd.DataFrame(assets['imputer'].transform(X_input), columns=X_input.columns, index=X_input.index)
                st.session_state.X_linear = pd.DataFrame(assets['scaler'].transform(X_imputed), columns=X_input.columns, index=X_input.index)

                p_cb = assets['cb'].predict_proba(X_input[assets['feature_cb']])[:, 1]
                p_lgb = assets['lgb'].predict_proba(X_input[assets['feature_lgb']])[:, 1]
                p_lr = assets['lr'].predict_proba(st.session_state.X_linear[assets['feature_lr']])[:, 1]

                final_prob = (w_cb * p_cb) + (w_lgb * p_lgb) + (w_lr * p_lr)

                st.session_state.results_df = pd.DataFrame({
                    'Patient_ID': X_input.index,
                    'Risk_Score': final_prob,
                    'Alert_Status': ['High Risk' if x >= threshold else 'Stable' for x in final_prob]
                }).set_index('Patient_ID')

        st.success(f"Batch Analysis Complete: {len(ts_files)} patients.")
        st.dataframe(st.session_state.results_df.style.background_gradient(subset=['Risk_Score'], cmap='Reds'))
        st.download_button('Export Predictions CSV Report', st.session_state.results_df.to_csv().encode('utf-8'), 'icu_risk_report.csv')

with tab2:
    st.header("Individual Patient Analysis")
    
    if st.session_state.results_df is None:
        st.info("Please process a batch in Tab 1 first.")
    else:
        patient_list = st.session_state.results_df.index.tolist()
        selected_id = st.selectbox("Select Patient Record ID", options=patient_list)
        
        risk_score = st.session_state.results_df.loc[selected_id, 'Risk_Score']
        p_data = st.session_state.df_processed.loc[selected_id]

        col_r1, col_r2, col_r3 = st.columns([1, 1, 2])
        with col_r1:
            st.metric("Mortality Risk", f"{risk_score:.2%}")
        with col_r2:
            if risk_score >= threshold:
                st.error("🚨 HIGH RISK")
            elif risk_score >= (threshold * 0.7):
                st.warning("🟠 WATCHLIST")
            else:
                st.success("🟢 STABLE")
        with col_r3:
            st.write("**Ensemble Consensus**")
            st.progress(risk_score)

        st.divider()

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Shock Index", f"{p_data['Shock_Index']:.2f}", help="HR / SysABP. > 0.9 is concerning.")
        t2.metric("Anion Gap", f"{p_data['Anion_Gap']:.1f}", help="Metabolic acidosis indicator.")
        t3.metric("GCS (Min)", f"{p_data['GCS_Min']:.0f}/15")
        t4.metric("Urine (48h)", f"{p_data['Urine_Sum']:.0f} mL")

        st.divider()

        st.subheader("📋 Physiological Systems Overview")
        
        sys1, sys2 = st.columns(2)
        
        with sys1:
            st.write("**❤️ Cardiovascular & Respiratory**")
            cardio_df = pd.DataFrame({
                "Metric": ["HR Final", "MAP Min", "SysABP Mean", "PaO2/FiO2", "RespRate Mean"],
                "Value": [p_data['HR_Final'], p_data['UniMAP_Min'], p_data['UniSysABP_Mean'], 
                          p_data['PaO2_FiO2_Ratio'], p_data['RespRate_Mean']]
            })
            st.table(cardio_df)

        with sys2:
            st.write("**🧪 Metabolic & Renal**")
            renal_df = pd.DataFrame({
                "Metric": ["Creatinine Max", "BUN Initial", "Glucose Mean", "pH Mean", "ALP Initial"],
                "Value": [p_data['Creatinine_Max'], p_data['BUN_Initial'], p_data['Glucose_Mean'], 
                          p_data['pH_Mean'], p_data['ALP_Initial']]
            })
            st.table(renal_df)

        st.subheader("📉 Trend & Delta Analysis (Change over 48h)")
        
        delta_cols = st.columns(5)
        delta_features = [
            ('Creatinine', 'Creatinine_Delta'),
            ('BUN', 'BUN_Delta'),
            ('WBC', 'WBC_Delta'),
            ('HCT', 'HCT_Delta'),
            ('Bilirubin', 'Bilirubin_Delta')
        ]
        
        for i, (label, col) in enumerate(delta_features):
            val = p_data[col]
            delta_color = "inverse" if val > 0 else "normal" 
            delta_cols[i].metric(label, f"Δ {val:.2f}", delta=val, delta_color=delta_color)

        st.divider()
        
        with st.expander("🔍 View Data Quality & Gaps"):
            st.write(f"**Total Missing Features:** {p_data['Total_Missing_Sum']:.0f}")
            st.write(f"**Longest Data Gap:** {p_data['Global_Last_Gap']:.0f} minutes")
            st.write(f"**Neurological Observation Density (Log):** {p_data['GCS_Log_Count']:.2f}")