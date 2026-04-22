import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from cleaning import run_icu_processing


BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR.parent 


@st.cache_resource
def load_bundle():
    boudle = joblib.load(path / 'icu_mortality_ensemble_v1.joblib')
    feature_name = joblib.load(path / 'feature_name.pkl')
    imputer = joblib.load(path / 'median_imputer.pkl')
    scaler = joblib.load(path / 'standard_scaler.pkl')
    return boudle, feature_name, imputer, scaler

bundle, feature_name, imputer, scaler = load_bundle()
feature_cols = bundle['feature_cols']


def run_ensemble_inference(df_linear, df_tree):
    df_linear_filtered = df_linear[feature_cols]
    df_tree_filtered = df_tree[feature_cols]    
    p_lr = bundle['cal_lr'].predict_proba(df_linear_filtered)[:, 1]
    p_lgb = bundle['cal_lgb'].predict_proba(df_tree_filtered)[:, 1]
    p_cb = bundle['cal_cb'].predict_proba(df_tree_filtered)[:, 1]    
    X_meta = np.column_stack([p_lr, p_lgb, p_cb])
    
    if hasattr(bundle['meta_learner'], 'predict_proba'):
        final_prob = bundle['meta_learner'].predict_proba(X_meta)[:, 1]
    else:
        final_prob = np.clip(bundle['meta_learner'].predict(X_meta), 0, 1)

    return final_prob


st.set_page_config(page_title='ICU Mortality Predictor', layout='wide')
st.title('🏥 ICU Mortality Ensemble Dashboard')


if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'df_combined' not in st.session_state:
    st.session_state.df_combined = None
if 'df_scaled' not in st.session_state:
    st.session_state.df_scaled = None
if 'df_results' not in st.session_state:
    st.session_state.df_results = None


st.sidebar.header('🔧 Model Configuration')
threshold = st.sidebar.slider(
    'Risk Threshold', 
    min_value=0.0, 
    max_value=1.0, 
    value=float(bundle['optimal_threshold']), 
    step=0.01,
    help='Adjust sensitivity. Lower values increase recall (fewer missed cases) but increase false alarms.')

st.sidebar.divider()
st.sidebar.subheader('Model Status')
st.sidebar.success('✅ Models Loaded')
if st.session_state.get('data_upload'):
    st.sidebar.success('✅ Environment Ready')
    st.sidebar.info(f'⚙️ Features Locked: {len(feature_cols)}')
else:
    st.sidebar.warning('⏳ Awaiting Data Input')


tab1, tab2 = st.tabs(['📂 Batch Processing', '🆔 Single Patient Analysis'])

with tab1:
    st.subheader('📑 Patient Data Processor')
    ts_files = st.file_uploader('Upload Patient Records (.txt)', type='txt', accept_multiple_files=True, key='data_upload')

    if ts_files:
        all_patients = []
        
        with st.spinner(f'Processing {len(ts_files)} records...'):
            for ts_file in ts_files:
                
                df = run_icu_processing(ts_file)

                # st.subheader(f'Raw Output from run_icu_processing ({ts_file.name})')
                # st.write(f'Initial Column Count: {len(df.columns)}')
                # st.dataframe(df) 
                # st.write('Initial Column Names:', list(df.columns))

                col_miss = [c for c in df.columns if c.endswith('_Miss')]
                df['Total_Missing_Sum'] = df[col_miss].sum(axis=1)
                
                col_count = [c for c in df.columns if c.endswith('_Count')]
                df['Total_Obs_Count'] = df[col_count].sum(axis=1)
                df[col_count] = df[col_count].apply(np.log1p)
                df = df.rename(columns={c: c.replace('_Count', '_Log_Count') for c in col_count})
                
                col_last_minute = [c for c in df.columns if c.endswith('_Last_Minute')]
                
                for col in col_last_minute:
                    gap_name = col.replace('_Last_Minute', '_Last_Gap')
                    df[gap_name] = 2880 - df[col]
                
                col_gap = [c for c in df.columns if c.endswith('_Last_Gap')]
                df['Global_Last_Gap'] = df[col_gap].min(axis=1)                
                df = df.drop(columns=col_last_minute)
                
                if 'ICUType' in df.columns:
                    current_val = df['ICUType'].iloc[0]
                else:
                    current_val = None
                
                icu_cols = ['ICUType_1.0', 'ICUType_2.0', 'ICUType_3.0', 'ICUType_4.0']
                icu_flags = pd.DataFrame([{col: (f'ICUType_{float(current_val)}' == col) for col in icu_cols}], index=df.index)
                df = pd.concat([df, icu_flags], axis=1)
                df[icu_cols] = df[icu_cols].astype(int)
                df = df.drop(columns=['ICUType'])
                
                df['Shock_Index'] = df['HR_Final'] / df['UniSysABP_Final']
                df['GCS_MAP_Product'] = df['GCS_Final'] * df['UniMAP_Final']
                df['Net_Cation_Balance'] = (df['Na_Final'] + df['K_Final']) - df['HCO3_Final']
                
                current_id = df['RecordID'].values[0] if 'RecordID' in df.columns else ts_file.name
                df.index = [current_id]
                df = df[feature_name]
                all_patients.append(df)

                # st.subheader(f'Output after processing')
                # st.write(f'After Column Count: {len(df.columns)}')
                # st.dataframe(df) 
                # st.write('Initial Column Names:', list(df.columns))

            X_combined = pd.concat(all_patients).replace([np.inf, -np.inf], np.nan)
            X_imputed = pd.DataFrame(imputer.transform(X_combined), columns=X_combined.columns, index=X_combined.index)
            X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
            
            # with st.expander('View Full Combined Dataset (All Patients)'):
            #     st.write(f'Total Patients: {len(X_combined)}')
            #     st.write(f'Total Columns: {X_combined.shape[1]}')
            #     st.dataframe(X_combined)

            st.session_state.df_combined = X_combined
            st.session_state.df_scaled = X_scaled
            final_prob = run_ensemble_inference(X_scaled[feature_cols], X_combined[feature_cols])
            st.session_state.df_results = pd.DataFrame({'Patient ID': X_combined.index, 'Risk Score': final_prob}).set_index('Patient ID')

    if st.session_state.df_results is not None:
        def get_proximity_status(score, t):
            if abs(score - t) <= 0.01: 
                return '⚠️ Borderline'
            elif score > t: 
                return '🚨 High Risk'
            else: 
                return '✅ Stable'

        st.session_state.df_results['Alert Status'] = [
            get_proximity_status(x, threshold) for x in st.session_state.df_results['Risk Score']
        ]
        
        st.success(f'Analysis Complete: {len(st.session_state.df_results)} patients processed.')
        
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader('Population Risk Summary')
            m1, m2 = st.columns(2)
            m1.metric('Total Patients', len(st.session_state.df_results))
            m2.metric('Alerts (>= Threshold)', (st.session_state.df_results['Risk Score'] >= threshold).sum())
            
            styled_df = st.session_state.df_results.style.background_gradient(
                subset=['Risk Score'], 
                cmap='Reds',
                vmin=threshold - 0.05,
                vmax=threshold + 0.20).format(precision=3)
            
            st.dataframe(styled_df, width='stretch')
        
        with c_right:
            st.subheader('Population Risk Distribution')
            
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
            fig_hist.patch.set_alpha(0)
            ax_hist.patch.set_alpha(0)

            ax_hist.hist(st.session_state.df_results['Risk Score'], bins=np.linspace(0, 1, 21), color='#00d4ff', alpha=0.9, edgecolor='none')
            ax_hist.axvline(threshold, color='#ff4b4b', linestyle='--', linewidth=1, label=f'Threshold: {threshold}')
            leg = ax_hist.legend(loc='upper right', frameon=False, fontsize=6)

            if leg:
                for text in leg.get_texts():
                    text.set_color("white")

            ax_hist.set_title('Shift slider to see alert changes', color='#FFFFFF', fontsize=6)
            ax_hist.tick_params(colors='#FFFFFF', labelsize=6)
            ax_hist.set_xlim(-0.01, 1.1)
            
            for spine in ax_hist.spines.values():
                spine.set_edgecolor('#555555')

            st.pyplot(fig_hist, transparent=True)


with tab2:
    st.subheader('👤 Patient-Centric Risk Profile')
    
    if st.session_state.df_results is None:
        st.info('Please process a batch in Tab 1 first.')
    else:

        st.session_state.df_results.index = st.session_state.df_results.index.astype(str)
        st.session_state.df_combined.index = st.session_state.df_combined.index.astype(str)

        # 1. Selection and Data Retrieval
        patient_options = st.session_state.df_results.index.tolist()
        selected_id = st.selectbox('Select Patient Record ID', options=patient_options)
        res = st.session_state.df_results.loc[selected_id]
        p_data = st.session_state.df_combined.loc[selected_id]

        # 2. Top-level Visual Risk Indicators
        col_gauge, col_metrics = st.columns([2, 1])

        with col_gauge:
            t_val = threshold * 100 
            p_risk = res['Risk Score'] * 100 
            
            fig_gauge = go.Figure(go.Indicator(
                mode = 'gauge+number',
                value = p_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {
                    'suffix': '%', 
                    'font': {'size': 50, 'color': 'white'}
                },
                gauge = {
                    'axis': {
                        'range': [None, 100], 
                        'tickwidth': 1, 
                        'ticksuffix': '%',
                        'tickcolor': 'white',
                        'tickfont': {'color': 'white'}
                    },
                    'bar': {'color': 'black', 'thickness': 0.15},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 1,
                    'bordercolor': 'white',
                    'steps': [
                        {'range': [0, (threshold - 0.01) * 100], 'color': '#2ecc71'},
                        {'range': [(threshold - 0.01) * 100, (threshold + 0.01) * 100], 'color': '#f1c40f'},
                        {'range': [(threshold + 0.01) * 100, 100], 'color': '#e74c3c'}
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 5},
                        'thickness': 0.8,
                        'value': t_val
                    }
                }
            ))

            fig_gauge.update_layout(
               title={
                   'text': f'Risk Assessment: Patient {selected_id}',
                   'font': {'size': 20, 'color': 'white'},
                   'y': 0.95,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'
               },
               height=450,
               margin=dict(l=30, r=30, t=100, b=20),
               paper_bgcolor='rgba(0,0,0,0)',
               font={'color': 'white', 'family': 'Arial'}
            )

            st.plotly_chart(fig_gauge, width='stretch')

        with col_metrics:
            status = res['Alert Status']
            st.write('Current Status')
            if 'High Risk' in status:
                st.error('🚨 High Mortality Risk Detected')
            elif 'Borderline' in status:
                st.warning('⚠️ Critical Proximity to Threshold')
            else:
                st.success('✅ Patient Currently Stable')
            
            dist = res['Risk Score'] - threshold
            st.metric('Threshold Delta', f'{dist:+.2%}', delta=dist, delta_color='inverse')

        st.divider()

        def fmt(val, precision=1, unit=''):
            if pd.isna(val):
                return '--'
            return f'{val:.{precision}f} {unit}'.strip()

        # --- 3. Physiology Breakdown (Organ System Surveillance) ---
        st.subheader('🩺 Organ System Surveillance')
        
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('🧠 Neuro/Resp')
            neuro_resp = pd.DataFrame({
                'Parameter': ['GCS Final', 'P/F Ratio', 'FiO2 Max', 'PaCO2 Max'],
                'Value': [
                    fmt(p_data.get('GCS_Final'), 0), 
                    fmt(p_data.get('PaO2_FiO2_Ratio'), 1), 
                    fmt(p_data.get('FiO2_Max', 0)*100, 0, '%'), 
                    fmt(p_data.get('PaCO2_Max'), 1, 'mmHg')
                ],
                'Reference': ['13-15', '> 300', '< 40%', '35-45 mmHg']
            })
            st.dataframe(neuro_resp, width='stretch', hide_index=True)

        with c2:
            st.markdown('❤️ Cardiovascular')
            cardio_data = pd.DataFrame({
                'Parameter': ['Shock Index', 'Lactate Max', 'MAP Min', 'Troponin Max'],
                'Value': [
                    fmt(p_data.get('Shock_Index'), 2), 
                    fmt(p_data.get('Lactate_Max'), 1, 'mmol/L'), 
                    fmt(p_data.get('UniMAP_Min'), 1, 'mmHg'), 
                    fmt(p_data.get('Troponin_Max'), 2, 'ng/mL')
                ],
                'Reference': ['0.5-0.7', '< 2.0', '> 65 mmHg', '< 0.04']
            })
            st.dataframe(cardio_data, width='stretch', hide_index=True)

        with c3:
            st.markdown('🧪 Renal/Metabolic')
            renal_data = pd.DataFrame({
                'Parameter': ['Creatinine Max', 'Bilirubin Init', 'Urine Mean', 'WBC Initial'],
                'Value': [
                    fmt(p_data.get('Creatinine_Max'), 2, 'mg/dL'), 
                    fmt(p_data.get('Bilirubin_Initial'), 1, 'mg/dL'), 
                    fmt(p_data.get('Urine_MeanRate'), 2, 'mL/kg/h'), 
                    fmt(p_data.get('WBC_Initial'), 1, 'k/uL')
                ],
                'Reference': ['0.7-1.3', '< 1.2', '> 0.5', '4.5-11.0']
            })
            st.dataframe(renal_data, width='stretch', hide_index=True)

        st.divider()

        # --- 4. Trend Visualization (Deltas) ---
        st.subheader('📈 Clinical Trajectory (48h Delta)')
        st.caption("Monitors change over 48h. '--' indicates no recording available.")
        
        d1, d2, d3, d4 = st.columns(4)
        
        def clinical_metric(col, label, key, higher_is_bad=True):
            val = p_data.get(key)
            if pd.isna(val):
                col.metric(label=label, value='--', delta=None)
            else:
                val = float(val)
                delta_str = f'{val:+.2f}'
                col.metric(label=label, value=f'{val:.2f}', delta=delta_str, delta_color='inverse' if higher_is_bad else 'normal')

        clinical_metric(d1, 'BUN Δ', 'BUN_Delta', higher_is_bad=True)
        clinical_metric(d2, 'WBC Δ', 'WBC_Delta', higher_is_bad=True)
        clinical_metric(d3, 'pH Slope', 'pH_Slope', higher_is_bad=False)
        clinical_metric(d4, 'Temp Slope', 'Temp_Slope', higher_is_bad=True)
            
        # 5. Raw Feature Inspection (Optional Expander)
        with st.expander('🔍 View All 100 Processed Features'):
            st.dataframe(p_data.to_frame().T)