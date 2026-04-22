import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import copy
from pathlib import Path

from scipy.stats import linregress
from datetime import timedelta
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR.parent

def run_icu_processing(ts_file):

  df_raw = pd.read_csv(ts_file)

  general_params = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType']
  general_data = {}

  for param in general_params:
      val = df_raw.loc[df_raw['Parameter'] == param, 'Value'].values[0]
      general_data[param] = val

  df_ts = df_raw[~df_raw['Parameter'].isin(general_params)].copy()
  df_ts['RecordID'] = int(general_data['RecordID'])

  df_patient = pd.DataFrame([general_data])
  df_patient['RecordID'] = df_patient['RecordID'].astype(int)

  #print(df_ts)
  #print(df_patient)

  df_patient.loc[df_patient['Gender'] < 0, 'Gender'] = np.nan
  df_patient.loc[df_patient['Height'] < 10, 'Height'] = np.nan
  df_patient.loc[df_patient['Height'] > 300, 'Height'] = np.nan

  def string_to_minute(time_string):
      h, m = map(int, time_string.split(':'))
      return h * 60 + m

  df_ts['Time_Minute'] = df_ts['Time'].apply(string_to_minute)
  df_ts = df_ts.sort_values(['Parameter', 'Time', 'Time_Minute']).reset_index(drop=True)
  df_ts_col = ['RecordID', 'Parameter', 'Value', 'Time', 'Time_Minute']
  df_ts = df_ts[df_ts_col]

  if df_ts['Parameter'].isna().sum() != 0:
      df_ts = df_ts.dropna(subset=['Parameter'])

  #print(df_ts)

  value_dict_icu = [
      {'col': 'Albumin', 'value_lower': 0.5, 'value_upper': 7.0},
      {'col': 'ALP', 'value_lower': 5, 'value_upper': 5000},
      {'col': 'ALT', 'value_lower': 0, 'value_upper': 20000},
      {'col': 'AST', 'value_lower': 0, 'value_upper': 20000},
      {'col': 'Bilirubin', 'value_lower': 0.1, 'value_upper': 100.0},
      {'col': 'BUN', 'value_lower': 2, 'value_upper': 300},
      {'col': 'Cholesterol', 'value_lower': 10, 'value_upper': 800},
      {'col': 'Creatinine', 'value_lower': 0, 'value_upper': 30.0},
      {'col': 'DiasABP', 'value_lower': 15, 'value_upper': 150},
      {'col': 'FiO2', 'value_lower': 0.21, 'value_upper': 1.0},
      {'col': 'GCS', 'value_lower': 3, 'value_upper': 15},
      {'col': 'Glucose', 'value_lower': 20, 'value_upper': 1500},
      {'col': 'HCO3', 'value_lower': 5, 'value_upper': 60},
      {'col': 'HCT', 'value_lower': 10, 'value_upper': 70},
      {'col': 'HR', 'value_lower': 20, 'value_upper': 300},
      {'col': 'K', 'value_lower': 1.5, 'value_upper': 10.0},
      {'col': 'Lactate', 'value_lower': 0.1, 'value_upper': 35.0},
      {'col': 'MAP', 'value_lower': 10, 'value_upper': 170},
      {'col': 'MechVent', 'value_lower': 0, 'value_upper': 1},
      {'col': 'Mg', 'value_lower': 0.3, 'value_upper': 6.0},
      {'col': 'Na', 'value_lower': 100, 'value_upper': 170},
      {'col': 'NIDiasABP', 'value_lower': 15, 'value_upper': 150},
      {'col': 'NIMAP', 'value_lower': 20, 'value_upper': 170},
      {'col': 'NISysABP', 'value_lower': 40, 'value_upper': 250},
      {'col': 'PaCO2', 'value_lower': 10, 'value_upper': 100},
      {'col': 'PaO2', 'value_lower': 20, 'value_upper': 700},
      {'col': 'pH', 'value_lower': 6.8, 'value_upper': 7.8},
      {'col': 'Platelets', 'value_lower': 10, 'value_upper': 1500},
      {'col': 'RespRate', 'value_lower': 5, 'value_upper': 60},
      {'col': 'SaO2', 'value_lower': 50, 'value_upper': 100},
      {'col': 'SysABP', 'value_lower': 40, 'value_upper': 250},
      {'col': 'Temp', 'value_lower': 25.0, 'value_upper': 42.5},
      {'col': 'TroponinI', 'value_lower': 0, 'value_upper': 50},
      {'col': 'TroponinT', 'value_lower': 0, 'value_upper': 50},
      {'col': 'Urine', 'value_lower': 0, 'value_upper': 20000},
      {'col': 'WBC', 'value_lower': 0.5, 'value_upper': 800},
      {'col': 'Weight', 'value_lower': 20, 'value_upper': 500}]


  def outlier_to_missing(df, value_dict):
    df_new = copy.deepcopy(df)
    for i in range(len(value_dict)):
      col = value_dict[i]['col']
      value_lower = value_dict[i]['value_lower']
      value_upper = value_dict[i]['value_upper']
      if len(df_new[df_new['Parameter'] == col]) == 0:
        pass
      else:
        cond_1 = df_new['Parameter'] == col
        cond_2 = df_new['Value'] < value_lower
        cond_3 = df_new['Value'] > value_upper
        df_new.loc[cond_1 & cond_2, 'Value'] = np.nan
        df_new.loc[cond_1 & cond_3, 'Value'] = np.nan
    return df_new

  df_ts_clean = outlier_to_missing(df_ts, value_dict_icu) 


  col_icu = ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight']


  def nan_filler(df, col):
    dfs_processed = []

    for i in col:
      df_temp = copy.deepcopy(df[df['Parameter'] == i])
      if len(df_temp) == 0:
        continue
      elif len(df_temp) == 1:
        dfs_processed.append(df_temp)
      else:
        df_temp_fill = df_temp.ffill()
        dfs_processed.append(df_temp_fill)

    df_fillna = pd.concat(dfs_processed, ignore_index=True)
    return df_fillna.reset_index(drop=True)


  df_ts_fill = nan_filler(df_ts, col_icu)


  col_stats = ['Albumin_Initial', 'Albumin_Min', 'Albumin_Max', 'Albumin_Mean', 'ALP_Initial', 'ALP_Max', 'ALT_Initial', 'ALT_Max', 'AST_Initial', 'AST_Max',
  'Bilirubin_Initial', 'Bilirubin_Max', 'BUN_Initial', 'BUN_Max', 'BUN_Mean', 'BUN_Creatinine_Ratio', 'Cholesterol_Initial', 'Cholesterol_Min',
  'Creatinine_Initial', 'Creatinine_Max', 'Creatinine_Mean', 'FiO2_Initial', 'FiO2_Max', 'FiO2_Mean',
  'GCS_Initial', 'GCS_Final', 'GCS_Min', 'GCS_Max', 'GCS_Mean', 'Glucose_Initial', 'Glucose_Min', 'Glucose_Max', 'Glucose_Mean',
  'HCO3_Initial', 'HCO3_Final', 'HCO3_Min', 'HCO3_Max', 'HCT_Initial', 'HCT_Min', 'HCT_Max', 'HCT_Mean',
  'HR_Initial', 'HR_Final', 'HR_Min', 'HR_Max', 'HR_Mean', 'HR_SD', 'K_Initial', 'K_Final', 'K_Min', 'K_Max', 'K_Mean',
  'Lactate_Initial', 'Lactate_Max', 'Lactate_Clearance_Rate_6H', 'Lactate_Clearance_Rate_12H', 'MechVent_Any',
  'Mg_Initial', 'Mg_Min', 'Mg_Max', 'Na_Initial', 'Na_Final', 'Na_Min', 'Na_Max', 'Na_Mean',
  'PaCO2_Initial', 'PaCO2_Min', 'PaCO2_Max', 'PaO2_Initial', 'PaO2_Min', 'PaO2_FiO2_Ratio',
  'pH_Initial', 'pH_Min', 'pH_Max', 'pH_Mean', 'Platelets_Initial', 'Platelets_Min',
  'RespRate_Initial', 'RespRate_Max', 'RespRate_Mean', 'SaO2_Initial', 'SaO2_Min',
  'Temp_Initial', 'Temp_Min', 'Temp_Max', 'Temp_Mean', 'Temp_SD',
  'Urine_Sum', 'Urine_MinRate', 'Urine_MaxRate', 'Urine_MeanRate',
  'WBC_Initial', 'WBC_Min', 'WBC_Max', 'Weight_Initial', 'Weight_Final',
  'UniDiasABP_Initial', 'UniDiasABP_Min', 'UniDiasABP_Max', 'UniDiasABP_Mean',
  'UniMAP_Initial', 'UniMAP_Final', 'UniMAP_Min', 'UniMAP_Max', 'UniMAP_Mean', 'UniMAP_SD',
  'UniSysABP_Initial', 'UniSysABP_Final', 'UniSysABP_Min', 'UniSysABP_Max', 'UniSysABP_Mean', 'UniSysABP_SD', 'Troponin_Max']

  col_count = ['Albumin_Count', 'ALP_Count', 'ALT_Count', 'AST_Count', 'BUN_Count', 'Bilirubin_Count',
  'Cholesterol_Count', 'Creatinine_Count', 'FiO2_Count', 'GCS_Count', 'Glucose_Count',
  'HCO3_Count', 'HCT_Count', 'HR_Count', 'K_Count', 'Lactate_Count', 'MechVent_Count',
  'Mg_Count', 'Na_Count', 'PaCO2_Count', 'PaO2_Count', 'pH_Count', 'Platelets_Count',
  'RespRate_Count', 'SaO2_Count', 'Temp_Count', 'Urine_Count', 'WBC_Count',
  'UniDiasABP_Count', 'UniMAP_Count', 'UniSysABP_Count', 'Troponin_Count']

  col_delta = ['Albumin_Delta', 'ALP_Delta', 'ALT_Delta', 'AST_Delta', 'Bilirubin_Delta', 'BUN_Delta', 'Cholesterol_Delta', 'Creatinine_Delta',
  'HCO3_Delta', 'HCT_Delta', 'K_Delta', 'Mg_Delta', 'Na_Delta', 'PaO2_Delta', 'Platelets_Delta', 'SaO2_Delta', 'WBC_Delta', 'Weight_Delta', 'Troponin_Delta']

  col_slope = ['FiO2_Slope', 'GCS_Slope', 'Glucose_Slope', 'HR_Slope', 'PaCO2_Slope', 'pH_Slope', 'RespRate_Slope', 'Temp_Slope', 'Urine_Rate_Slope', 'UniDiasABP_Slope', 'UniMAP_Slope', 'UniSysABP_Slope']

  col_miss = ['Albumin_Miss', 'ALP_Miss', 'ALT_Miss', 'AST_Miss', 'Bilirubin_Miss', 'BUN_Miss', 'Cholesterol_Miss', 'Creatinine_Miss',
  'FiO2_Miss', 'GCS_Miss', 'Glucose_Miss', 'HCO3_Miss', 'HCT_Miss', 'HR_Miss', 'K_Miss', 'Lactate_Miss', 'MechVent_Miss', 'Mg_Miss', 'Na_Miss',
  'PaCO2_Miss', 'PaO2_Miss', 'pH_Miss', 'Platelets_Miss', 'RespRate_Miss', 'SaO2_Miss', 'Temp_Miss', 'Urine_Miss', 'WBC_Miss', 'Weight_Miss',
  'UniDiasABP_Miss', 'UniMAP_Miss', 'UniSysABP_Miss', 'Troponin_Miss']

  col_last_minute = ['Albumin_Last_Minute', 'ALP_Last_Minute', 'ALT_Last_Minute', 'AST_Last_Minute', 'Bilirubin_Last_Minute', 'BUN_Last_Minute', 'Cholesterol_Last_Minute', 'Creatinine_Last_Minute',
  'FiO2_Last_Minute', 'GCS_Last_Minute', 'Glucose_Last_Minute', 'HCO3_Last_Minute', 'HCT_Last_Minute', 'HR_Last_Minute', 'K_Last_Minute', 'Lactate_Last_Minute',
  'MechVent_Last_Minute', 'Mg_Last_Minute', 'Na_Last_Minute', 'PaCO2_Last_Minute', 'PaO2_Last_Minute', 'pH_Last_Minute', 'Platelets_Last_Minute', 'RespRate_Last_Minute',
  'SaO2_Last_Minute', 'Temp_Last_Minute', 'Urine_Last_Minute', 'WBC_Last_Minute', 'Weight_Last_Minute', 'UniDiasABP_Last_Minute', 'UniMAP_Last_Minute', 'UniSysABP_Last_Minute', 'Troponin_Last_Minute']

  col_df = ['RecordID'] + col_stats + col_count + col_delta + col_slope + col_miss + col_last_minute



  def find_max_duration(df_data, threshold, condition_type='above'):

    if condition_type.lower() not in ['above', 'below']:
      raise ValueError('condition_type must be "above" or "below".')
    
    max_duration = 0
    max_begin = -1
    max_end = -1

    current_duration = 0
    current_begin = -1

    for i, value in enumerate(df_data['Value']):

      is_condition_met = False
      if condition_type.lower() == 'above':
        is_condition_met = (value >= threshold)
      elif condition_type.lower() == 'below':
        is_condition_met = (value <= threshold)

      if is_condition_met:
        if current_duration == 0:
          current_begin = i
        current_duration += 1
      else:
        if current_duration > max_duration:
          max_duration = current_duration
          max_begin = current_begin
          max_end = i - 1

        current_duration = 0
        current_begin = -1

    if current_duration > max_duration:
      max_duration = current_duration
      max_begin = current_begin
      max_end = len(df_data['Value']) - 1

    return max_duration, max_begin, max_end



  def find_time_duration(df_data, max_begin, max_end):
    time_begin = df_data['Time_Minute'].iloc[max_begin]
    time_end = df_data['Time_Minute'].iloc[max_end]
    time_duration = time_end - time_begin
    return time_duration, time_begin, time_end



  def find_max_bcr(df_sub, df_sub2):

    if len(df_sub2) == 0:
      return np.nan
    elif len(df_sub2) == 1:
      if df_sub2.isnull().sum().sum() != 0:
        return np.nan
      else:
        if df_sub2['Value'].iloc[0] == 0:
          return np.nan
        bcr = df_sub['Value'].iloc[0]/df_sub2['Value'].iloc[0]
        return bcr
    else:

      def aggregator(series):
        if (series == 0).any():
          return series.max()
        else:
          return series.mean()
      bun_series = df_sub.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': aggregator}).set_index('Time_Minute').sort_index()
      cr_series = df_sub2.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': aggregator}).set_index('Time_Minute').sort_index()

      combined_index = bun_series.index.union(cr_series.index)

      bun_aligned = bun_series.reindex(combined_index)
      cr_aligned = cr_series.reindex(combined_index)

      bun_imputed = bun_aligned.ffill()
      cr_imputed = cr_aligned.ffill()

      bun_imputed = bun_imputed.bfill()
      cr_imputed = cr_imputed.bfill()

      bcr_ratio_series = pd.Series(np.where(cr_imputed['Value'] == 0, np.nan, bun_imputed['Value']/cr_imputed['Value']))
      bcr = bcr_ratio_series.max()
      return bcr



  def find_lcr(df_sub, hour):
    TIME_INITIAL = df_sub['Time_Minute'].iloc[0]
    TARGET_TIME = hour*60

    initial_value = df_sub['Value'].iloc[0]


    if initial_value == 0:
      lcr = np.nan
    elif len(df_sub['Time_Minute'][df_sub['Time_Minute'] > TARGET_TIME]) == 0:
      lcr = np.nan
    else:
      target_index = df_sub['Time_Minute'][df_sub['Time_Minute'] > TARGET_TIME].idxmin()
      final_value = df_sub.loc[target_index]['Value']
      lcr = (initial_value-final_value)/initial_value*100

    return lcr



  def find_min_pfr(df_sub, df_sub2):

    if len(df_sub2) == 0:
      return np.nan
    elif len(df_sub2) == 1:
      if df_sub2.isnull().sum().sum() != 0:
        return np.nan
      else:
        if df_sub2['Value'].iloc[0] == 0:
          return np.nan
        pfr = df_sub['Value'].iloc[0]/df_sub2['Value'].iloc[0]
        return pfr
    else:

      def aggregator(series):
        if (series == 0).any():
          return series.max()
        else:
          return series.mean()

      pao2_series = df_sub.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': aggregator}).set_index('Time_Minute').sort_index()
      fio2_series = df_sub2.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': aggregator}).set_index('Time_Minute').sort_index()

      combined_index = pao2_series.index.union(fio2_series.index)

      pao2_aligned = pao2_series.reindex(combined_index)
      fio2_aligned = fio2_series.reindex(combined_index)

      pao2_imputed = pao2_aligned.ffill()
      fio2_imputed = fio2_aligned.ffill()

      pao2_imputed = pao2_imputed.bfill()
      fio2_imputed = fio2_imputed.bfill()

      pfr_ratio_series = pd.Series(np.where(fio2_imputed['Value'] == 0, np.nan, pao2_imputed['Value']/fio2_imputed['Value']))
      pfr = pfr_ratio_series.min()

      return pfr



  def compute_urine_rate_slope(df_sub):

      if len(df_sub) < 2:
          return 0.0

      df_sub = df_sub.sort_values('Time_Minute').reset_index(drop=True)

      times_hrs = df_sub['Time_Minute'].values / 60.0
      volumes = df_sub['Value'].values
      time_deltas = np.diff(times_hrs, prepend=0)
      time_deltas = np.where(time_deltas == 0, 0.016, time_deltas)
      rates = volumes / time_deltas
      slope, intercept, r_val, p_val, std_err = linregress(times_hrs, rates)

      return slope



  def find_uminr(df_sub, df_sub2):
    TIME_STABILITY_THRESHOLD = 0.1

    if len(df_sub2) == 0:
      weight_mean = np.nan
      return np.nan
    elif len(df_sub2) == 1:
      if df_sub2.isnull().sum().sum() != 0:
        weight_mean = np.nan
        return np.nan
      else:
        weight_mean = df_sub2['Value'].iloc[0]
    else:
      weight_mean = df_sub2['Value'].mean()

    urine_delta = df_sub.groupby('Time_Minute')['Value'].max()
    time_delta = df_sub.groupby('Time_Minute')['Time_Minute'].max().diff()/60
    umr_series = pd.Series(np.where((time_delta > TIME_STABILITY_THRESHOLD) & (weight_mean > 0), urine_delta/(weight_mean*time_delta), np.nan))
    uminr = umr_series.min()
    return uminr



  def find_umaxr(df_sub, df_sub2):
    TIME_STABILITY_THRESHOLD = 0.1

    if len(df_sub2) == 0:
      weight_mean = np.nan
      return np.nan
    elif len(df_sub2) == 1:
      if df_sub2.isnull().sum().sum() != 0:
        weight_mean = np.nan
        return np.nan
      else:
        weight_mean = df_sub2['Value'].iloc[0]
    else:
      weight_mean = df_sub2['Value'].mean()

    urine_delta = df_sub.groupby('Time_Minute')['Value'].max()
    time_delta = df_sub.groupby('Time_Minute')['Time_Minute'].max().diff()/60
    umr_series = pd.Series(np.where((time_delta > TIME_STABILITY_THRESHOLD) & (weight_mean > 0), urine_delta/(weight_mean*time_delta), np.nan))
    umaxr = umr_series.max()
    return umaxr



  def find_umeanr(df_sub, df_sub2):
    TIME_STABILITY_THRESHOLD = 0.1

    if len(df_sub2) == 0:
      weight_mean = np.nan
      return np.nan
    elif len(df_sub2) == 1:
      if df_sub2.isnull().sum().sum() != 0:
        weight_mean = np.nan
        return np.nan
      else:
        weight_mean = df_sub2['Value'].iloc[0]
    else:
      weight_mean = df_sub2['Value'].mean()

    urine_delta = df_sub.groupby('Time_Minute')['Value'].max()
    time_delta = df_sub.groupby('Time_Minute')['Time_Minute'].max().diff()/60
    umr_series = pd.Series(np.where((time_delta > TIME_STABILITY_THRESHOLD) & (weight_mean > 0), urine_delta/(weight_mean*time_delta), np.nan))
    umeanr = umr_series.mean()
    return umeanr



  def minutes_to_time_string(total_minutes):

    if pd.isna(total_minutes) or total_minutes is None or total_minutes < 0:
      return np.nan

    duration = timedelta(minutes=int(total_minutes))
    total_seconds = duration.total_seconds()
    total_hours = math.floor(total_seconds / 3600)
    remaining_seconds = total_seconds % 3600
    minutes = math.floor(remaining_seconds / 60)
    time_string = f"{int(total_hours):02d}:{int(minutes):02d}"

    return time_string



  def find_uniabp(df_sub, df_sub2):

    i_series = df_sub.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': 'min'}).set_index('Time_Minute').sort_index()
    ni_series = df_sub2.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': 'min'}).set_index('Time_Minute').sort_index()

    match_index = i_series.index.intersection(ni_series.index)

    if len(match_index) == 0:
      delta = 0
    else:
      i_series_match = i_series.reindex(match_index)
      ni_series_match = ni_series.reindex(match_index)
      delta = 0

    uni_index = i_series.index.union(ni_series.index)

    uni_series_preindex = i_series.reindex(uni_index).fillna(ni_series)
    uni_series = uni_series_preindex.reset_index()
    df_uni = uni_series.rename(columns = {'index':'Time_Minute', 'value':'Value'})

    recordid_series = pd.Series(data=np.repeat(df_sub['RecordID'][0], len(df_uni)), name='RecordID').reset_index(drop=True)
    parameter_series = pd.Series(data=np.repeat('Uni'+df_sub['Parameter'][0], len(df_uni)), name='Parameter').reset_index(drop=True)
    value_series = round(df_uni['Value'], 2)
    time_minute_series = df_uni['Time_Minute']
    time_series = time_minute_series.apply(minutes_to_time_string)
    df_result = pd.concat([recordid_series, parameter_series, value_series, time_series, time_minute_series], axis=1)
    df_result.columns = ['RecordID', 'Parameter', 'Value', 'Time', 'Time_Minute']

    return df_result



  def find_unidf(df_sub, df_sub2):
    if (len(df_sub) == 0) and (len(df_sub2) == 0):
      columns_list = ['RecordID', 'Parameter', 'Value', 'Time', 'Time_Minute']
      df_result = pd.DataFrame(columns=columns_list)
    elif len(df_sub) == 0:
      df_sub2['Parameter'] = 'Uni'+ df_sub2['Parameter'].str.replace('^NI', '', regex=True)
      df_result = df_sub2
    elif len(df_sub2) == 0:
      df_sub['Parameter'] = 'Uni'+df_sub['Parameter']
      df_result = df_sub
    else:
      df_result = find_uniabp(df_sub, df_sub2)

    return df_result



  def mad_stats(df1, df2):
    combined_index = df1.index.union(df2.index)

    df1_imputed = df1.reindex(combined_index).ffill().bfill()
    df2_imputed = df2.reindex(combined_index).ffill().bfill()

    mad = (df1_imputed['Value'] - df2_imputed['Value']).abs().mean()

    return mad



  def find_mad(df1, df2):
    if (len(df1) == 0) and (len(df2) == 0):
      return 0
    elif len(df1) < 2:
      return 0
    elif len(df2) < 2:
      return 0
    else:
      return mad_stats(df1, df2)



  def compute_end_to_end_slope(df_sun):

    clean_data = df_sub.dropna(subset=['Value'])

    if len(clean_data) < 2:
      return 0.0

    v_initial = clean_data['Value'].iloc[0]
    v_final = clean_data['Value'].iloc[-1]

    t_initial = clean_data['Time_Minute'].iloc[0]
    t_final = clean_data['Time_Minute'].iloc[-1]
    t_diff = t_final - t_initial

    if t_diff == 0.0:
      return 0.0
    else:
      slope = (v_final - v_initial)/t_diff
      return slope



  def compute_regression_slope(df_sub):
    clean_data = df_sub.dropna(subset=['Value'])

    if len(clean_data) < 2 or np.ptp(clean_data['Time_Minute']) == 0:
      return 0.0

    X = clean_data['Time_Minute'].values
    y = clean_data['Value'].values

    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    return slope



  def combine_troponin(df_sub, df_sub2):
    i_series = df_sub.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': 'max'}).set_index('Time_Minute').sort_index()
    t_series = df_sub2.groupby('Time_Minute').agg({'Time_Minute': 'min', 'Value': 'max'}).set_index('Time_Minute').sort_index()

    uni_index = i_series.index.union(t_series.index)

    uni_series_preindex = i_series.reindex(uni_index).fillna(t_series)
    uni_series = uni_series_preindex.reset_index()
    df_uni = uni_series.rename(columns = {'index':'Time_Minute', 'value':'Value'})

    recordid_series = pd.Series(data=np.repeat(df_sub['RecordID'][0], len(df_uni)), name='RecordID').reset_index(drop=True)
    parameter_series = pd.Series(data=np.repeat('Troponin', len(df_uni)), name='Parameter').reset_index(drop=True)
    value_series = df_uni['Value']
    time_minute_series = df_uni['Time_Minute']
    time_series = time_minute_series.apply(minutes_to_time_string)
    df_result = pd.concat([recordid_series, parameter_series, value_series, time_series, time_minute_series], axis=1)
    df_result.columns = ['RecordID', 'Parameter', 'Value', 'Time', 'Time_Minute']

    return df_result



  def find_troponin(df_sub, df_sub2):
    if (len(df_sub) == 0) and (len(df_sub2) == 0):
      columns_list = ['RecordID', 'Parameter', 'Value', 'Time', 'Time_Minute']
      return pd.DataFrame(columns=columns_list)
    elif len(df_sub) == 0:
      df_sub2['Parameter'] = 'Troponin'
      return df_sub2
    elif len(df_sub2) == 0:
      df_sub['Parameter'] = 'Troponin'
      return df_sub
    else:
      df_result = combine_troponin(df_sub, df_sub2)
      return df_result



  def albumin_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[ col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[ col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def alp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def alt_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def ast_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def bilirubin_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def bun_stats(df_sub, df_sub2, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_bcr = col+'_Creatinine_Ratio'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_bcr] = np.nan
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_bcr] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_bcr] = find_max_bcr(df_sub, df_sub2)
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_bcr] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_bcr] = find_max_bcr(df_sub, df_sub2)
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def cholesterol_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def creatinine_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def diasabp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def fio2_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def gcs_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def glucose_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record


  def hco3_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def hct_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def hr_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def k_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'

    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def lactate_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_miss = col+'_Miss'
    col_cr6 = col+'_Clearance_Rate_6H'
    col_cr12 = col+'_Clearance_Rate_12H'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_miss] = 1
      df_record[col_cr6] = np.nan
      df_record[col_cr12] = np.nan
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_miss] = 1
        df_record[col_cr6] = np.nan
        df_record[col_cr12] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_miss] = 0
        df_record[col_cr6] = np.nan
        df_record[col_cr12] = np.nan
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_miss] = 1
        df_record[col_cr6] = np.nan
        df_record[col_cr12] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_miss] = 0
        df_record[col_cr6] = find_lcr(df_sub, 6)
        df_record[col_cr12] = find_lcr(df_sub, 12)
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def map_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def mechvent_stats(df_sub, df_record, col):
    col_any = col+'_Any'
    col_count = col+'_Count'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_any] = np.nan
      df_record[col_count] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_any] = np.nan
        df_record[col_count] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_any] = (df_sub['Value'].iloc[0] == 1).any().astype(float)
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_any] = np.nan
        df_record[col_count] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_any] = (df_sub['Value'] == 1).any().astype(float)
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def mg_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def na_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def nidiasabp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def nimap_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'

    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record


  def nisysabp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def paco2_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def pao2_stats(df_sub, df_sub2, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_pfr = col+'_FiO2_Ratio'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_pfr] = np.nan
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_pfr] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_pfr] = find_min_pfr(df_sub, df_sub2)
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_pfr] = np.nan
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_pfr] = find_min_pfr(df_sub, df_sub2)
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def ph_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def platelets_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def resprate_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def sao2_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def sysabp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'

    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record


  def temp_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def troponini_stats(df_sub, df_record, col):
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_end_to_end_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_end_to_end_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def troponint_stats(df_sub, df_record, col):
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_end_to_end_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_end_to_end_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def urine_stats(df_sub, df_sub2, df_record, col):
    col_sum = col+'_Sum'
    col_minr = col+'_MinRate'
    col_maxr = col+'_MaxRate'
    col_meanr = col+'_MeanRate'
    col_count = col+'_Count'
    col_slope = col+'_Rate_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_sum] = np.nan
      df_record[col_minr] = np.nan
      df_record[col_maxr] = np.nan
      df_record[col_meanr] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_sum] = np.nan
        df_record[col_minr] = np.nan
        df_record[col_maxr] = np.nan
        df_record[col_meanr] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_sum] = df_sub['Value'].sum()
        df_record[col_minr] = find_uminr(df_sub, df_sub2)
        df_record[col_maxr] = find_umaxr(df_sub, df_sub2)
        df_record[col_meanr] = find_umeanr(df_sub, df_sub2)
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_urine_rate_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_sum] = np.nan
        df_record[col_minr] = np.nan
        df_record[col_maxr] = np.nan
        df_record[col_meanr] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_sum] = df_sub['Value'].sum()
        df_record[col_minr] = find_uminr(df_sub, df_sub2)
        df_record[col_maxr] = find_umaxr(df_sub, df_sub2)
        df_record[col_meanr] = find_umeanr(df_sub, df_sub2)
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_urine_rate_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def wbc_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  def weight_stats(df_sub, df_record, col):
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record

  def unidiasabp_stats(df_sub, df_record):
    col = 'UniDiasABP'
    col_initial = col+'_Initial'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        #df_record[col_td] = time_duration
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record

  def unimap_stats(df_sub, df_record):
    col = 'UniMAP'
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record


  def unisysabp_stats(df_sub, df_record):
    col = 'UniSysABP'
    col_initial = col+'_Initial'
    col_final = col+'_Final'
    col_min = col+'_Min'
    col_max = col+'_Max'
    col_mean = col+'_Mean'
    col_sd = col+'_SD'
    col_count = col+'_Count'
    col_slope = col+'_Slope'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_initial] = np.nan
      df_record[col_final] = np.nan
      df_record[col_min] = np.nan
      df_record[col_max] = np.nan
      df_record[col_mean] = np.nan
      df_record[col_sd] = np.nan
      df_record[col_count] = 0
      df_record[col_slope] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = 0
        df_record[col_count] = 1
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_initial] = np.nan
        df_record[col_final] = np.nan
        df_record[col_min] = np.nan
        df_record[col_max] = np.nan
        df_record[col_mean] = np.nan
        df_record[col_sd] = np.nan
        df_record[col_count] = 0
        df_record[col_slope] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_initial] = df_sub['Value'].iloc[0]
        df_record[col_final] = df_sub['Value'].iloc[-1]
        df_record[col_min] = df_sub['Value'].min()
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_mean] = df_sub['Value'].mean()
        df_record[col_sd] = df_sub['Value'].std()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_slope] = compute_regression_slope(df_sub)
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record


  def troponin_stats(df_sub, df_record):
    col = 'Troponin'
    col_max = col+'_Max'
    col_count = col+'_Count'
    col_delta = col+'_Delta'
    col_miss = col+'_Miss'
    col_last_minute = col+'_Last_Minute'
    if len(df_sub) == 0:
      df_record[col_max] = np.nan
      df_record[col_count] = 0
      df_record[col_delta] = 0
      df_record[col_miss] = 1
      df_record[col_last_minute] = 0
    elif len(df_sub) == 1:
      if df_sub.isnull().sum().sum() != 0:
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    else:
      if df_sub.isnull().sum().sum() == len(df_sub):
        df_record[col_max] = np.nan
        df_record[col_count] = 0
        df_record[col_delta] = 0
        df_record[col_miss] = 1
        df_record[col_last_minute] = 0
      else:
        df_record[col_max] = df_sub['Value'].max()
        df_record[col_count] = df_sub['Value'].count()
        df_record[col_delta] = df_sub['Value'].iloc[-1] - df_sub['Value'].iloc[0]
        df_record[col_miss] = 0
        df_record[col_last_minute] = df_sub['Time_Minute'].iloc[-1]
    return df_record



  df_temp = df_ts_fill
  df_record = pd.DataFrame(columns=col_df)
  df_record.loc[0] = np.nan
  df_record['RecordID'] = df_temp['RecordID'].iloc[0]

  for col in col_icu:
    if col == 'Albumin':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      albumin_stats(df_sub, df_record, col)
    elif col == 'ALP':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      alp_stats(df_sub, df_record, col)
    elif col == 'ALT':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      alt_stats(df_sub, df_record, col)
    elif col == 'AST':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      ast_stats(df_sub, df_record, col)
    elif col == 'Bilirubin':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      bilirubin_stats(df_sub, df_record, col)
    elif col == 'BUN':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df_sub2 = df_temp[df_temp['Parameter'] == 'Creatinine'].reset_index(drop=True)
      bun_stats(df_sub, df_sub2, df_record, col)
    elif col == 'Cholesterol':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      cholesterol_stats(df_sub, df_record, col)
    elif col == 'Creatinine':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      creatinine_stats(df_sub, df_record, col)
    elif col == 'DiasABP':
      col2 = 'NI'+col
      df1 = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df2 = df_temp[df_temp['Parameter'] == col2].reset_index(drop=True)
      df_sub = find_unidf(df1, df2)
      unidiasabp_stats(df_sub, df_record)
    elif col == 'FiO2':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      fio2_stats(df_sub, df_record, col)
    elif col == 'GCS':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      gcs_stats(df_sub, df_record, col)
    elif col == 'Glucose':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      glucose_stats(df_sub, df_record, col)
    elif col == 'HCO3':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      hco3_stats(df_sub, df_record, col)
    elif col == 'HCT':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      hct_stats(df_sub, df_record, col)
    elif col == 'HR':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      hr_stats(df_sub, df_record, col)
    elif col == 'K':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      k_stats(df_sub, df_record, col)
    elif col == 'Lactate':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      lactate_stats(df_sub, df_record, col)
    elif col == 'Mg':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      mg_stats(df_sub, df_record, col)
    elif col == 'MAP':
      col2 = 'NI'+col
      df1 = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df2 = df_temp[df_temp['Parameter'] == col2].reset_index(drop=True)
      df_sub = find_unidf(df1, df2)
      unimap_stats(df_sub, df_record)
    elif col == 'MechVent':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      mechvent_stats(df_sub,df_record, col)
    elif col == 'Na':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      na_stats(df_sub,df_record, col)
    elif col == 'NIDiasABP':
      pass
    elif col == 'NIMAP':
      pass
    elif col == 'NISysABP':
      pass
    elif col == 'PaCO2':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      paco2_stats(df_sub, df_record, col)
    elif col == 'PaO2':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df_sub2 = df_temp[df_temp['Parameter'] == 'FiO2'].reset_index(drop=True)
      pao2_stats(df_sub, df_sub2, df_record, col)
    elif col == 'pH':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      ph_stats(df_sub, df_record, col)
    elif col == 'Platelets':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      platelets_stats(df_sub, df_record, col)
    elif col == 'RespRate':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      resprate_stats(df_sub, df_record, col)
    elif col == 'SaO2':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      sao2_stats(df_sub, df_record, col)
    elif col == 'SysABP':
      col2 = 'NI'+col
      df1 = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df2 = df_temp[df_temp['Parameter'] == col2].reset_index(drop=True)
      df_sub = find_unidf(df1, df2)
      unisysabp_stats(df_sub, df_record)
    elif col == 'Temp':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      temp_stats(df_sub, df_record, col)
    elif col == 'TroponinI':
      df1 = df_temp[df_temp['Parameter'] == col].reset_index(drop=True) 
      df2 = df_temp[df_temp['Parameter'] == 'TroponinT'].reset_index(drop=True)
      df_sub = find_troponin(df1, df2)
      troponin_stats(df_sub, df_record)
    elif col == 'TroponinT':
      pass
    elif col == 'Urine':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      df_sub2 = df_temp[df_temp['Parameter'] == 'Weight'].reset_index(drop=True)
      urine_stats(df_sub, df_sub2, df_record, col)
    elif col == 'WBC':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      wbc_stats(df_sub, df_record, col)
    elif col == 'Weight':
      df_sub = df_temp[df_temp['Parameter'] == col].reset_index(drop=True)
      weight_stats(df_sub, df_record, col)


  df_record['RecordID'] = df_record['RecordID'].astype(int)
  df_combined = pd.merge(left=df_record, right=df_patient, on='RecordID', how='inner')
  #df_combined.to_csv(path / 'df_new.csv', index=False)
  return df_combined

