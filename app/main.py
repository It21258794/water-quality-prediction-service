from fastapi import FastAPI
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from collections import deque
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

def preprocess_data(df, targets=['PH', 'Conductivity', 'Turbidity'], sequence_length=5):
    dropped_columns = ['visibility', 'humidity', 'wind_speed', 'clouds', 'feels_like', 'pressure', 'dew_point', 'wind_deg']
    df = df.drop(columns=dropped_columns, errors='ignore')

    if 'rainfall' in df.columns:
        df['rainfall_binary'] = (df['rainfall'] > 0).astype(int)
        df = df.drop(columns=['rainfall'])

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.month
    df['Day'] = df['datetime'].dt.day
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['Hour'] = df['datetime'].dt.hour
    df['Minute'] = df['datetime'].dt.minute
    df = df.drop(columns=['datetime'])

    df = df.fillna(df.mean())

    if 'PH' in df.columns:
        df.loc[df['PH'] > 14, 'PH'] = df['PH'] / 10

    def remove_outliers_iqr(df, column, lower_percentile=0.25, upper_percentile=0.75):
        if column in df.columns:
            Q1, Q3 = df[column].quantile([lower_percentile, upper_percentile])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    df = remove_outliers_iqr(df, 'Turbidity', lower_percentile=0.05, upper_percentile=0.95)

    n_lags = 3
    for target in targets:
        for lag in range(1, n_lags + 1):
            df[f'{target}_lag{lag}'] = df[target].shift(lag)

    df.dropna(inplace=True)

    feature_columns = [col for col in df.columns if col not in targets]
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_scaled_features = pd.DataFrame(feature_scaler.fit_transform(df[feature_columns]), columns=feature_columns)
    df_scaled_targets = pd.DataFrame(target_scaler.fit_transform(df[targets]), columns=targets)

    df_scaled = pd.concat([df_scaled_features, df_scaled_targets], axis=1)

    def create_sequences(data, target_columns, sequence_length=5):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:i+sequence_length].values)
            y.append(data.iloc[i+sequence_length][target_columns].values)
        return np.array(X), np.array(y)

    X, y = create_sequences(df_scaled, targets, sequence_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_test, y_test, target_scaler


# Load LSTM model
lstm_model = keras.models.load_model('app/model/lag_lstm_water_quality.keras')

def get_future_dates(start_date, days_ahead):
    future_dates = [start_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return [date.strftime('%Y-%m-%d') for date in future_dates]




@app.post("/predict")
async def predict(data: dict, prediction_period: str, parameter: Optional[str] = Query(None, regex="^(PH|Conductivity|Turbidity)$")):
    print(parameter)
    df_center1 = pd.DataFrame(data['center1'])
    df_center2 = pd.DataFrame(data['center2'])
    df_center3 = pd.DataFrame(data['center3'])

    print(df_center1)
    print(df_center2)
    print(df_center3)

    X_test1, y_test1, scaler1 = preprocess_data(df_center1)
    X_test2, y_test2, scaler2 = preprocess_data(df_center2)
    X_test3, y_test3, scaler3 = preprocess_data(df_center3)

    current_input1 = X_test1[-1].copy()
    current_input2 = X_test2[-1].copy()
    current_input3 = X_test3[-1].copy()

    predictions = []
    prediction_periods = {'7': 7, '30': 30, '60': 60}
    days_ahead = prediction_periods.get(prediction_period)
    if not days_ahead:
        return {"error": "Invalid prediction period"}

    future_dates = get_future_dates(datetime.now(), days_ahead)

    for date in future_dates:
        daily_preds = {'date': date}
        daily_ph, daily_conduct, daily_turbidity = [], [], []

        for step in range(12):
            pred1 = lstm_model.predict(current_input1.reshape(1, -1, current_input1.shape[1]))
            pred2 = lstm_model.predict(current_input2.reshape(1, -1, current_input2.shape[1]))
            pred3 = lstm_model.predict(current_input3.reshape(1, -1, current_input3.shape[1]))

            pred1 = scaler1.inverse_transform(pred1).flatten()
            pred2 = scaler2.inverse_transform(pred2).flatten()
            pred3 = scaler3.inverse_transform(pred3).flatten()

            daily_ph.append(pred1[0])
            daily_conduct.append(pred1[1])
            daily_turbidity.append(pred1[2])

            current_input1 = np.roll(current_input1, shift=-1, axis=0)
            current_input2 = np.roll(current_input2, shift=-1, axis=0)
            current_input3 = np.roll(current_input3, shift=-1, axis=0)

            current_input1[-1, :3] = pred1
            current_input2[-1, :3] = pred2
            current_input3[-1, :3] = pred3

        if parameter is None or parameter == "PH":
            daily_preds['avgPh'] = float(np.mean(daily_ph))
        if parameter is None or parameter == "Conductivity":
            daily_preds['avgConductivity'] = float(np.mean(daily_conduct))
        if parameter is None or parameter == "Turbidity":
            daily_preds['avgTurbidity'] = float(np.mean(daily_turbidity))

        predictions.append(daily_preds)

    return {"predictions": predictions}