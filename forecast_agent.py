import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta

class ForecastAgent:
    def __init__(self, blackboard, hourly_load):   
        #Initialize agent properties
        self.bb = blackboard
        self.hourly_load = hourly_load
        self.models = {}
        self.model_performance = {}
        self.y_test = None
        self.X_test = None

    def train_models(self):    
        #Create new dataframes for train and test split
        X = self.hourly_load[['hour', 'day', 'month']]
        y = self.hourly_load['mw']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.y_test = y_test
        self.X_test = X_test

        #Train random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        #Train linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['linear_regression'] = lr_model

        #Train XGBRegressor model
        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model

    def evaluate_models(self):
        #For each model, calculate MAPE value
        for name, model in self.models.items():
            preds = model.predict(self.X_test)
            mape = np.mean(np.abs((self.y_test.values - preds) / self.y_test.values)) * 100
            self.model_performance[name] = mape

        #Store model with lowest MAPE value (best), into best_model and write to blackboard
        best_model = min(self.model_performance, key=self.model_performance.get)
        self.bb.write("best_model", best_model)
        self.bb.write("model_performance", self.model_performance)

    def forecast_next_24h(self):
        #Create new dataframe
        last_time = self.hourly_load["datetime"].iloc[-1]
        future_times = [last_time + timedelta(hours=i+1) for i in range(24)]
        future_df = pd.DataFrame({
            "hour": [t.hour for t in future_times],
            "day": [t.day for t in future_times],
            "month": [t.month for t in future_times],
        })
        future_datetimes = pd.DataFrame({"datetime": future_times})

        best_model = self.bb.read("best_model")
        preds = self.models[best_model].predict(future_df)

        self.bb.write("forecast", preds)
        self.bb.write("forecast_times", future_datetimes)