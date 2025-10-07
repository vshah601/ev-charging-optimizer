import pandas as pd
import numpy as np

class AdvisorAgent:

    def __init__(self, blackboard):
        self.bb = blackboard

    def generate_report(self):
        window = self.bb.read("best_window")["datetime"]
        start_hour = window.iloc[0].hour
        end_hour = window.iloc[-1].hour

        average_load = np.mean(self.bb.read("best_window")["forecast_mw"])   

        best_model = self.bb.read("best_model")
        model_performance = self.bb.read("model_performance")

        explanation = (
        f"The optimal 4-hour charging window is from {start_hour}:00 to {end_hour}:00\n"
        f"The estimated average load in this period is {average_load}\n"
        f"The forecasting model selected for the predictions is {best_model} "
        f"because it had the best performance metric (MAPE: {model_performance[best_model]:.2f})\n"
        )

        return explanation