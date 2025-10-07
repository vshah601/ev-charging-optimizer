import numpy as np
import pandas as pd

class ScheduleAgent:
    def __init__(self, blackboard):
        self.bb = blackboard

    def generate_schedule(self):
        #Read in next day forecast datetimes and convert to numpy series
        forecast_times = self.bb.read("forecast_times")
        forecast_times = forecast_times.iloc[:, 0].to_numpy()  # shape (24,)

        #Create next day dataframe with datetime and forecast load value columns
        next_day = pd.DataFrame({
            "datetime": forecast_times,
            "forecast_mw": self.bb.read("forecast")
        })

        #Create new rolling average column that averages each four hour window
        next_day["rolling_avg"] = next_day["forecast_mw"].rolling(window=4).mean()

        #Drop NaN values
        next_day = next_day.dropna(subset=["rolling_avg"]).reset_index(drop=True)
        
        #Find the best four hour window
        min_idx = next_day["rolling_avg"].idxmin()
        best_window = next_day.loc[min_idx-3:min_idx+1]  # 4 consecutive hours

        #Write to blackboard
        self.bb.write("best_window", best_window)