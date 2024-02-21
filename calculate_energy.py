import numpy as np
import pandas as pd

def calculate_energy_levels(trajectory_np_lambdas_arr, energy_np_lambda, var=1, steps=6000):
    x = np.vstack([f(np.arange(steps)/1000) for f in trajectory_np_lambdas_arr])
    energy = energy_np_lambda(x, 1/var/1000)
    energy_diff = np.max(energy) - np.min(energy)    
    ground_level = -(np.max(energy) + .1 * energy_diff)
    high_level = np.max(energy) + ground_level 
    low_level = np.min(energy) + ground_level
    lowest_level = low_level - .1 * energy_diff 
    grain = np.log(np.min(np.abs(np.diff(np.exp(energy)))))    
    res = pd.DataFrame([[-energy_diff], [high_level], [ground_level], [low_level], [lowest_level], [grain]],
      ["Diff", "High", "Gnd",  "Low", "Lowest", "Grain"], ["Log"])
    res["Abs alive"] = np.exp(res["Log"])
    res["Abs dead"] = 1. - np.exp(res["Log"])
    res["Max"] = 2.**32
    res["Threshold"] = res["Abs alive"]/2**-32
    print(res)

calculate_energy_levels([lambda t: -10 * t + 60/36 * np.square(t), lambda t: 60/36 * np.square(t)], lambda x, m: -m * 60/36 * x, var=7/8)
