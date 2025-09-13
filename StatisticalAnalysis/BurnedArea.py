import pickle
from utils import plotComparisonIntercept
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

with open('/Volumes/PubData/ModelComparisons/stats/burned_area.pickle', 'rb') as f:
    burned_area = pickle.load(f)

bsp_vals, sfire_vals = [], []
for dates in burned_area.keys():
    bsp_vals.append(burned_area[dates]["bsp"])
    sfire_vals.append(burned_area[dates]["sfire"])

bsp_vals = np.array(bsp_vals).reshape(-1, 1)
sfire_vals = np.array(sfire_vals).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bsp_vals, sfire_vals)
plotComparisonIntercept(bsp_vals, sfire_vals, ax)
ax.set_xlabel("BlueSky Burned Area (acres)")
ax.set_ylabel("WRF-SFIRE Burned Area (acres)")
max_value = np.max([np.max(bsp_vals), np.max(sfire_vals)])
min_value = np.min([np.min(bsp_vals), np.min(sfire_vals)])
ax.plot(np.arange(min_value - 20, max_value + 20, 10), np.arange(min_value - 20, max_value + 20, 10), '--k')
ax.set_aspect('equal')
plt.title("Total Burned Area (unit: tons)")
plt.show()
plt.show()