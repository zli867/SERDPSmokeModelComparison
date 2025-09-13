import pickle
from utils import plotComparisonNoIntercept
import matplotlib.pyplot as plt
import numpy as np

with open('/Volumes/PubData/ModelComparisons/stats/emissions.pickle', 'rb') as f:
    consumptions = pickle.load(f)

bsp_vals, sfire_vals = [], []
for dates in consumptions.keys():
    print(dates)
    print(consumptions[dates]["bsp"])
    print(consumptions[dates]["sfire"])
    bsp_vals.append(consumptions[dates]["bsp"])
    sfire_vals.append(consumptions[dates]["sfire"])

bsp_vals = np.array(bsp_vals).reshape(-1, 1)
sfire_vals = np.array(sfire_vals).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bsp_vals, sfire_vals)
plotComparisonNoIntercept(bsp_vals, sfire_vals, ax)
ax.set_xlabel("BlueSky Fuel Consumptions (tons)")
ax.set_ylabel("WRF-SFIRE Fuel Consumptions (tons)")
max_value = np.max([np.max(bsp_vals), np.max(sfire_vals)])
min_value = np.min([np.min(bsp_vals), np.min(sfire_vals)])
ax.plot(np.arange(min_value - 20, max_value + 20, 10), np.arange(min_value - 20, max_value + 20, 10), '--k')
ax.set_aspect('equal')
plt.title("Total Fuel Consumptions (unit: tons)")
plt.show()
plt.show()