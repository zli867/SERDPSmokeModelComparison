import pickle
from utils import plotComparisonNoIntercept
import matplotlib.pyplot as plt
import numpy as np

with open('/Volumes/PubData/ModelComparisons/stats/plume_height.pickle', 'rb') as f:
    consumptions = pickle.load(f)

for plume_scheme in ["Briggs", "FEPS", "SEV", "Freitas"]:
    bsp_vals, sfire_vals = [], []
    for dates in consumptions.keys():
        bsp_vals.append(consumptions[dates][plume_scheme])
        sfire_vals.append(consumptions[dates]["SFIRE"])


    bsp_vals = np.concatenate(bsp_vals).reshape(-1, 1)
    sfire_vals = np.concatenate(sfire_vals).flatten().reshape(-1, 1)
    print(np.mean(bsp_vals))
    print(np.mean(sfire_vals))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bsp_vals, sfire_vals)
    plotComparisonNoIntercept(bsp_vals, sfire_vals, ax)
    ax.set_xlabel("BlueSky %s Plume Height (m)" % plume_scheme)
    ax.set_ylabel("WRF-SFIRE Plume Height (m)")
    max_value = np.max([np.max(bsp_vals), np.max(sfire_vals)])
    min_value = np.min([np.min(bsp_vals), np.min(sfire_vals)])
    ax.plot(np.arange(min_value - 20, max_value + 20, 10), np.arange(min_value - 20, max_value + 20, 10), '--k')
    ax.set_aspect('equal')
    plt.title("Plume Height Consumptions (unit: m)")
    plt.show()



bsp_vals, sfire_vals = [], []
for dates in consumptions.keys():
    bsp_vals.append(consumptions[dates]["FEPS"])
    sfire_vals.append(consumptions[dates]["Freitas"])


bsp_vals = np.concatenate(bsp_vals).reshape(-1, 1)
sfire_vals = np.concatenate(sfire_vals).flatten().reshape(-1, 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bsp_vals, sfire_vals)
plotComparisonNoIntercept(bsp_vals, sfire_vals, ax)
ax.set_xlabel("BlueSky FEPS Plume Height (m)" )
ax.set_ylabel("Freitas Plume Height (m)")
max_value = np.max([np.max(bsp_vals), np.max(sfire_vals)])
min_value = np.min([np.min(bsp_vals), np.min(sfire_vals)])
ax.plot(np.arange(min_value - 20, max_value + 20, 10), np.arange(min_value - 20, max_value + 20, 10), '--k')
ax.set_aspect('equal')
plt.title("Plume Height Consumptions (unit: m)")
plt.show()
