
import pickle
import matplotlib.pyplot as plt

with open('/Volumes/Shield/ModelComparisons/stats_data/consumptions_category.pickle', 'rb') as f:
    res = pickle.load(f)

bsp_consumption_total = res["bsp"]
sfire_consumption_total = res["sfire"]
fig, axs = plt.subplots(2, 1, figsize=(16, 6))
ax_ravel = axs.ravel()
# bluesky
ax = ax_ravel[0]
fuel_types = list(bsp_consumption_total.keys())
fuel_consumptions = list(bsp_consumption_total.values())
ax.bar(fuel_types, fuel_consumptions, width=0.4)
ax.set_ylabel("Consumptions (tons)", fontsize=15)
ax.set_title("BlueSky Consumed Fuel", fontsize=15)
print(fuel_types)
print(fuel_consumptions)
# SFIRE
ax = ax_ravel[1]
fuel_types = list(sfire_consumption_total.keys())
fuel_consumptions = list(sfire_consumption_total.values())
ax.bar(fuel_types, fuel_consumptions, width=0.4)
ax.set_ylabel("Consumptions (tons)", fontsize=15)
ax.tick_params(axis='x', labelrotation=30)
ax.set_title("WRF-SFIRE Consumed Fuel", fontsize=15)
print(fuel_types)
print(fuel_consumptions)
plt.tight_layout()
plt.show()