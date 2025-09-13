import pickle
import matplotlib.pyplot as plt

with open('/Volumes/Shield/ModelComparisons/stats_data/burned_area_type.pickle', 'rb') as f:
    burned_area = pickle.load(f)

bsp_fuel_total = burned_area["bsp"]
sfire_fuel_total = burned_area["sfire"]
fig, axs = plt.subplots(2, 1, figsize=(16, 6))
ax_ravel = axs.ravel()
# bluesky
ax = ax_ravel[0]
fuel_types = list(bsp_fuel_total.keys())
fuel_burned_area = list(bsp_fuel_total.values())
ax.bar(fuel_types, fuel_burned_area, width=0.4)
ax.set_ylabel("Burned Area (acres)", fontsize=15)
ax.set_title("BlueSky Burned Fuel", fontsize=15)
print(fuel_types)
print(fuel_burned_area)
# SFIRE
ax = ax_ravel[1]
fuel_types = list(sfire_fuel_total.keys())
fuel_burned_area = list(sfire_fuel_total.values())
ax.bar(fuel_types, fuel_burned_area, width=0.4)
ax.set_ylabel("Burned Area (acres)", fontsize=15)
ax.tick_params(axis='x', labelrotation=30)
ax.set_title("WRF-SFIRE Burned Fuel", fontsize=15)
plt.tight_layout()
plt.show()
print(fuel_types)
print(fuel_burned_area)