import numpy as np
import matplotlib.pyplot as plt

frp_profile = np.array([0.03033772, 0.03033772, 0.03033772, 0.03033772, 0.03033772,
                        0.03033772, 0.03033772, 0.03434459, 0.03720664, 0.04006869,
                        0.05724098, 0.07441328, 0.09158558, 0.09730967, 0.06868918,
                        0.04006869, 0.03434459, 0.03033772, 0.03033772, 0.03033772,
                        0.03033772, 0.03033772, 0.03033772, 0.03033772])
WRAP_TIME_PROFILE = [0.0057, 0.0057, 0.0057, 0.0057, 0.0057, 0.0057,
                     0.0057, 0.0057, 0.0057, 0.0057, 0.0200, 0.0400,
                     0.0700, 0.1000, 0.1300, 0.1600, 0.1700, 0.1200,
                     0.0700, 0.0400, 0.0057, 0.0057, 0.0057, 0.0057]
print(np.sum(frp_profile))
print(np.sum(WRAP_TIME_PROFILE))
fig, ax = plt.subplots()
hours = np.arange(len(frp_profile))
print(len(hours))
plt.xlabel("Hours", fontsize=16)
plt.ylabel("Fractions", fontsize=16)
plt.plot(hours, frp_profile, marker='.')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.show()