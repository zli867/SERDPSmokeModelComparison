import matplotlib.pyplot as plt
import numpy as np

sev = {'FtBn_N31-A': {'N20_VIIRS': 302.7514386858259, 'N_VIIRS': 809.7586103166852, 'Aqua_MODIS': 305.6028137207031}, 'FtBn_S22-A': {'N20_VIIRS': 472.0696144104004, 'N_VIIRS': 691.0682830810547}, 'FtBn_S22-B': {'N20_VIIRS': 451.2508316040039}, 'FtBn_N30-B': {'N20_VIIRS': 486.86700439453125, 'N_VIIRS': 484.1328125, 'Aqua_MODIS': 316.5884475708008}, 'FtBn_N30-C': {'N20_VIIRS': 685.0614776611328, 'N_VIIRS': 867.9391326904297, 'Aqua_MODIS': 290.30367279052734, 'Terra_MODIS': 903.6111145019531}, 'FtBn_S50-B': {'N20_VIIRS': 476.49242146809894}, 'FtBn_S50-A': {'N20_VIIRS': 499.1991271972656}, 'FtBn_S51-A': {'N20_VIIRS': 570.8969116210938}, 'FtBn_S51-B': {'N20_VIIRS': 658.6797485351562, 'N_VIIRS': 480.99231974283856, 'Aqua_MODIS': 585.3639933268229}, 'FtBn_S52-A': {'N20_VIIRS': 544.462392171224}, 'FtBn_C48': {'N20_VIIRS': 358.87927682059154, 'N_VIIRS': 384.13311767578125, 'Aqua_MODIS': 445.84466552734375}, 'FtBn_C47-A': {'N20_VIIRS': 393.77550833565846, 'N_VIIRS': 366.65087236676897}, 'FtBn_C47-B': {'N20_VIIRS': 401.84745352608815}, 'FtBn_S46-A': {'N_VIIRS': 636.4863484700521, 'Aqua_MODIS': 531.9781290690104}, 'FtBn_N20-H': {'N20_VIIRS': 391.89270477294923, 'N_VIIRS': 383.9106414794922, 'Terra_MODIS': 848.134765625}, 'FtBn_N20-G': {'N20_VIIRS': 639.5402465820313, 'N_VIIRS': 540.889291381836, 'Terra_MODIS': 987.5809936523438}, 'FtBn_N20-E': {'N20_VIIRS': 430.7950897216797, 'N_VIIRS': 302.79318618774414, 'Terra_MODIS': 617.6757446289063}, 'FtBn_N20-A': {'N20_VIIRS': 319.84303588867186}, 'FtBn_N20-B': {'N20_VIIRS': 360.4710998535156, 'N_VIIRS': 317.4195899963379}, 'FtBn_N20-C': {'N20_VIIRS': 421.02916259765624, 'N_VIIRS': 342.1530395507813, 'Terra_MODIS': 607.8327255249023}, 'FtBn_S36-A': {'N20_VIIRS': 520.0978291829427, 'N_VIIRS': 535.3233540852865}, 'FtBn_S36-C': {'N20_VIIRS': 498.3185272216797, 'N_VIIRS': 644.6294148763021}, 'FtBn_S36-D': {'N20_VIIRS': 515.4978637695312, 'N_VIIRS': 616.687255859375, 'Aqua_MODIS': 693.7725626627604}, 'FtBn_S37-A': {'Aqua_MODIS': 432.2247680664062}, 'FtBn_S42-A': {'N20_VIIRS': 522.8620927598741, 'N_VIIRS': 505.4489339192708}, 'FtBn_S42-B': {'N20_VIIRS': 386.45619710286456, 'N_VIIRS': 391.9075639512804}}
freitas = {'FtBn_N31-A': {'N20_VIIRS': 1396.2886962890625, 'N_VIIRS': 1396.2886962890625, 'Aqua_MODIS': 1396.2886962890625}, 'FtBn_S22-A': {'N20_VIIRS': 1059.7896728515625, 'N_VIIRS': 1059.7896728515625}, 'FtBn_S22-B': {'N20_VIIRS': 1059.715202331543}, 'FtBn_N30-B': {'N20_VIIRS': 1180.2910766601562, 'N_VIIRS': 1180.2910766601562, 'Aqua_MODIS': 1180.2910766601562}, 'FtBn_N30-C': {'N20_VIIRS': 1179.5326843261719, 'N_VIIRS': 1179.5326843261719, 'Aqua_MODIS': 1179.5326843261719, 'Terra_MODIS': 1298.9627227783203}, 'FtBn_S50-B': {'N20_VIIRS': 79.21504211425781}, 'FtBn_S50-A': {'N20_VIIRS': 1028.1121622721355}, 'FtBn_S51-A': {'N20_VIIRS': 2112.280029296875}, 'FtBn_S51-B': {'N20_VIIRS': 2111.2416178385415, 'N_VIIRS': 79.25525665283203, 'Aqua_MODIS': 2111.2416178385415}, 'FtBn_S52-A': {'N20_VIIRS': 1678.8519083658855}, 'FtBn_C48': {'N20_VIIRS': 1249.272188459124, 'N_VIIRS': 1352.3016619001116, 'Aqua_MODIS': 1352.3016619001116}, 'FtBn_C47-A': {'N20_VIIRS': 1352.0963221958705, 'N_VIIRS': 1249.0359856741768}, 'FtBn_C47-B': {'N20_VIIRS': 1351.7911812918526}, 'FtBn_S46-A': {'N_VIIRS': 2025.3070882161458, 'Aqua_MODIS': 1634.10533396403}, 'FtBn_N20-H': {'N20_VIIRS': 1373.9377075195312, 'N_VIIRS': 1373.9377075195312, 'Terra_MODIS': 1373.9377075195312}, 'FtBn_N20-G': {'N20_VIIRS': 1374.0006439208985, 'N_VIIRS': 1374.0006439208985, 'Terra_MODIS': 1374.0006439208985}, 'FtBn_N20-E': {'N20_VIIRS': 1302.1361724853516, 'N_VIIRS': 75.88555335998535, 'Terra_MODIS': 1342.027328491211}, 'FtBn_N20-A': {'N20_VIIRS': 1168.3674362182617}, 'FtBn_N20-B': {'N20_VIIRS': 1306.8381103515626, 'N_VIIRS': 1136.485791015625}, 'FtBn_N20-C': {'N20_VIIRS': 1346.807080078125, 'N_VIIRS': 1306.9000854492188, 'Terra_MODIS': 1373.4034057617187}, 'FtBn_S36-A': {'N20_VIIRS': 1390.5086008707683, 'N_VIIRS': 1417.332555135091}, 'FtBn_S36-C': {'N20_VIIRS': 451.29334513346356, 'N_VIIRS': 1525.8336588541667}, 'FtBn_S36-D': {'N20_VIIRS': 120.13819885253906, 'N_VIIRS': 898.5829671223959, 'Aqua_MODIS': 1526.0635477701824}, 'FtBn_S37-A': {'Aqua_MODIS': 1456.1373413085937}, 'FtBn_S42-A': {'N20_VIIRS': 706.751478407118, 'N_VIIRS': 706.751478407118}, 'FtBn_S42-B': {'N20_VIIRS': 769.8713022867838, 'N_VIIRS': 817.8449757893881}}

sev_diffs = []
freitas_diffs = []
for burn_unit in sev.keys():
    sev_val = sev[burn_unit]
    freitas_val = freitas[burn_unit]
    sev_frp, freitas_frp = [], []
    for sat_name in sev_val.keys():
        sev_frp.append(sev_val[sat_name])
        freitas_frp.append(freitas_val[sat_name])
    if len(sev_frp) == 1:
        continue
    else:
        print(burn_unit)
        print("SEV, min: %.2f, max: %.2f, diff: %.2f" % (min(sev_frp), max(sev_frp), (max(sev_frp) - min(sev_frp)) / min(sev_frp)))
        print("Freitas, min: %.2f, max: %.2f, diff: %.2f" % (min(freitas_frp), max(freitas_frp), (max(freitas_frp) - min(freitas_frp)) / min(freitas_frp)))
        print()
        sev_diffs.append((max(sev_frp) - min(sev_frp)) / min(sev_frp))
        freitas_diffs.append((max(freitas_frp) - min(freitas_frp)) / min(freitas_frp))

# Define the bin edges with 0.1 intervals
bin_edges = np.arange(min(min(sev_diffs), min(freitas_diffs)),
                      max(max(sev_diffs), max(freitas_diffs)) + 0.1, 0.1)

sev_diffs = np.array(sev_diffs)
freitas_diffs = np.array(freitas_diffs)
threshold = 0.2
c = 0
for i in range(0, len(sev_diffs)):
    if sev_diffs[i] <= threshold:
        c += 1
print(c / len(sev_diffs))

c = 0
for i in range(0, len(freitas_diffs)):
    if freitas_diffs[i] <= threshold:
        c += 1
print(c / len(freitas_diffs))
# print(np.sum(sev_diffs[sev_diffs < 1]) / len(sev_diffs))
# print(np.sum(sev_diffs[freitas_diffs < 1]) / len(freitas_diffs))
# # Plot sev_diffs histogram
# plt.hist(sev_diffs, bins=bin_edges, edgecolor='black')
# plt.title("Histogram of sev_diffs")
# plt.xlabel("Difference")
# plt.ylabel("Frequency")
# plt.show()
#
# # Plot freitas_diffs histogram
# plt.hist(freitas_diffs, bins=bin_edges, edgecolor='black')
# plt.title("Histogram of freitas_diffs")
# plt.xlabel("Difference")
# plt.ylabel("Frequency")
# plt.show()