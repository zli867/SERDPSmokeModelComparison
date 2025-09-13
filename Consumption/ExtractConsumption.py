import json
from utils import findSpatialIndex, extract_wrf_time_array, anderson_category
import numpy as np

short_ton_to_ton = 0.907185

mapping_info = {
    "basal_accum_loading": ["ground fuels", "basal accumulations"],
    "duff_lower_loading": ["ground fuels", "duff lower"],
    "duff_upper_loading": ["ground fuels", "duff upper"],
    "ladderfuels_loading": ["canopy", "ladder fuels"],
    "lichen_loading": ["litter-lichen-moss", "lichen"],
    "litter_loading": ["litter-lichen-moss", "litter"],
    "midstory_loading": ["canopy", "midstory"],
    "moss_loading": ["litter-lichen-moss", "moss"],
    "nw_primary_loading": ["nonwoody", "primary live"],
    "nw_secondary_loading": ["nonwoody", "secondary live"],
    "overstory_loading": ["canopy", "overstory"],
    # "pile_clean_loading": [],
    # "pile_dirty_loading": [],
    # "pile_vdirty_loading": [],
    "shrubs_primary_loading": ["shrub", "primary live"],
    "shrubs_secondary_loading": ["shrub", "secondary live"],
    "snags_c1_foliage_loading": ["canopy", "snags class 1 foliage"],
    "snags_c1_wood_loading": ["canopy", "snags class 1 wood"],
    "snags_c1wo_foliage_loading": ["canopy", "snags class 1 no foliage"],
    "snags_c2_loading": ["canopy", "snags class 2"],
    "snags_c3_loading": ["canopy", "snags class 3"],
    "squirrel_midden_loading": ["ground fuels", "squirrel middens"],
    # "total_available_fuel_loading": [],
    "understory_loading": ["canopy", "understory"],
    # reference:
    # https://github.com/pnwairfire/bluesky/blob/1e29bc2450df688f1e00fa3a2ebb705963f0219a/bluesky/consumeutils.py#L203
    "w_rotten_3_9_loading": ["woody fuels", "1000-hr fuels rotten"],
    "w_rotten_9_20_loading": ["woody fuels", "10000-hr fuels rotten"],
    "w_rotten_gt20_loading": ["woody fuels", "10k+-hr fuels rotten"],
    "w_sound_0_quarter_loading": ["woody fuels", "1-hr fuels"],
    "w_sound_1_3_loading": ["woody fuels", "100-hr fuels"],
    "w_sound_3_9_loading": ["woody fuels", "1000-hr fuels sound"],
    "w_sound_9_20_loading": ["woody fuels", "10000-hr fuels sound"],
    "w_sound_gt20_loading": ["woody fuels", "10k+-hr fuels sound"],
    "w_sound_quarter_1_loading": ["woody fuels", "10-hr fuels"],
    "w_stump_lightered_loading": ["woody fuels", "stumps lightered"],
    "w_stump_rotten_loading": ["woody fuels", "stumps rotten"],
    "w_stump_sound_loading": ["woody fuels", "stumps sound"]
}


def return_mapped_type(mapping_info_level_1, mapping_info_level_2):
    for key in mapping_info.keys():
        if mapping_info_level_1 == mapping_info[key][0] and mapping_info_level_2 == mapping_info[key][1]:
            return key
    return None


def get_bsp_consumptions(bsp_filename):
    global short_ton_to_ton
    with open(bsp_filename) as jsfile:
        fire_data = json.load(jsfile)
    fires = fire_data["fires"]
    # {fire_name: {fuel_name: consumptions}}
    res_dict = {}
    for fire in fires:
        fire_name = fire["id"]
        consumption_res = {}
        res_dict[fire_name] = {}
        fuelbeds = fire["activity"][0]["active_areas"][0]["specified_points"][0]["fuelbeds"]
        for fuelbed in fuelbeds:
            consumption = fuelbed["consumption"]
            for mapping_info_level_1 in consumption.keys():
                for mapping_info_level_2 in consumption[mapping_info_level_1].keys():
                    values = consumption[mapping_info_level_1][mapping_info_level_2]["flaming"][0] + \
                             consumption[mapping_info_level_1][mapping_info_level_2]["residual"][0] + \
                             consumption[mapping_info_level_1][mapping_info_level_2]["smoldering"][0]
                    if values > 0:
                        mapped_type = return_mapped_type(mapping_info_level_1, mapping_info_level_2)
                        if mapped_type in consumption_res.keys():
                            consumption_res[mapped_type] += values
                        else:
                            consumption_res[mapped_type] = values

        # reduce to first level
        for fuel_type in consumption_res.keys():
            if consumption_res[fuel_type] > 0:
                current_type = mapping_info[fuel_type][0]
                res_dict[fire_name][current_type] = 0

        for fuel_type in consumption_res.keys():
            if consumption_res[fuel_type] > 0:
                current_type = mapping_info[fuel_type][0]
                res_dict[fire_name][current_type] += consumption_res[fuel_type] * short_ton_to_ton
    return res_dict


def get_sfire_consumptions(wrf_data, lon_min, lon_max, lat_min, lat_max):
    m2_to_acres = 0.000247105381
    dx = wrf_data.getncattr('DX')
    dy = wrf_data.getncattr('DY')
    sub_grid_ratio = int(wrf_data["FXLONG"].shape[1] / (wrf_data["XLONG"].shape[1] + 1))
    grid_area = (dx / sub_grid_ratio) * (dy / sub_grid_ratio)  # m^2
    # Fuel Loading (kg/m^2)
    fgi = np.array([0.166, 0.897, 0.675, 2.468, 0.785, 1.345, 1.092, 1.121, 0.780, 2.694, 2.582, 7.749, 13.024])

    # Read NFUEL_CAT, FUEL_FRAC
    fuel_cat = wrf_data["NFUEL_CAT"][:][0, :-sub_grid_ratio, :-sub_grid_ratio]
    fuel_frac = wrf_data["FUEL_FRAC"][:][:, :-sub_grid_ratio, :-sub_grid_ratio]
    lon_sub = wrf_data["FXLONG"][0, :-sub_grid_ratio, :-sub_grid_ratio]
    lat_sub = wrf_data["FXLAT"][0, :-sub_grid_ratio, :-sub_grid_ratio]

    # lon_min, lon_max, lat_min, lat_max
    bounds = [[lon_min, lat_min], [lon_min, lat_max], [lon_max, lat_min], [lon_max, lat_max]]
    y_range_idx = []
    x_range_idx = []
    for bound in bounds:
        corner_x, corner_y = findSpatialIndex(bound[0], bound[1], lon_sub, lat_sub)
        x_range_idx.append(corner_x)
        y_range_idx.append(corner_y)
    start_y_idx = min(y_range_idx)
    end_y_idx = max(y_range_idx)
    start_x_idx = min(x_range_idx)
    end_x_idx = max(x_range_idx)

    fuel_cat = fuel_cat[start_x_idx - 2: end_x_idx + 2, start_y_idx - 2: end_y_idx + 2]
    fuel_frac = fuel_frac[:, start_x_idx - 2: end_x_idx + 2, start_y_idx - 2: end_y_idx + 2]
    # DO NOT APPLY AREA FILTER
    # UNCOMMENT CODE BELOW
    mask_matrix = np.zeros((13, fuel_cat.shape[0], fuel_cat.shape[1]))
    for i in range(0, 13):
        fuel_cat_num = 1 + i
        mask_matrix_tmp = np.zeros(fuel_cat.shape)
        mask_matrix_tmp[fuel_cat == fuel_cat_num] = 1
        mask_matrix[i, :, :] = mask_matrix_tmp

    # total_emiss: time, emission species
    # total_consume: time, fuel category
    total_consumed_fuel = np.zeros(13)
    for t in range(0, fuel_frac.shape[0] - 1):
        fuel_frac_curr = fuel_frac[t, :, :]
        fuel_frac_nxt = fuel_frac[t + 1, :, :]
        fuel_frac_diff = fuel_frac_curr - fuel_frac_nxt
        consume_area = mask_matrix * fuel_frac_diff * grid_area  # m^2
        consume_fuel = consume_area * fgi[:, None, None]  # m2 * kg/m^2 = kg
        consume_fuel_total = np.sum(consume_fuel, axis=(1, 2))
        total_consumed_fuel += consume_fuel_total

    # unit: tons
    total_consumed_fuel = total_consumed_fuel / 1000

    # res: {"fuel_name": consumptions}
    res = {}
    for i in range(0, len(total_consumed_fuel)):
        res[anderson_category(i + 1)] = total_consumed_fuel[i]
    return res
