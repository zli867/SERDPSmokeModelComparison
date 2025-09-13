import json
from datetime import datetime, timedelta
import numpy as np


def get_bsp_plume_height(emission_file):
    plume_height_res = {}
    with open(emission_file) as jsfile:
        data = json.load(jsfile)
    for current_fire in data["fires"]:
        ign_start = datetime.strptime(current_fire["activity"][0]["active_areas"][0]["ignition_start"], '%Y-%m-%dT%H:%M:%S')
        ign_end = datetime.strptime(current_fire["activity"][0]["active_areas"][0]["ignition_end"], '%Y-%m-%dT%H:%M:%S')
        # calculate utc offset
        utc_offset = current_fire["activity"][0]["active_areas"][0]["utc_offset"]
        negative = False
        if utc_offset[0] == "-":
            delta = datetime.strptime(utc_offset[1:], "%H:%M") - datetime.strptime("00:00", "%H:%M")
            negative = True
        else:
            delta = datetime.strptime(utc_offset, "%H:%M") - datetime.strptime("00:00", "%H:%M")
        current_fire_name = current_fire["id"]
        plume_info = current_fire["activity"][0]["active_areas"][0]["specified_points"][0]["plumerise"]
        time_str_array = [] # local time
        emission_time = [] # utc
        plume_bottom = []
        plume_top = []
        for time_str in plume_info.keys():
            time_str_array.append(time_str)
        time_str_array.sort()
        for time_str in time_str_array:
            plume_values = plume_info[time_str]["heights"]
            plume_bottom.append(min(plume_values))
            plume_top.append(max(plume_values))
            datetime_object = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            if negative:
                utc_date = datetime_object + delta
            else:
                utc_date = datetime_object - delta
            emission_time.append(utc_date)
        if negative:
            ign_utc_start = ign_start + delta
            ign_utc_end = ign_end + delta
        else:
            ign_utc_start = ign_start - delta
            ign_utc_end = ign_end - delta
        start_idx, end_idx = len(emission_time), 0
        for i in range(0, len(emission_time)):
            if plume_top[i] > 0:
                start_idx = min(start_idx, i)
                end_idx = max(end_idx, i)
        plume_height_res[current_fire_name] = {
            "time": emission_time[start_idx: end_idx + 1],
            "plume_bottom": np.array(plume_bottom)[start_idx: end_idx + 1],
            "plume_top": np.array(plume_top)[start_idx: end_idx + 1],
            "start_time": ign_utc_start,
            "end_time": ign_utc_end
        }
    return plume_height_res


def heightAboveGround(ds):
    PH = ds["PH"][:]
    PHB = ds["PHB"][:]
    HGT = ds["HGT"][:]
    HGT = HGT[:, np.newaxis, :, :]
    height_above_sea_level = (PH + PHB) / 9.8
    return height_above_sea_level - HGT


def grid_cell_plume_height(vertical_column_weighted_tracer, vertical_column_thickness, quantiles):
    if np.sum(vertical_column_weighted_tracer) == 0:
        return 0
    vertical_column_tracer_normalized = vertical_column_weighted_tracer / np.sum(vertical_column_weighted_tracer)
    target_ratio = 1 - quantiles
    layer_num = len(vertical_column_tracer_normalized)
    current_sum = 0
    result_idx = None
    for layer_idx in range(0, layer_num):
        current_sum += vertical_column_tracer_normalized[layer_idx]
        if current_sum >= target_ratio:
            result_idx = layer_idx
            break
    # current plume height is sum of thickness from 0, 1, ..., result_idx
    # search index will be [result_idx - 1, result_idx] -> thickness[result_idx]
    residual_ratio = current_sum - target_ratio
    overflow_mass_ratio = (residual_ratio * np.sum(vertical_column_weighted_tracer)) / vertical_column_weighted_tracer[
        result_idx]
    overflow_height = overflow_mass_ratio * vertical_column_thickness[result_idx]
    plume_height = 0
    for idx in range(0, result_idx + 1):
        plume_height += vertical_column_thickness[idx]
    plume_height -= overflow_height
    return plume_height


def convertByteToStr(byte_list):
    str1 = ""
    for i in range(0, len(byte_list)):
        str1 = str1 + byte_list[i].decode('UTF-8')
    return str1


def calculate_2D_plume_height(wrf_fire_data, t_step, quantile_vertical=0.1, quantile_horizontal=0.1):
    # variables extraction
    layer_height = heightAboveGround(wrf_fire_data)
    wrf_tracer = wrf_fire_data["tr17_2"][:]
    thickness = layer_height[:, 1:, :, :] - layer_height[:, :-1, :, :]
    dx = wrf_fire_data.getncattr('DX')
    dy = wrf_fire_data.getncattr('DY')
    area = dx * dy
    wrf_tracer_mass = wrf_tracer * thickness * area
    xlat = wrf_fire_data["XLAT"][:][0, :, :]
    xlon = wrf_fire_data["XLONG"][:][0, :, :]
    time_str = wrf_fire_data["Times"][:]
    time_frames, layers, m, n = wrf_tracer.shape
    current_wrf_tracer_mass = wrf_tracer_mass[t_step, :, :, :]
    current_thickness = thickness[t_step, :, :, :]
    current_plume_height = np.zeros((m, n))
    current_total_tracer_mass = np.sum(current_wrf_tracer_mass)

    vertical_column_tracer_mass = np.sum(current_wrf_tracer_mass, axis=0)
    tracer_path = np.zeros(vertical_column_tracer_mass.shape)
    tracer_list = []

    for column in range(0, m):
        for row in range(0, n):
            tracer_list.append((vertical_column_tracer_mass[column, row], column, row))
            current_cell_height = grid_cell_plume_height(current_wrf_tracer_mass[:, column, row],
                                                         current_thickness[:, column, row],
                                                         quantile_vertical)
            current_plume_height[column, row] = current_cell_height
    # sort tracer list by mass from largest to smallest
    tracer_list.sort(reverse=True)

    select_total_mass = 0
    target_horizontal_ratio = 1 - quantile_horizontal
    for select_idx in range(0, len(tracer_list)):
        select_mass, max_col, max_row = tracer_list[select_idx]
        select_total_mass += select_mass
        tracer_path[max_col, max_row] = 1
        if select_total_mass >= target_horizontal_ratio * current_total_tracer_mass:
            break

    current_plume_height_selected = current_plume_height * tracer_path
    current_time_str = convertByteToStr(time_str[t_step])
    current_time_obj = datetime.strptime(current_time_str, "%Y-%m-%d_%H:%M:%S")
    return current_time_obj, current_plume_height_selected


def get_sfire_plume_height(wrf_fire_data, burn_center_lon, burn_center_lat):
    # concentration less than this quantile will be ignored
    quantile_vertical = 0.1
    quantile_horizontal = 0.1
    bin_step = 1

    wrf_tracer = wrf_fire_data["tr17_2"][:]
    xlat = wrf_fire_data["XLAT"][:][0, :, :]
    xlon = wrf_fire_data["XLONG"][:][0, :, :]
    dx = wrf_fire_data.getncattr('DX')
    dy = wrf_fire_data.getncattr('DY')
    time_frames, layers, m, n = wrf_tracer.shape
    start_time_idx = 0
    end_time_idx = time_frames - 1
    for t_step in range(0, time_frames):
        current_wrf_tracer_mass = wrf_tracer[t_step, :, :, :]
        if np.max(current_wrf_tracer_mass) > 0:
            start_time_idx = t_step
            break

    # Search start grid index
    distance = np.sqrt((xlat - burn_center_lat) ** 2 + (xlon - burn_center_lon) ** 2)
    idx = np.argwhere(distance == np.min(distance))
    ignition_x, ignition_y = idx[0]

    # store height of smoke height of centerline
    time_res = []
    height_res = []

    for t_step in range(start_time_idx, end_time_idx + 1):
        # for t_step in range(51, 51 + 1):
        current_time, current_plume_height_selected = calculate_2D_plume_height(wrf_fire_data, t_step,
                                                                                quantile_vertical, quantile_horizontal)
        # sampling to get 1d plume height
        # distance vs height
        m, n = current_plume_height_selected.shape
        distance = []
        plume_height = []
        for i in range(0, m):
            for j in range(0, n):
                # only include pixels when plume height is greater than 0
                if current_plume_height_selected[i, j] > 0:
                    delta_x = np.abs((i - ignition_x) * (dx / 1000))
                    delta_y = np.abs((j - ignition_y) * (dy / 1000))
                    pixel_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)  # unit km
                    pixel_height = current_plume_height_selected[i, j]
                    distance.append(pixel_distance)
                    plume_height.append(pixel_height)

        distance = np.array(distance)
        plume_height = np.array(plume_height)
        binned_distance = []
        binned_height = []
        start_distance = 0
        current_distance = start_distance
        while current_distance <= np.max(distance):
            lower_bound = current_distance
            upper_bound = current_distance + bin_step
            bin_height = plume_height[(distance >= lower_bound) & (distance < upper_bound)]
            if len(bin_height) == 0:
                mean_bin_height = 0
            else:
                mean_bin_height = np.mean(bin_height)
            mean_distance = (lower_bound + upper_bound) / 2
            binned_distance.append(mean_distance)
            binned_height.append(mean_bin_height)
            current_distance += bin_step

        time_res.append(current_time)
        height_res.append(np.max(binned_height))

    return {
        "time": time_res,
        "plume_top": height_res
    }


def interpolate_plumerise(ignition_start_time, ignition_end_time, plume_rise):
    new_plumerise = {}
    for key in plume_rise.keys():
        datetime_object = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S')
        for i in range(0, 3):
            cur_time = datetime_object + timedelta(minutes=20 * i)
            new_plumerise[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')] = {}
            for sub_key in plume_rise[key]:
                new_plumerise[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')][sub_key] = plume_rise[key][sub_key]
            if cur_time < ignition_start_time or cur_time > ignition_end_time:
                new_plumerise[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')]['heights'] = np.zeros(len(new_plumerise[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')]['heights']))
    return new_plumerise


def get_bsp_interpolated_plume_height(emission_file):
    plume_height_res = {}
    with open(emission_file) as jsfile:
        data = json.load(jsfile)
    for current_fire in data["fires"]:
        ign_start = datetime.strptime(current_fire["activity"][0]["active_areas"][0]["ignition_start"], '%Y-%m-%dT%H:%M:%S')
        ign_end = datetime.strptime(current_fire["activity"][0]["active_areas"][0]["ignition_end"], '%Y-%m-%dT%H:%M:%S')
        # calculate utc offset
        utc_offset = current_fire["activity"][0]["active_areas"][0]["utc_offset"]
        negative = False
        if utc_offset[0] == "-":
            delta = datetime.strptime(utc_offset[1:], "%H:%M") - datetime.strptime("00:00", "%H:%M")
            negative = True
        else:
            delta = datetime.strptime(utc_offset, "%H:%M") - datetime.strptime("00:00", "%H:%M")
        current_fire_name = current_fire["id"]
        plume_info = current_fire["activity"][0]["active_areas"][0]["specified_points"][0]["plumerise"]
        plume_info = interpolate_plumerise(ign_start, ign_end, plume_info)
        time_str_array = [] # local time
        emission_time = [] # utc
        plume_bottom = []
        plume_top = []
        for time_str in plume_info.keys():
            time_str_array.append(time_str)
        time_str_array.sort()
        for time_str in time_str_array:
            plume_values = plume_info[time_str]["heights"]
            plume_bottom.append(min(plume_values))
            plume_top.append(max(plume_values))
            datetime_object = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            if negative:
                utc_date = datetime_object + delta
            else:
                utc_date = datetime_object - delta
            emission_time.append(utc_date)
        if negative:
            ign_utc_start = ign_start + delta
            ign_utc_end = ign_end + delta
        else:
            ign_utc_start = ign_start - delta
            ign_utc_end = ign_end - delta
        start_idx, end_idx = len(emission_time), 0
        for i in range(0, len(emission_time)):
            if plume_top[i] > 0:
                start_idx = min(start_idx, i)
                end_idx = max(end_idx, i)
        plume_height_res[current_fire_name] = {
            "time": emission_time[start_idx: end_idx + 1],
            "plume_bottom": np.array(plume_bottom)[start_idx: end_idx + 1],
            "plume_top": np.array(plume_top)[start_idx: end_idx + 1],
            "start_time": ign_utc_start,
            "end_time": ign_utc_end
        }
    return plume_height_res