import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
from utils import findSpatialIndex, extract_wrf_time_array
import json


def hourlyFraction(timeprofile, utc_offset):
    """

    :param timeprofile: timeprofile form bsp_output, a dict
    :param utc_offset: utc_offset from bsp_output, a string
    :return: a dictionary: {"flaming": [hourly fraction], "smoldering": [hourly fraction], "residual": hourly fraction}
    """
    # convert utc_offset to time delta
    negative = False
    if utc_offset[0] == "-":
        delta = datetime.strptime(utc_offset[1:], "%H:%M") - datetime.strptime("00:00", "%H:%M")
        negative = True
    else:
        delta = datetime.strptime(utc_offset, "%H:%M") - datetime.strptime("00:00", "%H:%M")
    # generate hourly fraction
    emission_time = []
    flaming_hourly_frac = []
    smoldering_hourly_frac = []
    residual_hourly_farc = []
    for key in timeprofile.keys():
        datetime_object = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S')
        # local time to utc
        if negative:
            utc_date = datetime_object + delta
        else:
            utc_date = datetime_object - delta
        emission_time.append(utc_date)
        flaming_hourly_frac.append(timeprofile[key]['flaming'])
        smoldering_hourly_frac.append(timeprofile[key]['smoldering'])
        residual_hourly_farc.append(timeprofile[key]['residual'])
    return {"emission_time": emission_time,
            "flaming_hourly_frac": np.array(flaming_hourly_frac),
            "smoldering_hourly_frac": np.array(smoldering_hourly_frac),
            "residual_hourly_farc": np.array(residual_hourly_farc)
            }


def classifiedEmission(fuelbeds, select_species):
    """

    :param fuelbeds:
    :param select_species:
    :return:
    """
    # initialize flaming residual and smoldering emissions
    flaming = {}
    for select_specie in select_species:
        flaming[select_specie] = 0
    smoldering = {}
    for select_specie in select_species:
        smoldering[select_specie] = 0
    residual = {}
    for select_specie in select_species:
        residual[select_specie] = 0

    for fuel_idx in range(0, len(fuelbeds)):
        fuelbed_flaming = fuelbeds[fuel_idx]["emissions"]['flaming']
        fuelbed_smoldering = fuelbeds[fuel_idx]["emissions"]['smoldering']
        fuelbed_residual = fuelbeds[fuel_idx]["emissions"]['residual']
        for specie in select_species:
            if specie in fuelbed_flaming.keys():
                flaming[specie] += fuelbed_flaming[specie][0]
            if specie in fuelbed_smoldering.keys():
                smoldering[specie] += fuelbed_smoldering[specie][0]
            if specie in fuelbed_residual.keys():
                residual[specie] += fuelbed_residual[specie][0]

    # generate flaming smoldering residual emission array
    flaming_arry = []
    smoldering_arry = []
    residual_arry = []
    for select_specie in select_species:
        flaming_arry.append(flaming[select_specie])
        smoldering_arry.append(smoldering[select_specie])
        residual_arry.append(residual[select_specie])
    flaming_arry = np.array(flaming_arry)
    smoldering_arry = np.array(smoldering_arry)
    residual_arry = np.array(residual_arry)
    return {"flaming_emission": flaming_arry, "smoldering_emission": smoldering_arry, "residual_emission": residual_arry}


def get_bsp_emissions(bsp_filename, select_species):
    """

    :param emission_file: bluesky output
    :param select_species: select chemical species
    :return: a dictionary including fire name id and its emissions for each hour, the unit is metric ton
    """
    short_ton_to_ton = 0.9071847
    emission_dict = {}
    with open(bsp_filename) as jsfile:
        data = json.load(jsfile)
    for current_fire in data["fires"]:
        current_fire_name = current_fire["id"]
        current_fire_timeprofile = hourlyFraction(current_fire["activity"][0]["active_areas"][0]["timeprofile"], current_fire["activity"][0]["active_areas"][0]["utc_offset"])
        current_fire_emission = classifiedEmission(current_fire["activity"][0]['active_areas'][0]["specified_points"][0]["fuelbeds"], select_species)
        # TODO: UTC?
        species_num = len(select_species)
        emission_dict[current_fire_name] = {
            "time": current_fire_timeprofile["emission_time"]
        }
        for i in range(0, len(select_species)):
            select_specie = select_species[i]
            emission_dict[current_fire_name][select_specie] = {
                "flaming": current_fire_emission["flaming_emission"][i] * current_fire_timeprofile["flaming_hourly_frac"] * short_ton_to_ton,
                "smoldering": current_fire_emission["smoldering_emission"][i] * current_fire_timeprofile["smoldering_hourly_frac"] * short_ton_to_ton,
                "residual": current_fire_emission["residual_emission"][i] * current_fire_timeprofile["residual_hourly_farc"] * short_ton_to_ton
            }
    return emission_dict


def interpolate_timeprofile(ignition_start_time, ignition_end_time, time_profile):
    round_start_time = datetime(ignition_start_time.year, ignition_start_time.month, ignition_start_time.day,
                                ignition_start_time.hour, (ignition_start_time.minute // 20) * 20)
    # print(round_start_time)
    # round_end_time = datetime(ignition_end_time.year, ignition_end_time.month, ignition_end_time.day,
    #                             ignition_end_time.hour, (ignition_end_time.minute // 20) * 20)
    new_timeprofiles = {}
    for key in time_profile.keys():
        datetime_object = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S')
        for i in range(0, 3):
            split_nums = 3
            if datetime_object.hour == round_start_time.hour:
                split_nums = 3 - round_start_time.minute // 20
            cur_time = datetime_object + timedelta(minutes=20 * i)
            if split_nums != 0 and round_start_time <= cur_time:
                new_timeprofiles[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')] = {
                    'area_fraction': 0.0, 'flaming': 0.0, 'residual': 0.0, 'smoldering': 0.0
                }
                for sub_key in new_timeprofiles[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')].keys():
                    new_timeprofiles[datetime.strftime(cur_time, '%Y-%m-%dT%H:%M:%S')][sub_key] = time_profile[key][sub_key] / split_nums
    return new_timeprofiles


def get_bsp_interpolated_emissions(bsp_filename, select_species):
    short_ton_to_ton = 0.9071847
    emission_dict = {}
    with open(bsp_filename) as jsfile:
        data = json.load(jsfile)

    for current_fire in data["fires"]:
        ignition_start_time = datetime.strptime(current_fire["activity"][0]['active_areas'][0]["ignition_start"], "%Y-%m-%dT%H:%M:%S")
        ignition_end_time = datetime.strptime(current_fire["activity"][0]['active_areas'][0]["ignition_end"], "%Y-%m-%dT%H:%M:%S")
        fire_id = current_fire["id"]
        time_profile = current_fire["activity"][0]["active_areas"][0]["timeprofile"]
        utc_offset = current_fire["activity"][0]['active_areas'][0]["utc_offset"]
        negative = False
        if utc_offset[0] == "-":
            delta = datetime.strptime(utc_offset[1:], "%H:%M") - datetime.strptime("00:00", "%H:%M")
            negative = True
        else:
            delta = datetime.strptime(utc_offset, "%H:%M") - datetime.strptime("00:00", "%H:%M")

        time_profile = interpolate_timeprofile(ignition_start_time, ignition_end_time, time_profile)
        current_fire_emission = classifiedEmission(current_fire["activity"][0]['active_areas'][0]["specified_points"][0]["fuelbeds"], select_species)
        print(current_fire_emission)
        emission_time = []
        for key in time_profile.keys():
            datetime_object = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S')
            # local time to utc
            if negative:
                utc_date = datetime_object + delta
            else:
                utc_date = datetime_object - delta
            emission_time.append(utc_date)
        emission_dict[fire_id] = {"time": emission_time}
        current_fire_timeprofile = {"flaming_hourly_frac": np.zeros(len(emission_time)), "smoldering_hourly_frac": np.zeros(len(emission_time)), "residual_hourly_farc": np.zeros(len(emission_time))}
        # refactor time profile
        for key in time_profile.keys():
            datetime_object = datetime.strptime(key, '%Y-%m-%dT%H:%M:%S')
            # local time to utc
            if negative:
                utc_date = datetime_object + delta
            else:
                utc_date = datetime_object - delta
            idx = emission_time.index(utc_date)
            current_fire_timeprofile["flaming_hourly_frac"][idx] = time_profile[key]["flaming"]
            current_fire_timeprofile["smoldering_hourly_frac"][idx] = time_profile[key]["smoldering"]
            current_fire_timeprofile["residual_hourly_farc"][idx] = time_profile[key]["residual"]
        for i in range(0, len(select_species)):
            select_specie = select_species[i]
            emission_dict[fire_id][select_specie] = {
                "flaming": current_fire_emission["flaming_emission"][i] * current_fire_timeprofile["flaming_hourly_frac"] * short_ton_to_ton,
                "smoldering": current_fire_emission["smoldering_emission"][i] * current_fire_timeprofile["smoldering_hourly_frac"] * short_ton_to_ton,
                "residual": current_fire_emission["residual_emission"][i] * current_fire_timeprofile["residual_hourly_farc"] * short_ton_to_ton
            }
    return emission_dict


def get_sfire_emissions(wrf_data, select_species, lon_min, lon_max, lat_min, lat_max):
    # res: {"time", "select_specie"}
    emission_file = "/Volumes/Shield/FireFrameworkCF/Emissions/data/wrf_sfire/namelist.fire_emissions_RADM2GOCART"
    g_to_ton = 1/1000000
    m2_to_acres = 0.000247105381
    dx = wrf_data.getncattr('DX')
    dy = wrf_data.getncattr('DY')
    sub_grid_ratio = int(wrf_data["FXLONG"].shape[1] / (wrf_data["XLONG"].shape[1] + 1))
    grid_area = (dx / sub_grid_ratio) * (dy / sub_grid_ratio)  # m^2
    wrf_time_array = extract_wrf_time_array(wrf_data)
    # Fuel Loading (kg/m^2)
    fgi = np.array([0.166, 0.897, 0.675, 2.468, 0.785, 1.345, 1.092, 1.121, 0.780, 2.694, 2.582, 7.749, 13.024])
    species_mapping = {
        'co': {'unit': 'g/kg'},
        'no': {'unit': 'g/kg'},
        'no2': {'unit': 'g/kg'},
        'so2': {'unit': 'g/kg'},
        'nh3': {'unit': 'g/kg'},
        'p25': {'unit': 'g/kg'},
        'oc1': {'unit': 'g/kg'},
        'oc2': {'unit': 'g/kg'},
        'bc1': {'unit': 'g/kg'},
        'bc2': {'unit': 'g/kg'},
        'ald': {'unit': 'mole/kg'},
        'csl': {'unit': 'mole/kg'},
        'eth': {'unit': 'mole/kg'},
        'hc3': {'unit': 'mole/kg'},
        'hc5': {'unit': 'mole/kg'},
        'hcho': {'unit': 'mole/kg'},
        'iso': {'unit': 'mole/kg'},
        'ket': {'unit': 'mole/kg'},
        'mgly': {'unit': 'mole/kg'},
        'ol2': {'unit': 'mole/kg'},
        'olt': {'unit': 'mole/kg'},
        'oli': {'unit': 'mole/kg'},
        'ora2': {'unit': 'mole/kg'},
        'tol': {'unit': 'mole/kg'},
        'xyl': {'unit': 'mole/kg'},
        'bigalk': {'unit': 'mole/kg'},
        'ch4': {'unit': 'mole/kg'},
        'area': {'unit': 'm2'}
    }
    emiss_mapping = {}
    species_name = []
    # Using readlines()
    file1 = open(emission_file, 'r')
    lines = file1.readlines()
    for line in lines:
        if line[0] == '!' or line[0] == '&' or line[0] == '/':
            continue
        elif 'compatible_chem_opt' in line or 'printsums' in line:
            continue
        else:
            line = line.replace("\n", "")
            line = line.replace(' ', '')
            split_line = line.split('=')
            species = split_line[0]
            emiss_factor = split_line[1].split(',')
            emiss_factor_arry = np.zeros(13)
            for i in range(0, len(emiss_factor_arry)):
                emiss_factor_arry[i] = float(emiss_factor[i])
            emiss_mapping[species] = emiss_factor_arry
            species_name.append(species)
    print(species_name)
    print("Remeber to check emission table file: %s" % emission_file)
    # Generate factor matrix (species * category)
    emiss_factor_matrix = np.zeros((len(species_name), 13))
    for i in range(0, len(species_name)):
        emiss_factor_matrix[i, :] = emiss_mapping[species_name[i]]

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

    total_emiss = np.zeros((fuel_frac.shape[0] - 1, len(species_name)))
    total_consume = np.zeros((fuel_frac.shape[0] - 1, 13))
    for t in range(0, fuel_frac.shape[0] - 1):
        fuel_frac_curr = fuel_frac[t, :, :]
        fuel_frac_nxt = fuel_frac[t + 1, :, :]
        fuel_frac_diff = fuel_frac_curr - fuel_frac_nxt
        consume_area = mask_matrix * fuel_frac_diff * grid_area  # m^2
        consume_fuel = consume_area * fgi[:, None, None]
        consume_fuel_total = np.sum(consume_fuel, axis=(1, 2))
        consume_area_total = np.sum(consume_area, axis=(1, 2))
        # emiss_matrix = (species * category)
        emiss_matrix = consume_fuel_total * emiss_factor_matrix
        total_emiss_species = np.sum(emiss_matrix, axis=1)
        total_emiss[t, :] = total_emiss_species
        total_consume[t, :] = consume_area_total

    # Original Emiss (total_emiss, total_consume)
    total_consume = np.reshape(np.sum(total_consume, axis=1), (-1, 1))
    emiss_time = wrf_time_array[0: -1]
    data_matrix = np.hstack((total_emiss, total_consume))
    # convert unit
    species_name.append('area')
    print(species_name)
    for i in range(0, len(species_name)):
        if species_mapping[species_name[i]]['unit'] == 'g/kg':
            data_matrix[:, i] = data_matrix[:, i] * g_to_ton
        if species_mapping[species_name[i]]['unit'] == 'm2':
            data_matrix[:, i] = data_matrix[:, i] * m2_to_acres

    res = {}
    res["time"] = emiss_time
    for current_specie in select_species:
        specie_idx = species_name.index(current_specie)
        emission_series = data_matrix[:, specie_idx]
        res[current_specie] = emission_series
    return res

