import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom
from datetime import datetime, timedelta
from utils import smoke_concentration
import pyproj
import copy

def adjust_ratio(current_height, surface_height):
    # For open terrain (grassland) the typical range is 0.01-0.05 m
    z0 = 0.03
    d = 0
    ratio = np.log((surface_height - d)/z0)/np.log((current_height - d)/z0)
    return ratio


def WRFGridInfo(ds):
    crs = pyproj.Proj(proj='lcc',  # projection type: Lambert Conformal Conic
                      lat_1=ds.TRUELAT1, lat_2=ds.TRUELAT2,  # Cone intersects with the sphere
                      lat_0=ds.MOAD_CEN_LAT, lon_0=ds.STAND_LON,  # Center point
                      a=6370000, b=6370000)  # This is it! The Earth is a perfect sphere
    # Grid parameters
    dx, dy = ds.DX, ds.DY
    nx, ny = len(ds.dimensions["west_east"]), len(ds.dimensions["south_north"])
    # Down left corner of the domain
    e, n = crs(ds.CEN_LON, ds.CEN_LAT)
    x0 = -(nx - 1) / 2. * dx + e
    y0 = -(ny - 1) / 2. * dy + n
    # 2d grid
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
    Xcenters = xx
    Ycenters = yy
    lat_ctr = ds["XLAT"][0, :, :]
    lon_ctr = ds["XLONG"][0, :, :]
    xcell = dx
    ycell = dy
    # Boundary X, Y
    x_bdry = np.arange(nx + 1) * dx + x0 - dx / 2
    y_bdry = np.arange(ny + 1) * dy + y0 - dy / 2
    Xbounds, Ybounds = np.meshgrid(x_bdry, y_bdry)
    x_max = np.max(Xbounds)
    x_min = np.min(Xbounds)
    y_max = np.max(Ybounds)
    y_min = np.min(Ybounds)
    wrf_time_array = []
    wrf_time = ds["Times"][:]
    for i in range(0, wrf_time.shape[0]):
        current_time_str = ""
        for j in range(0, wrf_time.shape[1]):
            current_time_str = current_time_str + wrf_time[i][j].decode()
        current_time_obj = datetime.strptime(current_time_str, '%Y-%m-%d_%H:%M:%S')
        wrf_time_array.append(current_time_obj)
    res_dict = {"crs": crs, "X": Xcenters, "Y": Ycenters,
                "time": wrf_time_array,
                "Lat": lat_ctr, "Lon": lon_ctr,
                "XCELL": xcell, "YCELL": ycell, "X_bdry": [x_min, x_max], "Y_bdry": [y_min, y_max],}
    return res_dict


def CMAQGridInfo(cmaq_ds):
    time_data = cmaq_ds['TFLAG'][:]
    lat_1 = cmaq_ds.getncattr('P_ALP')
    lat_2 = cmaq_ds.getncattr('P_BET')
    lat_0 = cmaq_ds.getncattr('YCENT')
    lon_0 = cmaq_ds.getncattr('XCENT')
    crs = pyproj.Proj("+proj=lcc +a=6370000.0 +b=6370000.0 +lat_1=" + str(lat_1)
                      + " +lat_2=" + str(lat_2) + " +lat_0=" + str(lat_0) +
                      " +lon_0=" + str(lon_0))
    xcell = cmaq_ds.getncattr('XCELL')
    ycell = cmaq_ds.getncattr('YCELL')
    xorig = cmaq_ds.getncattr('XORIG')
    yorig = cmaq_ds.getncattr('YORIG')

    ncols = cmaq_ds.getncattr('NCOLS')
    nrows = cmaq_ds.getncattr('NROWS')

    # > for X, Y cell centers
    x_center_range = np.linspace(xorig + xcell / 2, (xorig + xcell / 2) + xcell * (ncols - 1), ncols)
    y_center_range = np.linspace(yorig + ycell / 2, (yorig + ycell / 2) + ycell * (nrows - 1), nrows)

    Xcenters, Ycenters = np.meshgrid(x_center_range, y_center_range)

    # > for X, Y cell boundaries (i.e., cell corners)
    x_bound_range = np.linspace(xorig, xorig + xcell * ncols, ncols + 1)
    y_bound_range = np.linspace(yorig, yorig + ycell * nrows, nrows + 1)

    Xbounds, Ybounds = np.meshgrid(x_bound_range, y_bound_range)

    x_max = np.max(Xbounds)
    x_min = np.min(Xbounds)
    y_max = np.max(Ybounds)
    y_min = np.min(Ybounds)

    cmaq_time_array = []
    for i in range(0, time_data.shape[0]):
        time_data_tmp = time_data[i, 0, :]
        time_str = str(time_data_tmp[0]) + str(time_data_tmp[1]).rjust(6, '0')
        parsed = datetime.strptime(time_str, '%Y%j%H%M%S')
        cmaq_time_array.append(parsed)

    lon_ctr, lat_ctr = crs(Xcenters, Ycenters, inverse=True)
    lon_bounds, lat_bounds = crs(Xbounds, Ybounds, inverse=True)
    lat_min = np.min(lat_bounds)
    lat_max = np.max(lat_bounds)
    lon_min = np.min(lon_bounds)
    lon_max = np.max(lon_bounds)
    res_dict = {"crs": crs, "X": Xcenters, "Y": Ycenters, "X_bdry": [x_min, x_max], "Y_bdry": [y_min, y_max],
                "time": cmaq_time_array, "Lat": lat_ctr, "Lon": lon_ctr, "Lat_bdry": [lat_min, lat_max], "Lon_bdry": [lon_min, lon_max],
                "XCELL": xcell, "YCELL": ycell}
    return res_dict


def wind_2_uv(wdspd, wddir):
    u = -wdspd * np.sin(np.deg2rad(wddir))
    v = -wdspd * np.cos(np.deg2rad(wddir))
    return u, v


def uv_2_wind(u, v):
    wdspd = np.sqrt(u**2 + v**2)
    wddir = np.mod(180+np.rad2deg(np.arctan2(u, v)), 360)
    return wdspd, wddir


def calculate_transport_dist(traj_dict):
    x = traj_dict["x"].copy()
    y = traj_dict["y"].copy()
    x.append(traj_dict['dist'][0])
    y.append(traj_dict['dist'][1])
    dist = 0
    for i in range(1, len(x)):
        dist += np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
    return dist


def get_model_coord_idx(x, y, model_info):
    # Nearest grid
    distance = (model_info["X"] - x) ** 2 + (model_info["Y"] - y) ** 2
    x_idx, y_idx = np.where(distance == np.min(distance))
    return x_idx, y_idx


def line_equation(sx, sy, ex, ey):
    A = ey - sy
    B = sx - ex
    C = ex * sy - sx * ey
    return A, B, C


def point_to_line_distance(sx, sy, ex, ey, x0, y0):
    A, B, C = line_equation(sx, sy, ex, ey)
    d = np.abs(A * x0 + B * y0 + C) / np.sqrt(A * A + B * B)
    x_d = (B * B * x0 - A * B * y0 - A * C) / (A * A + B * B)
    y_d = (- A * B * x0 + A * A * y0 - B * C) / (A * A + B * B)
    # whether the (x_d, y_d) is in the line
    product = (x_d - sx) * (x_d - ex) + (y_d - sy) * (y_d - ey)
    if product < 0:
        # in the line
        return d, x_d, y_d
    else:
        # dist1 = np.sqrt((x_d - sx) ** 2 + (y_d - sy) ** 2)
        # dist2 = np.sqrt((x_d - ex) ** 2 + (y_d - ey) ** 2)
        dist1 = np.sqrt((x0 - sx) ** 2 + (y0 - sy) ** 2)
        dist2 = np.sqrt((x0 - ex) ** 2 + (y0 - ey) ** 2)
        if dist2 > dist1:
            return dist1, sx, sy
        else:
            return dist2, ex, ey


def get_polygons_ctr(polygons):
    multi_poly = geom.MultiPolygon(polygons)
    multi_poly_centroid = multi_poly.centroid
    unit_coord_lon, unit_coord_lat = multi_poly_centroid.x, multi_poly_centroid.y
    return unit_coord_lon, unit_coord_lat
# DEBUG for distance
# sx, sy = 0.5, 0.5
# ex, ey = 1, 1
# # sx, sy = -1, 0
# # ex, ey = 1, 0
# point_x, point_y = -1, -1
# plt.plot([sx, ex], [sy, ey])
# plt.scatter([point_x], [point_y])
# d, x_d, y_d = point_to_line_distance(sx, sy, ex, ey, point_x, point_y)
# plt.scatter([x_d], [y_d], marker='x')
# plt.title("d: " + str(d))
# plt.show()


def hourly_met_data(met_df, select_date):
    """

    :param met_df: dataframe of met file
    :param select_date: datetime object
    :return: an hourly {"monitor_name": {"loc": [], "wdspd": [], "wddir": [], "time": []}}
    """
    res = {}
    select_df = met_df[(met_df["UTC_time"] >= select_date) & (met_df["UTC_time"] < select_date + timedelta(days=1))]
    select_df = select_df.reset_index(drop=True)
    # monitor location
    monitor_names = list(set(select_df["monitor"].to_numpy()))
    for monitor_name in monitor_names:
        # filter day time (~ 9am)
        current_df = select_df[
            (select_df["monitor"] == monitor_name) & (select_df["UTC_time"] > select_date + timedelta(hours=15))]
        if len(current_df) > 0:
            res[monitor_name] = {}
            res[monitor_name]["loc"] = [current_df["lon"].to_numpy()[0], current_df["lat"].to_numpy()[0]]
    # hourly wddir, wdspd and time array
    for monitor_name in res.keys():
        monitor_cur_df = select_df[select_df["monitor"] == monitor_name]
        cur_wdspd = []
        cur_wddir = []
        cur_time = []
        start_utc_time = select_date
        end_utc_time = select_date + timedelta(days=1)
        while start_utc_time < end_utc_time:
            cur_df = monitor_cur_df[(monitor_cur_df["UTC_time"] >= start_utc_time) & (
                        monitor_cur_df["UTC_time"] < start_utc_time + timedelta(hours=1))]
            if len(cur_df) > 0:
                if len(cur_df) == 1:
                    cur_wdspd.append(cur_df["wdspd"].values[0])
                    cur_wddir.append(cur_df["wddir"].values[0])
                else:
                    u, v = wind_2_uv(cur_df["wdspd"].to_numpy(), cur_df["wddir"].to_numpy())
                    mean_u, mean_v = np.mean(u), np.mean(v)
                    mean_spd, mean_dir = uv_2_wind(mean_u, mean_v)
                    cur_wdspd.append(mean_spd)
                    cur_wddir.append(mean_dir)
                cur_time.append(start_utc_time)
            start_utc_time += timedelta(hours=1)
        res[monitor_name]["wdspd"] = np.array(cur_wdspd)
        res[monitor_name]["wddir"] = np.array(cur_wddir)
        res[monitor_name]["time"] = cur_time
    return res


def hourly_u_v_sigma(met_data_dict, select_time):
    # calculate std(u) and std(v)
    wdspd, wddir = [], []
    for monitor_name in met_data_dict.keys():
        time_array = met_data_dict[monitor_name]["time"]
        if select_time in time_array:
            time_idx = time_array.index(select_time)
            wdspd.append(met_data_dict[monitor_name]["wdspd"][time_idx])
            wddir.append(met_data_dict[monitor_name]["wddir"][time_idx])
    u_bias, v_bias = wind_2_uv(np.array(wdspd), np.array(wddir))
    sigma_u = np.std(u_bias)
    sigma_v = np.std(v_bias)
    return sigma_u, sigma_v


def calculate_uv_sigma(met_data_dict, select_date):
    time_array, u_sigma, v_sigma = [select_date + timedelta(hours=t) for t in range(0, 24)], [], []
    for select_time in time_array:
        cur_u_sigma, cur_v_sigma = hourly_u_v_sigma(met_data_dict, select_time)
        u_sigma.append(cur_u_sigma)
        v_sigma.append(cur_v_sigma)
    return {"time": time_array, "u_sigma": u_sigma, "v_sigma": v_sigma}


def agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series):
    traj = {"start_time": [], "start_points": [], "end_time": [], "end_points": []}
    backward_res = {"time": [], "x": [], "y": [], "dist": []}
    x_traj, y_traj = [monitor_x], [monitor_y]
    distances = []
    transport_distance_array = [0]
    traj_idx = 0
    for cur_time_idx in range(len(sampling_time) - 2, -1, -1):
        start_point = (x_traj[traj_idx], y_traj[traj_idx])
        traj["start_points"].append(start_point)
        traj["start_time"].append(sampling_time[cur_time_idx + 1])
        cur_u, cur_v = cur_u_series[cur_time_idx] + 0.00001, cur_v_series[cur_time_idx] + 0.00001
        # # TODO: check this, original I use cur_time_idx
        # cur_u, cur_v = cur_u_series[cur_time_idx + 1] + 0.00001, cur_v_series[cur_time_idx + 1] + 0.00001
        end_point = start_point[0] - 3600 * cur_u, start_point[1] - 3600 * cur_v
        traj["end_time"].append(sampling_time[cur_time_idx])
        traj["end_points"].append(end_point)
        x_traj.append(end_point[0])
        y_traj.append(end_point[1])
        d, x_d, y_d = point_to_line_distance(start_point[0], start_point[1], end_point[0], end_point[1], unit_x, unit_y)
        # (x_d, y_d) to start_point
        cur_transport_dist = np.sqrt((start_point[0] - x_d) ** 2 + (start_point[1] - y_d) ** 2)
        transport_distance = transport_distance_array[-1] + cur_transport_dist
        ratio = np.sqrt((start_point[0] - x_d) ** 2 + (start_point[1] - y_d) ** 2) / np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)
        intercept_time = sampling_time[cur_time_idx + 1] + (sampling_time[cur_time_idx] - sampling_time[cur_time_idx + 1]) * ratio
        distances.append((d, x_d, y_d, intercept_time, transport_distance, len(traj["start_time"])))
        traj_idx += 1

    # select the minimum distance one, the intersection point is (x_d, y_d)
    distances.sort()
    nearest_point = distances[0]
    d, source_x, source_y, d_time, trans_dist, f_idx = nearest_point[0], nearest_point[1], nearest_point[2], nearest_point[3], nearest_point[4], nearest_point[5]
    # forward time duration
    forward_duration = traj["start_time"][0] - d_time
    # get backward_res
    for i in range(0, f_idx):
        backward_res["time"].append(traj["start_time"][i])
        backward_res["x"].append(traj["start_points"][i][0])
        backward_res["y"].append(traj["start_points"][i][1])
    backward_res["dist"] = (source_x, source_y)
    trans_dist = calculate_transport_dist(backward_res)
    return d_time, forward_duration, source_x, source_y, trans_dist


def floor_round_time(datetime_obj):
    number_20 = timedelta(minutes=20 * (datetime_obj.minute // 20))
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def neareast_round_time(datetime_obj):
    number_20 = timedelta(minutes=20 * np.round(datetime_obj.minute / 20))
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def ceil_round_time(datetime_obj):
    if datetime_obj == floor_round_time(datetime_obj):
        return datetime_obj
    else:
        number_20 = timedelta(minutes=20 * ((datetime_obj.minute // 20) + 1))
        return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day, datetime_obj.hour) + number_20


def agent_based_relocate_idx(wind_obs, sigma_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    np.random.seed(0)
    sampling_size = 1000
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    sampling_x_idx, sampling_y_idx = [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            mean = [current_u_obs[i], current_v_obs[i]]
            sigma_idx = sigma_obs["time"].index(wind_obs["time"][i])
            cov = [[sigma_obs["u_sigma"][sigma_idx] ** 2, 0], [0, sigma_obs["v_sigma"][sigma_idx] ** 2]]
            cur_samping_u, cur_sampling_v = np.random.multivariate_normal(mean, cov, sampling_size).T
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    for i in range(0, sampling_size):
        cur_u_series, cur_v_series = sampling_u[:, i], sampling_v[:, i]
        d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
        # forward the none integer part, wind use previous time step wind
        round_d_time = floor_round_time(d_time)
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_idx = model_info["time"].index(round_d_time)
        cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
        transport_seconds = (ceil_round_time(d_time) - d_time).seconds
        forward_x += cur_model_u * transport_seconds
        forward_y += cur_model_v * transport_seconds
        # then, treat the integer part
        steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
        cur_forward_time = d_time + timedelta(seconds=transport_seconds)
        cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
        for j in range(cur_idx, cur_idx + steps):
            x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
            cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
            forward_x += cur_model_u * 1200
            forward_y += cur_model_v * 1200
        forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        sampling_x_idx.append(forward_x_idx)
        sampling_y_idx.append(forward_y_idx)
    return sampling_x_idx, sampling_y_idx


def agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)

    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    # print("distance post %.2f" % np.sqrt((forward_x - unit_x) ** 2 + (forward_y - unit_y) ** 2))
    # forward the none integer part, wind use previous time step wind
    round_d_time = floor_round_time(d_time)
    x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    cur_idx = model_info["time"].index(round_d_time)
    cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
    transport_seconds = (ceil_round_time(d_time) - d_time).seconds
    forward_x += cur_model_u * transport_seconds
    forward_y += cur_model_v * transport_seconds
    # then, treat the integer part
    steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
    cur_forward_time = d_time + timedelta(seconds=transport_seconds)
    cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
    for j in range(cur_idx, cur_idx + steps):
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
        forward_x += cur_model_u * 1200
        forward_y += cur_model_v * 1200
    forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    return forward_x_idx, forward_y_idx


def equal_time_trajectory(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(wrf_sfire_ds["tr17_2"][current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx = agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                # agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
                x_idx, y_idx = agent_based_relocate_idx_mean(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


def dist_based_relocate_idx(wind_obs, sigma_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    np.random.seed(0)
    sampling_size = 1000
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            mean = [current_u_obs[i], current_v_obs[i]]
            sigma_idx = sigma_obs["time"].index(wind_obs["time"][i])
            cov = [[sigma_obs["u_sigma"][sigma_idx] ** 2, 0], [0, sigma_obs["v_sigma"][sigma_idx] ** 2]]
            cur_samping_u, cur_sampling_v = np.random.multivariate_normal(mean, cov, sampling_size).T
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    sampling_x_idx, sampling_y_idx, sampling_t_idx = [], [], []
    for i in range(0, sampling_size):
        cur_u_series, cur_v_series = sampling_u[:, i], sampling_v[:, i]
        d_time, forward_duration, forward_x, forward_y, target_dist = agent_based_backward_loc(monitor_x, monitor_y,
                                                                                               unit_x, unit_y,
                                                                                               sampling_time,
                                                                                               cur_u_series,
                                                                                               cur_v_series)
        round_d_time = floor_round_time(d_time)
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_idx = model_info["time"].index(round_d_time)
        cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
        transport_seconds = (ceil_round_time(d_time) - d_time).seconds
        delta_x, delta_y = cur_model_u * transport_seconds, cur_model_v * transport_seconds
        final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
        if target_dist < np.sqrt(delta_x ** 2 + delta_y ** 2):
            delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
            forward_x += delta_t * cur_model_u
            forward_y += delta_t * cur_model_v
            final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
            target_dist = 0
            d_time += timedelta(seconds=delta_t)
        else:
            forward_x += cur_model_u * transport_seconds
            forward_y += cur_model_v * transport_seconds
            target_dist = target_dist - np.sqrt((cur_model_u * transport_seconds) ** 2 + (cur_model_v * transport_seconds) ** 2)
            d_time = d_time + timedelta(seconds=transport_seconds)
            cur_idx = model_info["time"].index(neareast_round_time(d_time))
            for j in range(cur_idx, len(model_info["time"]) - 1):
                x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
                cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
                cur_delta_x, cur_delta_y = cur_model_u * 1200, cur_model_v * 1200
                if target_dist <= np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2):
                    delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
                    forward_x += delta_t * cur_model_u
                    forward_y += delta_t * cur_model_v
                    d_time += timedelta(seconds=delta_t)
                    final_forward_x, final_forward_y = forward_x, forward_y
                    target_dist = 0
                    break
                else:
                    forward_x += cur_delta_x
                    forward_y += cur_delta_y
                    final_forward_x, final_forward_y = forward_x, forward_y
                    target_dist = target_dist - np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2)
                    d_time += timedelta(seconds=20 * 60)
        final_t_idx = model_info["time"].index(neareast_round_time(d_time))
        forward_x_idx, forward_y_idx = get_model_coord_idx(final_forward_x, final_forward_y, model_info)
        sampling_x_idx.append(forward_x_idx)
        sampling_y_idx.append(forward_y_idx)
        sampling_t_idx.append(final_t_idx)
    # should I do concentration interpolation?
    return sampling_t_idx, sampling_x_idx, sampling_y_idx


def dist_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)

    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    d_time, forward_duration, forward_x, forward_y, target_dist = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    round_d_time = floor_round_time(d_time)
    x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    cur_idx = model_info["time"].index(round_d_time)
    cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
    transport_seconds = (ceil_round_time(d_time) - d_time).seconds
    delta_x, delta_y = cur_model_u * transport_seconds, cur_model_v * transport_seconds
    final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
    if target_dist < np.sqrt(delta_x ** 2 + delta_y ** 2):
        delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
        forward_x += delta_t * cur_model_u
        forward_y += delta_t * cur_model_v
        final_t_idx, final_forward_x, final_forward_y = cur_idx, forward_x, forward_y
        target_dist = 0
        d_time += timedelta(seconds=delta_t)
    else:
        forward_x += cur_model_u * transport_seconds
        forward_y += cur_model_v * transport_seconds
        target_dist = target_dist - np.sqrt((cur_model_u * transport_seconds) ** 2 + (cur_model_v * transport_seconds) ** 2)
        d_time = d_time + timedelta(seconds=transport_seconds)
        cur_idx = model_info["time"].index(neareast_round_time(d_time))
        for j in range(cur_idx, len(model_info["time"]) - 1):
            x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
            cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
            cur_delta_x, cur_delta_y = cur_model_u * 1200, cur_model_v * 1200
            if target_dist <= np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2):
                delta_t = target_dist / np.sqrt(cur_model_u ** 2 + cur_model_v ** 2)
                forward_x += delta_t * cur_model_u
                forward_y += delta_t * cur_model_v
                d_time += timedelta(seconds=delta_t)
                final_forward_x, final_forward_y = forward_x, forward_y
                target_dist = 0
                break
            else:
                forward_x += cur_delta_x
                forward_y += cur_delta_y
                final_forward_x, final_forward_y = forward_x, forward_y
                target_dist = target_dist - np.sqrt(cur_delta_x ** 2 + cur_delta_y ** 2)
                d_time += timedelta(seconds=20 * 60)
    final_t_idx = model_info["time"].index(neareast_round_time(d_time))
    forward_x_idx, forward_y_idx = get_model_coord_idx(final_forward_x, final_forward_y, model_info)
    return final_t_idx, forward_x_idx, forward_y_idx


def equal_dist_trajectory(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        surface_model_conc = np.squeeze(wrf_sfire_ds["tr17_2"][:, 0, :, :])
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = surface_model_conc[i, x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                t_idx, x_idx, y_idx = dist_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([surface_model_conc[t_idx[idx], x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                t_idx, x_idx, y_idx = dist_based_relocate_idx_mean(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(surface_model_conc[t_idx, x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


def equal_time_trajectory_CTM(wrf_ds, ctm_ds, met_df, fire_obj, monitor_info):
    model_info = CMAQGridInfo(ctm_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(ctm_ds["PM25_TOT"][current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx = agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                x_idx, y_idx = agent_based_relocate_idx_mean(current_wind_obs, wrf_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


# TODO: New Algorithm TESTING
def agent_based_relocate_idx_mean_adv(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)

    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    # prior
    d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    delta_u, delta_v = (forward_x - unit_x) / forward_duration.seconds, (forward_y - unit_y) / forward_duration.seconds
    cur_u_series, cur_v_series = cur_u_series - delta_u, cur_v_series - delta_v
    d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    # forward the none integer part, wind use previous time step wind
    round_d_time = floor_round_time(d_time)
    x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    cur_idx = model_info["time"].index(round_d_time)
    cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
    transport_seconds = (ceil_round_time(d_time) - d_time).seconds
    forward_x += cur_model_u * transport_seconds
    forward_y += cur_model_v * transport_seconds
    # then, treat the integer part
    steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
    cur_forward_time = d_time + timedelta(seconds=transport_seconds)
    cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
    for j in range(cur_idx, cur_idx + steps):
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
        forward_x += cur_model_u * 1200
        forward_y += cur_model_v * 1200
    forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
    return forward_x_idx, forward_y_idx, delta_u, delta_v


def adv_agent_based_relocate_idx(wind_obs, sigma_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info, delta_u, delta_v):
    np.random.seed(0)
    sampling_size = 1000
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    sampling_x_idx, sampling_y_idx = [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            mean = [current_u_obs[i], current_v_obs[i]]
            sigma_idx = sigma_obs["time"].index(wind_obs["time"][i])
            cov = [[sigma_obs["u_sigma"][sigma_idx] ** 2, 0], [0, sigma_obs["v_sigma"][sigma_idx] ** 2]]
            cur_samping_u, cur_sampling_v = np.random.multivariate_normal(mean, cov, sampling_size).T
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    for i in range(0, sampling_size):
        cur_u_series, cur_v_series = sampling_u[:, i] - delta_u, sampling_v[:, i] - delta_v
        d_time, forward_duration, forward_x, forward_y, _ = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
        # forward the none integer part, wind use previous time step wind
        round_d_time = floor_round_time(d_time)
        x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        cur_idx = model_info["time"].index(round_d_time)
        cur_model_u, cur_model_v = model_ds["U10"][:][cur_idx, x_idx, y_idx][0], model_ds["V10"][:][cur_idx, x_idx, y_idx][0]
        transport_seconds = (ceil_round_time(d_time) - d_time).seconds
        forward_x += cur_model_u * transport_seconds
        forward_y += cur_model_v * transport_seconds
        # then, treat the integer part
        steps = int((forward_duration - timedelta(seconds=transport_seconds)).seconds / (20 * 60))
        cur_forward_time = d_time + timedelta(seconds=transport_seconds)
        cur_idx = model_info["time"].index(neareast_round_time(cur_forward_time))
        for j in range(cur_idx, cur_idx + steps):
            x_idx, y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
            cur_model_u, cur_model_v = model_ds["U10"][:][j, x_idx, y_idx][0], model_ds["V10"][:][j, x_idx, y_idx][0]
            forward_x += cur_model_u * 1200
            forward_y += cur_model_v * 1200
        forward_x_idx, forward_y_idx = get_model_coord_idx(forward_x, forward_y, model_info)
        sampling_x_idx.append(forward_x_idx)
        sampling_y_idx.append(forward_y_idx)
    return sampling_x_idx, sampling_y_idx


def equal_time_trajectory_adv(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = obs_met_data[assigned_met_monitor]
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(wrf_sfire_ds["tr17_2"][current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx, delta_u, delta_v = agent_based_relocate_idx_mean_adv(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])
                x_idx, y_idx = adv_agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info, delta_u, delta_v)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


# TODO: Least square method
def agent_based_relocate_prior(wind_obs, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time):
    # backward traj
    current_u_obs, current_v_obs = wind_2_uv(wind_obs["wdspd"], wind_obs["wddir"])
    sampling_time, sampling_u, sampling_v = [], [], []
    for i in range(0, len(wind_obs["time"])):
        if start_time - timedelta(hours=1) <= wind_obs["time"][i] <= end_time:
            cur_samping_u, cur_sampling_v = current_u_obs[i], current_v_obs[i]
            sampling_u.append(cur_samping_u)
            sampling_v.append(cur_sampling_v)
            sampling_time.append(wind_obs["time"][i])
    sampling_u, sampling_v = np.array(sampling_u), np.array(sampling_v)
    cur_u_series, cur_v_series = sampling_u[:], sampling_v[:]
    d_time, forward_duration, forward_x, forward_y, trans_dist = agent_based_backward_loc(monitor_x, monitor_y, unit_x, unit_y, sampling_time, cur_u_series, cur_v_series)
    target_delta_u, target_delta_v = (forward_x - unit_x) / forward_duration.seconds, (forward_y - unit_y) / forward_duration.seconds
    # print("distance prior %.2f" % np.sqrt((forward_x - unit_x) ** 2 + (forward_y - unit_y) ** 2))
    trans_time, ratio = [], []
    trans_start_time = d_time + forward_duration - timedelta(hours=1)
    while forward_duration.seconds > 3600:
        ratio.append(1)
        trans_time.append(trans_start_time)
        trans_start_time -= timedelta(hours=1)
        forward_duration -= timedelta(seconds=3600)
    ratio.append(forward_duration.seconds / 3600)
    trans_time.append(trans_start_time)
    return trans_time, ratio, target_delta_u, target_delta_v


def equal_time_trajectory_adv_lsq(wrf_sfire_ds, met_df, fire_obj, monitor_info):
    model_info = WRFGridInfo(wrf_sfire_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}
    smoke_conc = smoke_concentration("tr17_2", wrf_sfire_ds)
    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = copy.deepcopy(obs_met_data[assigned_met_monitor])
        # for prior steps
        trans_time, ratio, target_delta_u, target_delta_v = [], [], [], []
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(smoke_conc[current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                cur_trans_time, cur_ratio, cur_target_delta_u, cur_target_delta_v = agent_based_relocate_prior(current_wind_obs, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time)
                trans_time.append(cur_trans_time)
                ratio.append(cur_ratio)
                target_delta_u.append(cur_target_delta_u)
                target_delta_v.append(cur_target_delta_v)
        # build matrix
        flattened_list = [dt for sublist in trans_time for dt in sublist]
        # Find min and max
        min_time, max_time = min(flattened_list), max(flattened_list)
        t_array = []
        current_time = min_time
        while current_time <= max_time:
            t_array.append(current_time)
            current_time += timedelta(hours=1)  # Increment by 1 hour
        A = np.zeros((len(target_delta_u), len(t_array)))
        for i in range(0, len(trans_time)):
            for j in range(0, len(trans_time[i])):
                cur_col = t_array.index(trans_time[i][j])
                A[i][cur_col] = ratio[i][j]
        target_delta_u, target_delta_v = np.array(target_delta_u), np.array(target_delta_v)
        # solve delta_u, delta_v
        delta_u, delta_v = np.linalg.solve(A, target_delta_u), np.linalg.solve(A, target_delta_v)
        # convert obs data
        for i in range(0, len(t_array)):
            cur_idx = current_wind_obs["time"].index(t_array[i])
            cur_spd, cur_dir = current_wind_obs["wdspd"][cur_idx], current_wind_obs["wddir"][cur_idx]
            cur_delta_u, cur_delta_v = delta_u[i], delta_v[i]
            current_u_obs, current_v_obs = wind_2_uv(cur_spd, cur_dir)
            current_u_obs, current_v_obs = current_u_obs + cur_delta_u, current_v_obs + cur_delta_v
            cur_spd, cur_dir = uv_2_wind(current_u_obs, current_v_obs)
            current_wind_obs["wdspd"][cur_idx], current_wind_obs["wddir"][cur_idx] = cur_spd, cur_dir

        # post estimations
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(smoke_conc[current_time_idx, 0, :, :])
            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx = agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                # agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
                x_idx, y_idx = agent_based_relocate_idx_mean(current_wind_obs, wrf_sfire_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res


def equal_time_trajectory_adv_lsq_CTM(wrf_ds, ctm_ds, met_df, fire_obj, monitor_info):
    model_info = CMAQGridInfo(ctm_ds)
    fire_date = datetime(fire_obj["start_time"].year, fire_obj["start_time"].month, fire_obj["start_time"].day)
    obs_met_data = hourly_met_data(met_df, fire_date)
    obs_sigma = calculate_uv_sigma(obs_met_data, fire_date)
    unit_x, unit_y = model_info["crs"](fire_obj["coord"][0], fire_obj["coord"][1])
    uncertainty_res = {}

    for conc_monitor_name in monitor_info.keys():
        uncertainty_res[conc_monitor_name] = {}
        lower_quantile, upper_quantile, mean_conc = [], [], []
        assigned_met_monitor = monitor_info[conc_monitor_name]["met"]
        conc_monitor_lon, conc_monitor_lat = monitor_info[conc_monitor_name]["coord"][0], monitor_info[conc_monitor_name]["coord"][1]
        conc_monitor_x, conc_monitor_y = model_info["crs"](conc_monitor_lon, conc_monitor_lat)
        # get current met
        current_wind_obs = copy.deepcopy(obs_met_data[assigned_met_monitor])
        # for prior steps
        trans_time, ratio, target_delta_u, target_delta_v = [], [], [], []
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(ctm_ds["PM25_TOT"][current_time_idx, 0, :, :])

            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                cur_trans_time, cur_ratio, cur_target_delta_u, cur_target_delta_v = agent_based_relocate_prior(current_wind_obs, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time)
                trans_time.append(cur_trans_time)
                ratio.append(cur_ratio)
                target_delta_u.append(cur_target_delta_u)
                target_delta_v.append(cur_target_delta_v)
        # build matrix
        flattened_list = [dt for sublist in trans_time for dt in sublist]
        # Find min and max
        min_time, max_time = min(flattened_list), max(flattened_list)
        t_array = []
        current_time = min_time
        while current_time <= max_time:
            t_array.append(current_time)
            current_time += timedelta(hours=1)  # Increment by 1 hour
        A = np.zeros((len(target_delta_u), len(t_array)))
        for i in range(0, len(trans_time)):
            for j in range(0, len(trans_time[i])):
                cur_col = t_array.index(trans_time[i][j])
                A[i][cur_col] = ratio[i][j]
        target_delta_u, target_delta_v = np.array(target_delta_u), np.array(target_delta_v)
        # solve delta_u, delta_v
        delta_u, delta_v = np.linalg.solve(A, target_delta_u), np.linalg.solve(A, target_delta_v)
        # convert obs data
        for i in range(0, len(t_array)):
            cur_idx = current_wind_obs["time"].index(t_array[i])
            cur_spd, cur_dir = current_wind_obs["wdspd"][cur_idx], current_wind_obs["wddir"][cur_idx]
            cur_delta_u, cur_delta_v = delta_u[i], delta_v[i]
            current_u_obs, current_v_obs = wind_2_uv(cur_spd, cur_dir)
            current_u_obs, current_v_obs = current_u_obs + cur_delta_u, current_v_obs + cur_delta_v
            cur_spd, cur_dir = uv_2_wind(current_u_obs, current_v_obs)
            current_wind_obs["wdspd"][cur_idx], current_wind_obs["wddir"][cur_idx] = cur_spd, cur_dir

        # post estimations
        for i in range(0, len(current_wind_obs["time"])):
            current_time_idx = model_info["time"].index(current_wind_obs["time"][i])
            current_model_conc = np.squeeze(ctm_ds["PM25_TOT"][current_time_idx, 0, :, :])
            if current_wind_obs["time"][i] < fire_obj["start_time"]:
                x_idx, y_idx = get_model_coord_idx(conc_monitor_x, conc_monitor_y, model_info)
                conc_val = current_model_conc[x_idx, y_idx][0]
                lower_quantile.append(conc_val)
                upper_quantile.append(conc_val)
                mean_conc.append(conc_val)
            else:
                traj_start_time, traj_end_time = fire_obj["start_time"], current_wind_obs["time"][i]
                x_idx, y_idx = agent_based_relocate_idx(current_wind_obs, obs_sigma, wrf_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                # calculate correct concentration based x_idx, y_idx
                sampling_conc = np.array([current_model_conc[x_idx[idx], y_idx[idx]] for idx in range(0, len(x_idx))])
                lower_quantile.append(np.quantile(sampling_conc, 0.16))
                upper_quantile.append(np.quantile(sampling_conc, 0.84))
                # calculate the mean concentration
                # agent_based_relocate_idx_mean(wind_obs, model_ds, monitor_x, monitor_y, unit_x, unit_y, start_time, end_time, model_info):
                x_idx, y_idx = agent_based_relocate_idx_mean(current_wind_obs, wrf_ds, conc_monitor_x, conc_monitor_y, unit_x, unit_y, traj_start_time, traj_end_time, model_info)
                mean_conc.append(current_model_conc[x_idx, y_idx][0])

        uncertainty_res[conc_monitor_name]["time"] = current_wind_obs["time"]
        uncertainty_res[conc_monitor_name]["lower"] = lower_quantile
        uncertainty_res[conc_monitor_name]["upper"] = upper_quantile
        uncertainty_res[conc_monitor_name]["mean"] = mean_conc
    return uncertainty_res
