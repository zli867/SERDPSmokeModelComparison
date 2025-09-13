import pyproj
import numpy as np
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import fiona
from shapely.geometry import shape

def get_crs(ds):
    projection = ds.GetProjection()
    crs = pyproj.Proj(projection)
    return crs


def read_sensing_info(ds):
    geotransform = ds.GetGeoTransform()
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    x_corner_range = np.linspace(0, ncols - 1, ncols)
    y_corner_range = np.linspace(0, nrows - 1, nrows)
    x, y = np.meshgrid(x_corner_range, y_corner_range)
    X = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    Y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    # shift to center
    X += geotransform[1] / 2
    Y += geotransform[4] / 2
    crs = get_crs(ds)
    info_dict = {
        "X": X,
        "Y": Y,
        "crs": crs
    }
    return info_dict


def extract_wrf_time_array(ds):
    wrf_time_array = []
    wrf_time = ds["Times"][:]
    for i in range(0, wrf_time.shape[0]):
        current_time_str = ""
        for j in range(0, wrf_time.shape[1]):
            current_time_str = current_time_str + wrf_time[i][j].decode()
        current_time_obj = datetime.strptime(current_time_str, '%Y-%m-%d_%H:%M:%S')
        wrf_time_array.append(current_time_obj)
    return wrf_time_array


def findSpatialIndex(fire_x, fire_y, X_ctr, Y_ctr):
    """

    :param fire_x: X of fire location in CMAQ projection
    :param fire_y: Y of fire location in CMAQ projection
    :param X_ctr: CMAQ grid X center
    :param Y_ctr: CMAQ grid Y center
    :return: x_idx, y_idx which are the fire location in CMAQ grid
    """
    dist = np.sqrt((X_ctr - fire_x) ** 2 + (Y_ctr - fire_y) ** 2)
    x_idx, y_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return x_idx, y_idx


def plotPolygons(polygon_list, ax, color, crs=None, linewidth=1.0):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                if crs is not None:
                    xs, ys = crs(xs, ys)
                ax.plot(xs, ys, color, linewidth=linewidth)

        else:
            xs, ys = current_polygon.exterior.xy
            if crs is not None:
                xs, ys = crs(xs, ys)
            ax.plot(xs, ys, color, linewidth=linewidth)


def discrete_cmap(invertals, base_color_scheme="viridis"):
    cmap = plt.get_cmap(base_color_scheme, invertals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # cmap.set_over(color=high_value_color, alpha=1.0)
    return cmap


def anderson_category(fuel_cat_num):
    # fuel_types = ['Short grass (1 ft)', 'Timber (grass and understory)',
    #               'Tall grass (2.5 ft)', 'Chaparral (6 ft)',
    #               'Brush (2 ft)', 'Dormant brush, hardwood slash',
    #               'Southern rough', 'Closed timber litter',
    #               'Hardwood litter', 'Timber (litter + understory)',
    #               'Light logging slash', 'Medium logging slash',
    #               'Heavy logging slash']
    fuel_types = ['Short grass', 'Grass with Timber/\nShrub Overstory',
                  'Tall grass', 'Chaparral',
                  'Brush', 'Dormant brush, hardwood slash',
                  'Southern rough', 'Closed, Short Needle\nTimber Litter',
                  'Hardwood/Long Needle Pine\nTimber Litter', 'Mature/Overmature Timber\nand Understory',
                  'Light slash', 'Medium slash',
                  'Heavy slash']
    return fuel_types[fuel_cat_num - 1]


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
                "time": cmaq_time_array, "Lat": lat_ctr, "Lon": lon_ctr, "Lat_bdry": [lat_min, lat_max], "Lon_bdry": [lon_min, lon_max]}
    return res_dict


def discrete_conc_cmap(invertals, base_color_scheme="Spectral_r", low_value_color='white', high_value_color='purple'):
    cmap = plt.get_cmap(base_color_scheme, invertals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = colors.to_rgba(low_value_color)
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    cmap.set_over(color=high_value_color, alpha=1.0)
    # cmap.set_under(color=low_value_color, alpha=1.0)
    return cmap


# def get_conf_intercept(alpha, lr, X, y):
#     """
#     Returns (1-alpha) 2-sided confidence intervals
#     for sklearn.LinearRegression coefficients
#     as a pandas DataFrame
#     """
#     coefs = np.r_[[lr.intercept_], lr.coef_]
#     X_aux = np.zeros((X.shape[0], X.shape[1] + 1))
#     X_aux[:, 1:] = X
#     X_aux[:, 0] = 1
#     dof = -np.diff(X_aux.shape)[0]
#     mse = np.sum((y - lr.predict(X)) ** 2) / dof
#     var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
#     t_val = stats.t.isf(alpha / 2, dof)
#     gap = t_val * np.sqrt(mse * var_params)
#     return {
#         "coeffs": coefs,
#         "lower": coefs - gap,
#         "upper": coefs + gap
#     }
#
#
def plotComparisonIntercept(X, Y, ax):
    lin_model = LinearRegression().fit(X, Y)
    line_x = np.linspace(0, np.max(X)).reshape(-1, 1)
    line_y = lin_model.predict(line_x)
    # Confident intervals
    r2_score = lin_model.score(X, Y)
    ax.plot(line_x, line_y)
    print(lin_model.coef_)
    print(lin_model.intercept_)
    ax.legend(loc='lower right')


def plotComparisonNoIntercept(X, Y, ax):
    lin_model = sm.OLS(Y, X, hasconst=False)
    results = lin_model.fit()
    line_x = np.linspace(0, np.max(X)).reshape(-1, 1)
    # Confident intervals
    cf = results.conf_int(alpha=0.05, cols=None)
    reg = LinearRegression(fit_intercept=False).fit(X, Y)
    line_y = reg.predict(line_x)
    correlation_coefficient, _ = stats.pearsonr(X.flatten(), Y.flatten())
    performance = 'y={:.2f}x \n slope: [{:.2f}, {:.2f}] \n r = {:.2f}'.format(results.params[0], cf[0, 0], cf[0, 1],
                                                                              correlation_coefficient)
    ax.plot(line_x, line_y, 'r', label=performance)
    ax.legend(loc='upper left', frameon=False)


def plotComparisonNoInterceptSimple(X, Y, ax, color):
    lin_model = sm.OLS(Y, X, hasconst=False)
    results = lin_model.fit()
    line_x = np.linspace(0, np.max(X)).reshape(-1, 1)
    # Confident intervals
    cf = results.conf_int(alpha=0.05, cols=None)
    reg = LinearRegression(fit_intercept=False).fit(X, Y)
    line_y = reg.predict(line_x)
    correlation_coefficient, _ = stats.pearsonr(X.flatten(), Y.flatten())
    performance = 'y={:.2f}x, r = {:.2f}'.format(results.params[0], correlation_coefficient)
    ax.plot(line_x, line_y, color, label=performance)



def smoke_concentration(tracer_name, d):
    # https://github.com/openwfm/wrfxpy/blob/335669bab63cbca88e827f288c11d875124559cc/src/vis/vis_utils.py#L149
    P = d.variables['P'][:] + d.variables['PB'][:]
    T00 = d.variables['T00'][:]  # Load the full array first
    T00 = T00[:, np.newaxis, np.newaxis, np.newaxis]  # Then reshape
    T = d.variables['T'][:] + T00  # temperature (K)
    r_d = 287  # specific gas constant (J/kg/K)
    rho = P / (r_d * T)  # dry air density  (kg/m^3)
    s = d.variables[tracer_name][:, :, :, :] * rho
    return s


def StatePolygon(state_name_list):
    """
    List of States:
        Maryland, Iowa, Delaware, Ohio, Pennsylvania, Nebraska, Washington, Puerto Rico, Alabama,
        Arkansas, New Mexico, Texas, California, Kentucky, Georgia, Wisconsin, Oregon,
        Missouri, Virginia, Tennessee, Louisiana, New York, Michigan, Idaho, Florida,
        Alaska, Illinois, Montana, Minnesota, Indiana, Massachusetts, Kansas, Nevada,
        Vermont, Connecticut, New Jersey, District of Columbia, North Carolina, Utah,
        North Dakota, South Carolina, Mississippi, Colorado, South Dakota, Oklahoma,
        Wyoming, West Virginia, Maine, Hawaii, New Hampshire, Arizona, Rhode Island
    :param state_name: input the state name in the list
    :return: polygon of the state
    """
    us_shape_name = "/Volumes/Shield/HealthAnalysis/data/geo/US/cb_2018_us_state_20m.shp"
    us_states = fiona.open(us_shape_name)
    state_geo = []
    for state_name in state_name_list:
        for us_state in us_states:
            cur_name = us_state['properties']['NAME']
            if cur_name == state_name:
                state_geo.append(shape(us_state['geometry']))
    return state_geo


def fillPolygons(polygon_list, ax, color, crs=None, linewidth=1.0):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                if crs is not None:
                    xs, ys = crs(xs, ys)
                ax.fill(xs, ys, color, linewidth=linewidth)

        else:
            xs, ys = current_polygon.exterior.xy
            if crs is not None:
                xs, ys = crs(xs, ys)
            ax.fill(xs, ys, color, linewidth=linewidth)