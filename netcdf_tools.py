from collections import OrderedDict
import logging
import numbers

import numpy as np
import netCDF4
from netCDF4 import Dataset

from ccfepyutils.utils import is_scalar, to_array

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_netcdf_atrribute(group, name, value):
    """Set NetCDF group attirbute safely replacing None values with np.nan"""
    if value is None:
        value = np.nan
    elif isinstance(value, (np.ndarray, list, tuple)) and len(value) > 0 and isinstance(value[0], str):
        value = np.array(value).astype(np.str_)
    setattr(group, name, value)

def add_netcdf_standalone_variable(group, name, value):
    value = to_array(value)  # make sure iterable and np array string format etc
    if name not in group.dimensions:
        group.createDimension(name, len(value))
    if name not in group.variables:
        format_nc = get_netcdf_format(value)
        var = group.createVariable(name, format_nc, name)
    var = group.variables[name]
    var[:] = value

def get_netcdf_atrribute(group, name):
    """Set NetCDF group attirbute safely replacing None values with np.nan"""
    value = group.__dict__[name]
    if value is np.nan:
        value = None
    return value

def get_netcdf_variable(group, name):
    return group[name][:]

def dict_to_netcdf(root, name, dictionary):
    """Save values in dict to attributes in group"""
    assert isinstance(dictionary, dict)
    grp = root.createGroup(name)
    for key, value in dictionary.items():
        add_netcdf_standalone_variable(grp, key, value)
        # set_netcdf_atrribute(grp, key, value)

def netcdf_to_dict(group, name):
    """Load attributes in group to dict"""
    out = OrderedDict()
    for key in group.__dict__:
        out[key] = get_netcdf_variable(group, key)
        # out[key] = get_netcdf_atrribute(group, key)
    return out

def dataframe_to_netcdf(root, name, df):
    """Write dataframe to netcdf file avoiding buggy df.to_xarray.to_netcdf()"""
    group = root.createGroup(name)
    str_arrays_grp = group.createGroup('str_arrays')
    index = df.index
    index_name = index.name if index.name is not None else 'index'
    # Save index values
    index = index.values
    index_dim = group.createDimension(index_name, len(index))

    cd_format = get_netcdf_format(index)
    if 'S' in cd_format:  # Store strings as attributes
        set_netcdf_atrribute(str_arrays_grp, index_name, index)
    else:
        index_var = group.createVariable(index_name, cd_format, index_name)
        index_var[:] = index
    group.index = index_name

    # str_out = netCDF4.stringtochar(index.astype(s_format))
    # index_var[:] = str_out

    for col in df.columns:
        values = df[col].values
        cd_format = get_netcdf_format(values)
        if 'S' in cd_format:  # Store strings as attributes
            set_netcdf_atrribute(str_arrays_grp, col, values)
        else:  # Store numbers as variables
            var = group.createVariable(col, get_netcdf_format(values), (index_name,))
            var[:] = values

def get_netcdf_format(variable):
    if len(variable) == 0:
        logger.warning('empty array, dont know type')
        return 'f8'
    if isinstance(variable[0], str):
        s_len = int(max(map(len, variable)))
        s_format = 'S{:d}'.format(s_len)
        return s_format
    elif isinstance(variable[0], numbers.Integral):
        return 'i4'
    elif isinstance(variable[0], numbers.Real):
        return 'f8'