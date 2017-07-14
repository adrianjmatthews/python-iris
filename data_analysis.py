"""Data analysis module using iris.

Classes that provide iris data i/o wrappers to data analysis methods,
often also using the iris module, for analysis of meteorological and
oceanographic gridded data.

Author: Adrian Matthews

Create documentation with pydoc -w data_analysis

Programming style:

Object oriented!

Class methods typically set an attribute(s) of the class instance.
They typically do not return an argument.

Printed output and information:

The __init__ method of each class has a 'verbose' keyword argument.

verbose=False (or 0) suppresses printed output

verbose=True (or 1) prints limited output (typically a statement that a
particular class attribute has been created)

verbose=2 prints extended output (typically the value of that class
attribute).

"""

# All import statements here, before class definitions
from __future__ import division, print_function, with_statement # So can run this in python2
import datetime
import os.path
import pdb
import random

import gsw # Gibbs TEOS-10 seawater routines
import iris
from iris.time import PartialDateTime
import numpy as np
import scipy
from windspharm.iris import VectorWind

import mypaths

# Headers for use in calls to print
h1a='<<<=============================================================\n'
h1b='=============================================================>>>\n'
h2a='<<<---------------------------\n'
h2b='--------------------------->>>\n'

# var_name, standard_name/long_name pairs
# var_name is used for file names and as the variable name in netcdf files
# When iris extracts a cube from a netcdf file with a given name string, it
#   first looks for the standard_name, then long_name, then var_name.
# However, there is a prescribed list of allowed standard_name values
# 'tendency_of_atmosphere_relative_vorticity' (or any alternative) is not on
# the list, and an error will result if this is set as a standard_name.
# Use the iris.cube.Cube.rename(name) method to set the name of a cube
#   This sets the standard_name attribute if name is a valid standard_name
#   Otherwise it sets the long_name attribute to name.
# ie do not explicitly set the standard_name attribute as this will cause an
#   error if the desired name is not a valid standard_name
# rename method sets var_name to None if name is not a valid standard_name
# So var_name needs to be reset explicitly after rename has been called
# To set the name of cube1 to the name of cube 2, use
#   cube1.rename(cube2.name())
#   cube1.var_name=cube2.var_name
var_name2long_name={
    'bsiso1-1':'BSISO_1-1_index',
    'bsiso1-2':'BSISO_1-2_index',
    'bsiso1_amp':'BSISO_1_amplitude',
    'bsiso1_cat':'BSISO_1_category',
    'bsiso1_theta':'BSISO_1_phase_angle',
    'bsiso2-1':'BSISO_2-1_index',
    'bsiso2-2':'BSISO_2-2_index',
    'bsiso2_amp':'BSISO_2_amplitude',
    'bsiso2_cat':'BSISO_2_category',
    'bsiso2_theta':'BSISO_2_phase_angle',
    'chi':'atmosphere_horizontal_velocity_potential',
    'div':'divergence_of_wind',
    'domegady_duwnddp':'meridional_derivative_of_lagrangian_tendency_of_air_pressure_times_pressure_derivative_of_zonal_wind',
    'dvrtdt':'tendency_of_atmosphere_relative_vorticity',
    'elevation':'height_above_reference_ellipsoid',
    'ew':'water_vapor_partial_pressure_in_air',
    'ke':'specific_kinetic_energy_of_air',
    'lat':'latitude',
    'lhfd':'surface_downward_latent_heat_flux',
    'lon':'longitude',
    'mslp':'air_pressure_at_sea_level',
    'mltt':'ocean_mixed_layer_thickness_defined_by_temperature',
    'm_beta_vwnd':'minus_meridional_derivative_of_coriolis_parameter_times_northward_wind',
    'm_domegadx_dvwnddp':'minus_zonal_derivative_of_lagrangian_tendency_of_air_pressure_times_pressure_derivative_of_northward_wind',
    'm_ff_div':'minus_coriolis_parameter_times_atmosphere_relative_vorticity',
    'm_omega_dvrtdp':'minus_lagrangian_tendency_of_air_pressure_times_pressure_derivative_of_atmosphere_relative_vorticity',
    'm_uwnd_dvrtdx':'minus_eastward_wind_times_zonal_derivative_of_atmosphere_relative_vorticity',
    'm_vwnd_dvrtdy':'minus_northward_wind_times_meridional_derivative_of_atmosphere_relative_vorticity',
    'm_vrt_div':'minus_divergence_of_wind_times_atmosphere_relative_vorticity',
    'olr':'toa_outgoing_longwave_flux',
    'omega':'lagrangian_tendency_of_air_pressure',
    'ppt':'lwe_precipitation_rate',
    'pa':'air_pressure',
    'psfc':'surface_air_pressure',
    'psi':'atmosphere_horizontal_streamfunction',
    'pv':'ertel_potential_vorticity',
    'res_dvrtdt':'residual_tendency_of_atmosphere_relative_vorticity',
    'rhum':'relative_humidity',
    'rmm1':'RMM_1_index',
    'rmm2':'RMM_2_index',
    'rmm_amp':'RMM_amplitude',
    'rmm_cat':'RMM_category',
    'sa':'sea_water_absolute_salinity',
    'shum':'specific_humidity',
    'source_dvrtdt':'total_source_of_tendency_of_atmosphere_relative_vorticity',
    'sst':'sea_surface_temperature',
    'swp':'sea_water_pressure',
    'swpd':'sea_water_potential_density',
    'swr':'surface_downwelling_shortwave_flux',
    'ta':'air_temperature',
    'tsc':'sea_water_conservative_temperature',
    'uwnd':'eastward_wind',
    'vrt':'atmosphere_relative_vorticity',
    'vwnd':'northward_wind',
    'wndspd':'wind_speed',
    'wwnd':'upward_air_velocity',
    'zg':'geopotential_height',
    }

# Create inverse dictionary
def invert_dict(d):
    return dict([(v,k) for k,v in d.items()])
var_long_name2name=invert_dict(var_name2long_name)

#==========================================================================

def source_info(aa):
    """Create attributes of an object based on the source attribute.

    aa is an object.

    Attributes created are:

    data_source:  e.g., 'ncepdoe', 'olrinterp'
    level_type:  e.g., 'plev' for pressure level, 'toa' for top of atmosphere
    frequency: e.g., 'd' for daily, '3' for 3-hourly

    outfile_frequency: e.g., 'year' or 'month'
    wildcard: e.g., '????' or '??????'
    timedelta: datetime.timedelta object corresponding to the frequency attribute
    
    """
    # Split source attribute string using underscores as separators
    xx=aa.source.split('_')
    if len(xx)!=3:
        raise RuntimeError("source attribute '{0.source!s}' must have three parts separated by underscores".format(aa))
    aa.data_source=xx[0]
    aa.level_type=xx[1]
    aa.frequency=xx[2]
    # Check data_source attribute is valid
    valid_data_sources=['erainterim','ncepdoe','ncepncar','olrcdr','olrinterp','sg579m031oi01','sg534m031oi01','sg532m031oi01','sg620m031oi01','sg613m031oi01','sgallm031oi01','sstrey','trmm3b42v7','tropflux']
    if aa.data_source not in valid_data_sources:
        raise ValueError('data_source {0.data_source!s} not valid'.format(aa))
    # Set outfile_frequency attribute depending on source information
    if aa.source in ['erainterim_plev_6h','erainterim_plev_d','ncepdoe_plev_6h','ncepdoe_plev_d','ncepncar_sfc_d','ncepncar_plev_d','olrcdr_toa_d','olrinterp_toa_d','sstrey_sfc_7d','sg579m031oi01_zlev_h','sg534m031oi01_zlev_h','sg532m031oi01_zlev_h','sg620m031oi01_zlev_h','sg613m031oi01_zlev_h','sgallm031oi01_zlev_h','sstrey_sfc_d','tropflux_sfc_d']:
        aa.outfile_frequency='year'
        aa.wildcard='????'
    elif aa.source in ['trmm3b42v7_sfc_3h','trmm3b42v7_sfc_d']:
        aa.outfile_frequency='month'
        aa.wildcard='??????'
    else:
        raise ValueError('Need to specify outfile_frequency for this data source.')
    # timedelta attribute
    if aa.frequency[-1]=='h':
        if aa.frequency=='h':
            aa.timedelta=datetime.timedelta(hours=1)
        else:
            aa.timedelta=datetime.timedelta(hours=int(aa.frequency[:-1]))
    elif aa.frequency[-1]=='d':
        if aa.frequency=='d':
            aa.timedelta=datetime.timedelta(days=1)
        else:
            aa.timedelta=datetime.timedelta(days=int(aa.frequency[:-1]))
    else:
        raise ToDoError('Need to code up for different frequency attribute.')
            
    # Printed output
    if aa.verbose:
        ss=h2a+'source_info.  Created attributes: \n'+\
            'data source: {0.data_source!s} \n'+\
            'level_type: {0.level_type!s} \n'+\
            'frequency: {0.frequency!s} \n'+\
            'outfile_frequency: {0.outfile_frequency!s} \n'+\
            'timedelta: {0.timedelta!s} \n'+\
            'wildcard: {0.wildcard!s} \n'+h2b
        print(ss.format(aa))
                
#==========================================================================

def clean_callback(cube,field,filename):
    """Deletes many attributes on iris load.
    
    Problem.  iris concatenate and merge (to create a single cube from
    a cube list) is very picky and will fail if there are any
    mismatching metadata between the cubes.  This function removes
    attributes from the time coordinate and basic metadata that
    typically fall foul of this.  These attributes are not useful
    anyway.

    Data providers (such as NOAA) sometimes change the attributes
    (trivially) in the middle of their data sets, so these all need to
    be stripped out.

    Ideally, this clean callback will only be used in preprocess.py,
    and all subsequent analysis will not need it.  Hence attributes
    that are built up during analysis will be self-consistent and will
    not need removing.
    
    Usage: as an argument in iris load  (...,callback=clean_callback).
    
    """
    # Delete the problem attributes from the time coordinate:
    # 'actual_range' expected to be different between cubes, so delete
    # 'coordinate_defines' usually set to 'start' or 'point'
    #     'start' indicates that e.g., the 0000 UTC time stamp refers to the
    #         start of the day over which the daily average (eg NCEP data)
    #         has been carried out over.  Unfortunately, NCEP erroneously
    #         change this to 'point' in 2015.  Need to delete it.
    att_list=['actual_range','coordinate_defines']
    for attribute in att_list:
        if attribute in cube.coord('time').attributes:
            del cube.coord('time').attributes[attribute]
    # Or set the attributes dictionary of the time coordinate to empty:
    #cube.coord('time').attributes = {}
    
    # Similarly delete some of the main attributes
    att_list=['actual_range','history','unpacked_valid_range','references',
              'References','dataset_title','title','long_name',
              'Conventions','GRIB_id','GRIB_name','comments','institution',
              'least_significant_digit','level_desc','parent_stat',
              'platform','precision','source','statistic','title','var_desc',
              'NCO','creation_date','invalid_units','metodology',
              'producer_agency','time_range','website','short_name']
    for attribute in att_list:
        if attribute in cube.attributes:
            del cube.attributes[attribute]

    # A cube also has some "attributes" that are not in the attributes dictionary
    # Set these to empty strings
    cube.long_name=''

    # Set cell methods to empty tuple
    cube.cell_methods=()

#==========================================================================

def create_cube(array,oldcube,new_axis=False,new_var_name=False):
    """Create an iris cube from a numpy array and attributes from an old cube.

    Inputs:

    <array> must be a numpy array.

    <oldcube>  must be an iris cube.

    <new_axis> is either False, or an iris cube axis (e.g., a time axis).

    <new_var_name> is either False, or a string variable name, e.g., 'psi'

    Output:

    <newcube> is an iris cube of the same shape as <array>

    Usage:

    If <new_axis> is False then array and oldcube must have the same
    shape and dimensions.  <newcube> is then simply an iris cube
    created from <array> and the coordinate axes and attributes of
    <oldcube>.

    If <new_axis> is an iris cube axis (e.g., a time axis), then
    <array> and <oldcube> must have the same number of dimensions, but
    one of the axes (the one that will correspond to <new_axis>) can
    be a different length.  For example, <array> could have shape
    (30,73,144), <oldcube> could have shape (365,73,144) with
    dimensions of time,latitude,longitude, and <new_axis> would be a
    time axis of length 30.  <newcube> would then be a cube with data
    from <array>, the time axis of <new_axis> and latitude and
    longitude axes from <oldcube>, plus all the attributes of
    <oldcube>.

    If <new_var_name> is False, the new cube has the same name as the
    old cube, otherwise it is renamed.
    
    """
    if new_axis:
        # new_axis is an iris coordinate axis
        # Find the index of the corresponding axis in oldcube
        coord_name=new_axis.standard_name
        print('coord_name: {0!s}'.format(coord_name))
        coord_names=[dimc.standard_name for dimc in oldcube.dim_coords]
        print('coord_names: {0!s}'.format(coord_names))
        if coord_name not in coord_names:
            raise ValueError('coord_name is not in coord_names')
        coord_index=coord_names.index(coord_name)
        print('coord_index: {0!s}'.format(coord_index))
        # Create a list of two-lists, each of form [dim_coord,index]
        kdim=0
        dim_coords=[]
        for xx in oldcube.dim_coords:
            dim_coords.append([xx,kdim])
            kdim+=1
        # Overwrite the coord_index'th axis with new_axis two-list
        dim_coords[coord_index]=[new_axis,coord_index]
    else:
        # new_axis is False.
        # Check that array and oldcube have same dimensions
        if array.shape!=oldcube.shape:
            raise ValueError('Shape of array and oldcube must match.')
        # Create a list of two-lists, each of form [dim_coord,index]
        kdim=0
        dim_coords=[]
        for xx in oldcube.dim_coords:
            dim_coords.append([xx,kdim])
            kdim+=1
    # Create cube
    newcube=iris.cube.Cube(array,units=oldcube.units,attributes=oldcube.attributes,cell_methods=oldcube.cell_methods,dim_coords_and_dims=dim_coords)
    # Name cube
    if new_var_name:
        new_long_name=var_name2long_name[new_var_name]
        newcube.rename(new_long_name)
        newcube.var_name=new_var_name
    else:
        newcube.rename(oldcube.name())
        newcube.var_name=oldcube.var_name
    # Add aux coords
    for xx in oldcube.aux_coords:
        newcube.add_aux_coord(xx)

    return newcube

#==========================================================================

def conv_float32(old_array):
    """Convert a numpy array type to float32 ('single precision').

    Default numpy array type is float64 (64 bit, or 'double
    precision').  This leads to a doubling of disk space needed to
    store data, compared to float32 (32 bit, 'single precision'), with
    only spurious increase in precision.

    It is not a problem having float64 type numbers in memory, only
    when they are written to disk.  Hence, just call this function as
    a final wrapper for a numpy array that is about to be converted to
    an iris cube.  Do not bother with all the previous intermediate
    steps, as float64 is insidious and can easily creep back into to
    your arrays somewhere.

    Input: old_array should be a numpy array.

    Returns new_array, a numpy array of type float32.

    """

    new_array=old_array.astype(type('float32',(np.float32,),{}))
    return new_array

#==========================================================================

def block_times(aa,verbose=False):
    """Set start and end times for current block.

    Input <aa> is an object with the following attributes:

    outfile_frequency:  e.g., 'year' or 'month'
    year: integer for current year, e.g., 2016
    month: integer for current month, in range 1 to 12.

    Returns time,time2 which are datetime.datetime objects for the
    start and end times for the current block.  These are then
    typically used to create an iris constraint on time for extracting
    data.

    If outfile_frequency is 'year', start and end times are 00 UTC on
    1 Jan of current year, and 1 second before 00 UTC on 1 Jan the
    following year.

    If outfile_frequency is 'month', start and end times are 00 UTC on
    1st of current month and year, and 1 second before 00 UTC on 1st
    of the following month.

    """
    timedelta_second=datetime.timedelta(seconds=1)
    if aa.outfile_frequency=='year':
        time1=datetime.datetime(aa.year,1,1)
        time2=datetime.datetime(aa.year+1,1,1)-timedelta_second
    elif aa.outfile_frequency=='month':
        time1=datetime.datetime(aa.year,aa.month,1)
        if aa.month!=12:
            time2=datetime.datetime(aa.year,aa.month+1,1)-timedelta_second
        else:
            time2=datetime.datetime(aa.year+1,1,1)-timedelta_second
    else:
        raise ToDoError("Need to write code for other outfile_frequency.")
    if verbose:
        ss=h2a+'time1: {0!s} \n'+\
            'time2: {1!s} \n'+h2b
        print(ss.format(time1,time2))
    return time1,time2

#==========================================================================

def replace_wildcard_with_time(aa,string1):
    """Replace wild card question marks with current time.

    Input <aa> is an object with the following attributes:

    outfile_frequency:  e.g., 'year' or 'month'
    wildcard: e.g., '????' or '??????'
    year: integer for current year, e.g., 2016
    month: integer for current month, in range 1 to 12.

    Input <string1> is a string, usually a filename, which contains
    wild card question marks, e.g., ???? or ??????.

    The wild card question marks are replaced with the current date,
    e.g., YYYY or YYYYMM, depending on aa.outfile_frequency.

    Output is <string2>, typically a filename referring to a specific
    date or range of dates (e.g., year and month).

    """
    # Set wild card string
    if aa.outfile_frequency=='year':
        x2=str(aa.year)
    elif aa.outfile_frequency=='month':
        x2=str(aa.year)+str(aa.month).zfill(2)
    else:
        raise ToDoError('Need to code up for different outfile_frequency')
    # Replace wild card characters
    string2=string1.replace(aa.wildcard,x2)

    return string2

#==========================================================================

def iter_generator(input):
    """Convert <input> to an iterator and return it.

    If <input> is an integer, convert it to a list so it can be
    iterated over.  Otherwise, return unchanged.

    Typically, YEAR or MONTH is passed to this function.  They may
    already be in the form of a range-type iterator or a list, or just a single
    integer value.  This function converts the single integer value to
    an iterator (a list).  """
    if type(input)==type(1979):
        # input is an integer.  Convert to a list
        output=[input]
    else:
        # Return input unchanged
        output=input
    return output

#==========================================================================

def lat_direction(cube_in,direction,verbose=True):
    """Make cube latitude run north to south or south to north.

    Inputs:

    <cube_in> is an iris cube

    <direction> is a string, either 's2n' or 'n2s'

    Use iris.util.reverse if needed.

    Return <cube_out>, whose latitude axis runs according to
    <direction>.
    """
    # Check direction is valid
    if direction not in ['s2n','n2s']:
        raise ValueError('Direction is invalid.')
    # Find direction of latitude coordinate of input cube
    lat_coord=cube_in.coord('latitude')
    lat_beg=lat_coord.points[0]
    lat_end=lat_coord.points[-1]
    if lat_beg<lat_end:
        input_direction='s2n'
    else:
        input_direction='n2s'
    # If input cube does not have desired latitude direction, reverse it
    if direction!=input_direction:
        # Find index of latitude coordinate
        dim_coord_names=[xx.var_name for xx in cube_in.dim_coords]
        lat_index=dim_coord_names.index('latitude')
        # Reverse latitude
        cube_out=iris.util.reverse(cube_in,lat_index)
        # Check direction of output cube
        lat_coord=cube_out.coord('latitude')
        lat_beg=lat_coord.points[0]
        lat_end=lat_coord.points[-1]
        if lat_beg<lat_end:
            output_direction='s2n'
        else:
            output_direction='n2s'
        if direction!=output_direction:
            raise ValueError('Ouput direction does not match direction.')
    # Printed output
    if verbose:
        ss=h2a+'name: {4!s} \n'+\
            'lat_direction. \n'+\
            'direction, input_direction: {0!s}, {1!s} \n'+\
            'lat_index: {2!s} \n'+\
            'direction, output_direction: {0!s}, {3!s} \n'+h2b
        print(ss.format(direction,input_direction,lat_index,output_direction,cube_in.name()))

    return cube_out
    
#==========================================================================

def truncate(cube_in,truncation):
    """Spectrally truncate (triangular truncation) iris cube."""

    # Find value of south2north
    lat_coord=cube_in.coord('latitude')
    lat_beg=lat_coord.points[0]
    lat_end=lat_coord.points[-1]
    if lat_beg<lat_end:
        south2north=True
    else:
        south2north=False
    print('lat_beg, lat_end, south2north: {0!s}, {1!s}, {2!s} '.format(lat_beg,lat_end,south2north))
    # Create spoof uwnd and vwnd cubes to initiate VectorWind instance.
    uwnd=create_cube(cube_in.data,cube_in,new_var_name='uwnd')
    vwnd=create_cube(cube_in.data,cube_in,new_var_name='vwnd')
    # Create VectorWind instance
    ww=VectorWind(uwnd,vwnd)
    # Truncate input cube
    cube_out=ww.truncate(cube_in,truncation=truncation)
    # Add cell method to describe truncation
    cm=iris.coords.CellMethod('point',coords=['latitude','longitude'],comments='truncated to T'+str(truncation))
    cube_out.add_cell_method(cm)
    if south2north:
        # Output of windspharm method.  Latitude runs north to south
        # Reverse latitude direction (to be consistent with input cube)
        cube_out=lat_direction(cube_out,'s2n')
    return cube_out

#==========================================================================

def concatenate_cube(cubelist):
    """concatenate_cube cube list when one cube has singleton time dimension.

    Input is <cubelist>, an iris CubeList

    Output is <cube>, a concatenate_cube'd version of <cubelist>

    The problem: If a cube list contains a cube that has a singleton
    time coordinate, then iris will have relegated that time
    coordinate to a scalar coordinate when it extracted the cube.  The
    other cubes will have time coordinates as dimension coordinates.
    If the cubelist concatenate_cube method is used, it will fail because
    of this mismatch of dimensions.

    Example:
    x1=iris.load('uwnd_200_????.nc') # Load data stored in single file per year
    time1=datetime.datetime(1994,12,31)
    time2=datetime.datetime(1995,1,6)
    time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
    with iris.FUTURE.context(cell_datetime_objects=True):
       x2=x1.extract(time_constraint)
    x3=x2.concatenate_cube()

    x2 is a cube list containing two cubes.  The first cube only has
    one time (31 Dec 1994) and the time coordinate will have been
    relegated to a scalar coordinate by the extract method.  The
    second cube has 6 time values (1-6 Jan 1995) and its time
    coordinate will be a dimension coordinate.  The
    x2.concatenate_cube() will then fail.

    Solution: Wherever this is likely to happen, replace the standard
    cube list concatenate_cube method, i.e., in the example above,
    replace

    x3=x2.concatenate_cube() 

    with a call to this function:

    x3=concatenate_cube(x2)

    This function uses iris.util.new_axis() to promote the singleton
    scalar time coordinate to a dimension coordinate.  The cubes can
    then be concatenate_cube'd using the standard cube list
    concatenate_cube method.

    """
    # If there is only one cube in the cube list, concatenate_cube as usual
    if len(cubelist)==1:
        cube=cubelist.concatenate_cube()
        return cube
    # If there is more than one cube in the cube list, check for singleton
    #   time dimensions
    for index in range(len(cubelist)):
        cubec=cubelist[index]
        time_coord=cubec.coord('time')
        if time_coord.shape==(1,):
            print('concatenate_cube.  Promoting singleton time dimension.')
            cubec=iris.util.new_axis(cubec,'time')
            cubelist[index]=cubec
    # Concatente into single cube
    cube=cubelist.concatenate_cube()

    return cube

#==========================================================================

def cube_section(cube_in,xdat,ydat,verbose=False):
    """Calculate an irregular section through a cube.

    Called from e.g., curved_section.py

    Inputs:

    cube_in: input iris cube

    xdat,ydat : 2-tuples.  First element is string of coordinate name.
    Second element is list of coordinate values.  Both coordinate
    names must be coordinates of cube_in.

    e.g., xdat is ('longitude',[4.3, 5.7, 6.1])
          ydat is ('latitude', [13.1, 12.6, 5.6])

    Outputs:

    cube_out: iris cube, with data from cube_in interpolated onto the
    xdat,ydat values.

    """
    xdat_coord=xdat[0]
    xdat_vals=xdat[1]
    ydat_coord=ydat[0]
    ydat_vals=ydat[1]
    if verbose:
        ss=h1a+'section \n'+\
            'xdat_coord: {0!s}\n'+\
            'xdat_vals: {1!s}\n'+\
            'ydat_coord: {2!s}\n'+\
            'ydat_vals: {3!s}\n'+h1b
        print(ss.format(xdat_coord,xdat_vals,ydat_coord,ydat_vals))
    if len(xdat_vals)!=len(ydat_vals):
        raise ValueError('xdat_vals and ydat_vals must have same number of values.')
    x1=iris.cube.CubeList([])
    for ival in range(len(xdat_vals)):
        xdat_valc=xdat_vals[ival]
        ydat_valc=ydat_vals[ival]
        samples=[(xdat_coord,xdat_valc),(ydat_coord,ydat_valc)]
        x2=cube_in.interpolate(samples,iris.analysis.Linear())
        x1.append(x2)
    cube_out=x1.merge_cube()

    return cube_out

#==========================================================================

class ToDoError(ValueError):

    """An exception that indicates I need to write some more code.

    Use raise ToDoError as a placeholder for code to be written for a
    different situation."""

    pass

#==========================================================================

class TimeDomain(object):

    """A set of single or paired (start,end) times.

    A set of single times can be used as the basis of (lagged)
    composite analysis.

    A set of paired (start,end) times can be used as the basis of
    calculating time means and other statistics.

    They have a unique string identifier, e.g., 'jul9899'.

    They are stored/archived as ascii files in
    /gpfs/home/e058/home/data/tdomain/

    Once created, the ascii time domain files should not be modified
    or deleted.  A modified version of the time domain should be given
    a new id, and a new ascii file created for it.

    For historical reasons (from using the cdat module) the format of
    the ascii file is that from a print cdtime.comptime object,
    e.g. for 'jul9899':

    # Optional descriptive comment line(s) beginning with '#'
    # July 1998, 1999
    1998-07-01 00:00:0.0, 1998-07-31 23:59:0.0
    1999-07-01 00:00:0.0, 1999-07-31 23:59:0.0

    This format must be adhered to.

    The TimeDomain class has methods to read and write these ascii
    files, and to convert to/from datetime objects, for use with iris
    scripts.

    Attributes:

    self.idx : unique string identifier, e.g., 'jul9899'.  Must
    be supplied when creating instance of TimeDomain class.

    self.basedir : directory path for timedomain ascii files.  Has
    default value that can be overwritten as a keyword argument.

    self.filename : full path name for timedomain ascii file.  Created
    from self.basedir and self.idx.

    self._format2ascii : format used for conversion to ascii format.
    Do not change.

    self._format2datetime : format used for conversion to datetime
    object.  Do not change.

    self.lines : ascii representation of timedomain.  A nested list of
    strings.

    self.datetimes : datetime representation of timedomain.  A nested
    list of datetime objects.

    self.type : either 'single' or 'event'.

    self.nevents : integer equivalent to length of self.lines, i.e., number
    single times or events.

    self.header : tuple of two strings, each beginning with '#', that
    are descriptors of the time domain

    self.ndaytot : float of total number of days covered by time
    domain if it is of type 'event'.

    self.ndays : sorted list of floats of number of days in each event
    in time domain, if time domain is of type 'event'.

    """
    
    def __init__(self,idx,basedir=mypaths.DIR_TDOMAIN_DEFAULT,tseriesdir=mypaths.DIR_TSERIES_DEFAULT,verbose=True):
        self.idx=idx
        self.basedir=basedir
        self.tseriesdir=tseriesdir
        self.verbose=verbose
        self.filename=os.path.join(self.basedir,self.idx+'.txt')
        self._format2ascii="%Y-%m-%d %H:%M"
        self._format2datetime="%Y-%m-%d %H:%M:%S:%f"
        if self.verbose:
            print(self)

    def __repr__(self):
        return 'TimeDomain({0.idx!r},basedir={0.basedir!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'TimeDomain({0.idx!s},basedir={0.basedir!s},verbose={0.verbose!s}) \n'+\
                'filename={0.filename!s} \n'+h1b
            return ss.format(self)
        else:
            return self.__repr__()

    def read_ascii(self):
        """Read the ascii time domain file.

        Discard any comments (lines beginning with '#')

        Create lines and comments attribute.
        """
        file1=open(self.filename)
        lines=file1.readlines()
        file1.close()
        self.comments=[xx for xx in lines if xx[0]=='#']
        self.lines=[xx for xx in lines if xx[0]!='#']
        if self.verbose:
            ss='read_ascii:  {0.idx!s}: Created attributes "comments", "lines". \n'
            if self.verbose==2:
                ss+='   comments: {0.comments!s} \n'
                ss+='   lines: {0.lines!s} \n'
            print(ss.format(self))

    def write_ascii(self):
        """Write the ascii file for a newly created time domain."""
        if os.path.isfile(self.filename):
            raise UserWarning('Warning: time domain already exists.  Not overwriting.')
        else:
            # Write ascii strings
            file1=open(self.filename,'w')
            if self.header:
                file1.writelines(self.header)
            file1.writelines(self.lines)
            file1.close()
            if self.verbose:
                ss='write_ascii:  {0.idx!s}: Created ascii file. \n'
                print(ss.format(self))

    def ascii2datetime(self):
        """Convert ascii representation of timedomain to datetime representation.
        Create a datetimes attribute.

        """
        datetimes=[]
        for line in self.lines:
            # Separate comma-separated string times
            a=line.strip()
            a=a.split(',')
            datetimes_row=[]
            for t1 in a:
                # Convert decimal seconds, e.g., 0.0 from the ascii format
                # to integer seconds and integer microseconds for the
                # datetime object
                # The seconds will be after the final ":" in the t1 string
                t1a=t1.strip()
                index_decimal_point=0
                found_decimal_point=False
                while not found_decimal_point:
                    index_decimal_point-=1
                    if t1a[index_decimal_point]=='.':
                        found_decimal_point=True
                index_colon=0
                found_colon=False
                while not found_colon:
                    index_colon-=1
                    if t1a[index_colon]==':':
                        found_colon=True
                second='%02d' % int(t1a[index_colon+1:index_decimal_point])
                microsecond='%06d' % (1000000*int(t1a[index_decimal_point+1:]))
                t1b=t1a[:index_colon+1]+second+':'+microsecond
                # Convert string time to a datetime object and append
                xx=datetime.datetime.strptime(t1b,self._format2datetime)
                datetimes_row.append(xx)
            # Append this row of datetime objects to master list
            datetimes.append(datetimes_row)
        self.datetimes=datetimes
        if self.verbose:
            ss='ascii2datetime:  {0.idx!s}: Created attribute "datetimes". \n'
            if self.verbose==2:
                ss+='   datetimes: {0.datetimes!s} \n'
            print(ss.format(self))

    def ascii2partial_date_time(self):
        """Convert ascii representation of timedomain to PartialDateTime.
        
        Create a partial_date_times attribute.

        This method should not be needed.  I was confused over partial
        date times and datetimes.  Just use datetimes.

        """
        raise DeprecationWarning('This method should not be needed.  Use ascii2datetime instead.')
        
        # First create datetimes attribute if it does not exist
        try:
            datetimes=self.datetimes
        except AttributeError:
            self.ascii2datetime()
            datetimes=self.datetimes
        print('datetimes: {0!s}'.format(datetimes))
        # Convert datetime objects to PartialDateTime objects
        partial_date_times=[]
        for datetimes_row in datetimes:
            partial_date_times_row=[]
            for t1 in datetimes_row:
                pdtc=PartialDateTime(year=t1.year,month=t1.month,day=t1.day,hour=t1.hour,minute=t1.minute,second=t1.second,microsecond=t1.microsecond)
                partial_date_times_row.append(pdtc)
            # Append this row of PartialDateTime objects to master list
            partial_date_times.append(partial_date_times_row)
        self.partial_date_times=partial_date_times
        if self.verbose:
            ss='ascii2partial_date_time:  {0.idx!s}: Created attribute "partial_date_times". \n'
            if self.verbose==2:
                ss+='   partial_date_times: {0.partial_date_times!s} \n'
            print(ss.format(self))
        
    def datetime2ascii(self):
        """Convert datetime representation of timedomain to ascii.

        Create a lines attribute.

        """
        # Convert datetime objects to ascii strings
        lines=[]
        for row in self.datetimes:
            lines_row=''
            for t1 in row:
                # Convert datetime object to formatted string
                # Integer seconds and integer microseconds in the datetime
                # object must be converted to decimal seconds for the ascii
                # format
                decimal_second=str(t1.second+float(t1.microsecond)/1e6)
                xx=datetime.datetime.strftime(t1,self._format2ascii)+':'+decimal_second+', '
                lines_row+=xx
            # Remove the final ', ' from the last time
            lines_row=lines_row[:-2]
            # Add newline character
            lines_row+='\n'
            lines.append(lines_row)
        self.lines=lines
        if self.verbose:
            ss='read_ascii:  {0.idx!s}: Created attribute "lines". \n'
            if self.verbose==2:
                ss+='   lines: {0.lines!s} \n'
            print(ss.format(self))

    def time_domain_type(self):
        """Determine if time domain is 'single' or 'event' type.

        'Single' type is list of single times.

        'Event' type is list of paired (start,end) times.

        Creates type attribute.
        """
        t1=self.datetimes[0]
        if len(t1)==1:
            self.type='single'
        elif len(t1)==2:
            self.type='event'
        else:
            raise ValueError('Error: There must be either 1 or 2 times in each row.')
        if self.verbose:
            ss='time_domain_type:  {0.idx!s}: Created attribute "type". \n'
            if self.verbose==2:
                ss+='   type: {0.type!s} \n'

    def f_nevents(self):
        self.nevents=len(self.datetimes)

    def f_create_time_domain_from_indices(self):
        """Create time domain from algorithm on index time series."""
        if len(self.idx)==10 and self.idx[:3] in ['rmm','bs1','bs2']:
            # Time domains based on RMM or BSISO indices
            # Format is 'PPPXXXYYYZ' where
            #   'PPP' is name of index: 'rmm','bs1','bs2'
            #   'XXX' is a counter e.g. '001'
            #   'YYY' is a season e.g. 'djf', 'jja', 'n2a', 'm2o'
            #   'Z' is the category ('1' to '8')
            indexname=self.idx[0:3]
            counter=self.idx[3:6]
            season=self.idx[6:9]
            category=int(self.idx[9])
            print('indexname: {0!s}'.format(indexname))
            print('counter: {0!s}'.format(counter))
            print('season: {0!s}'.format(season))
            print('category: {0!s}'.format(category))
            # Counters:
            header1='# '+indexname+': '
            if counter=='001':
                header1+='Index amplitude >=1, time range 1 Jan 1979 to 31 Dec 2015 \n'
                time1=datetime.datetime(1979,1,1)
                time2=datetime.datetime(2015,12,31)
                time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
                amp_threshold=1
            elif counter=='002':
                header1+='Index amplitude >=0.75, time range 1 Jan 1979 to 31 Dec 2015 \n'
                time1=datetime.datetime(1979,1,1)
                time2=datetime.datetime(2015,12,31)
                time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
                amp_threshold=0.75
            elif counter=='003':
                # Same conditions as Fig. 9 in Lee et al. (2013)
                header1+='Index amplitude >=1.5, time range 1 Jan 1981 to 31 Dec 2010 \n'
                time1=datetime.datetime(1981,1,1)
                time2=datetime.datetime(2010,12,31)
                time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
                amp_threshold=1.5
            else:
                raise ValueError('counter is not valid.')
            # Seasons
            if season=='djf':
                valid_months=[12,1,2]
            elif season=='n2a':
                valid_months=[11,12,1,2,3,4]
            elif season=='m2o':
                valid_months=[5,6,7,8,9,10]
            else:
                raise ValueError('season is not valid.')
            # Category
            if category not in list(range(1,8+1)):
                raise ValueError('category is not valid.')
            header2='# Season='+season+', category='+str(category)+' \n'
            # Set header attribute for time domain
            self.header=(header1,header2)
            print(header1)
            print(header2)
            # Read input time series of index amplitude and category
            if indexname=='rmm':
                source='rmm_nl_d'
                f1=os.path.join(self.tseriesdir,source,'std','rmm_amp_-999.nc')
                f2=os.path.join(self.tseriesdir,source,'std','rmm_cat_-999.nc')
                with iris.FUTURE.context(netcdf_promote=True):
                    x1=iris.load_cube(f1,'RMM_amplitude')
                    x2=iris.load_cube(f2,'RMM_category')
            elif indexname=='bs1':
                source='bsiso_nl_d'
                f1=os.path.join(self.tseriesdir,source,'std','bsiso1_amp_-999.nc')
                f2=os.path.join(self.tseriesdir,source,'std','bsiso1_cat_-999.nc')
                with iris.FUTURE.context(netcdf_promote=True):
                    x1=iris.load_cube(f1,'BSISO_1_amplitude')
                    x2=iris.load_cube(f2,'BSISO_1_category')
            else:
                raise ValueError('Invalid indexname.')
            # Extract data for valid time range
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1a=x1.extract(time_constraint)
                x2a=x2.extract(time_constraint)
            time_coord=x1a.coord('time')
            time_units=time_coord.units
            # Flags indicate whether relevant time is within an event (True)
            # or not (False)
            flag_previous_time=False
            flag_current_time=False
            datetimes_list=[]
            # Create list of datetime.datetime pairs of (start,end) dates
            #   for events
            # Loop over time
            for timevalc in time_coord.points:
                timecompc=time_units.num2date(timevalc)
                time_constraint2=iris.Constraint(time=timevalc)
                # Extract index amplitude and category at current time
                ampc=float(x1a.extract(time_constraint2).data)
                catc=float(x2a.extract(time_constraint2).data)
                # Set flag for current time
                if ampc>=amp_threshold and catc==category and timecompc.month in valid_months:
                    flag_current_time=True
                else:
                    flag_current_time=False
                print(timevalc,timecompc,ampc,catc,flag_previous_time,flag_current_time)
                # Use flags to determine start and end points of events
                if not flag_previous_time and flag_current_time:
                    print('Start of event')
                    start_event=timecompc
                if flag_previous_time and not flag_current_time:
                    print('End of event')
                    end_event=timecomp_previous_time
                    datetimes_list.append([start_event,end_event])
                # Current time becomes previous time in next iteration
                flag_previous_time=flag_current_time
                timecomp_previous_time=timecompc
            # Set datetimes attribute and write time domain
            self.datetimes=datetimes_list
            self.datetime2ascii()
            self.write_ascii()

    def f_ndays(self):
        """Calculate number of days covered by time domain.

        Create ndays and ndaytot attributes.
        """
        if self.type=='event':
            ndaytot=0
            ndays=[]
            for (time_beg,time_end) in self.datetimes:
                timedeltac=time_end-time_beg
                ndayc=timedeltac.total_seconds()/86400+1
                ndays.append(ndayc)
                ndaytot+=ndayc
            ndays.sort()
            self.ndays=ndays
            self.ndaytot=ndaytot
        else:
            raise ValueError('Time domain must be of type "event" to calculate number of days.')

    def info(self):
        """Calculate and print information on time domain."""
        self.read_ascii()
        self.ascii2datetime()
        self.f_nevents()
        print('nevents: {0.nevents!s}'.format(self))
        self.time_domain_type()
        print('type: {0.type!s}'.format(self))
        if self.type=='event':
            self.f_ndays()
            print('ndays: {0.ndays!s}'.format(self))
            print('ndaytot: {0.ndaytot!s}'.format(self))

    def f_randomise_times(self,idx_new,max_day_shift,time_first,time_last):
        """Create copy of time domain with randomised times.

        Inputs:

        idx_new : string id of new (randomised) time domain

        max_day_shift : integer (typical value 15)

        time_first, time_last : datetime.datetime objects
        corresponding to the time range of the data this randomised
        time domain will be applied to.
        
        For each random time, the year of the original time is
        replaced by a random year between time_first.year and
        time_last.year, and the day is shifted randomly between
        plus/minus max_day_shift.  This procedure ensures the
        randomised time is at approximately the same time of year
        (i.e., same part of seasonal cycle) as the original time.

        If the time domain type is 'event', the randomisation is
        carried out on the start time of each pair of datetimes, and
        the time difference between the start and end time of each
        pair is preserved.

        Returns tdomain_rand, a new (randomised) TimeDomain object.
        """
        # Loop over datetimes in time domain
        datetimes_rand=[]
        for dtc in self.datetimes:
            # Extract start datetime if type is 'event', or only datetime if type is 'single'
            dt0=dtc[0]
            # If type is 'event' extract end datetime
            if self.type=='event':
                dt1=dtc[1]
                timedelta=dt1-dt0
            # Randomise datetime(s) and check they lie within allowed range
            checked_valid_time_range=False
            while not checked_valid_time_range:
                year_rand=random.randrange(time_first.year,time_last.year+1)
                day_increment_rand=random.randrange(-max_day_shift,max_day_shift+1)
                dt0_rand=datetime.datetime(year_rand,dt0.month,dt0.day,dt0.hour,dt0.minute,dt0.second)+datetime.timedelta(days=day_increment_rand)
                print(year_rand,day_increment_rand,dt0,dt0_rand)
                # If type is 'event' change end datetime consistently to preserve time difference between start and end time of this datetime pair
                if self.type=='event':
                    dt1_rand=dt0_rand+timedelta
                    print(dt1_rand)
                # Check randomised times are within allowable range
                if (self.type=='single' and time_first<=dt0<=time_last) or (self.type=='event' and time_first<=dt0 and dt1<=time_last):
                    checked_valid_time_range=True
                    print('Randomised datetimes lie inside allowed range.  Proceed.')
                else:
                    print('Randomised datetimes lie outside allowed range.  Randomise again.')
            if self.type=='event':
                datetimes_rand.append([dt0_rand,dt1_rand])
            elif self.type=='single':
                datetimes_rand.append([dt0_rand])
            else:
                raise ValueError('Invalid time domain type.')
        # Create new time domain from randomised datetimes
        tdomain_rand=TimeDomain(idx_new)
        tdomain_rand.datetimes=datetimes_rand

        return tdomain_rand

#==========================================================================

class DataConverter(object):

    """Converter to iris-friendly standard format netcdf files.

    Called from preprocess.py.

    Using iris, read in data from different sources.  Convert to
    standard cube format and file format.  Write to new netcdf file.

    Code is flexible to allow for new input data sources.

    Attributes:

    self.basedir : string name of base directory for all data

    self.source : string name of data source, e.g., 'ncepdoe_plev_d',
    'ncepdoe_plev_6h', 'ncepdoe_sfc_6h', 'olrcdr_toa_d' etc.  There
    is a different source for each different combination of source
    data set (e.g., NCEP-DOE reanalysis), type of level (e.g.,
    pressure level), and frequency of input data (e.g., daily).  The
    string should be in the format
    '<data_source>_<level_type>_<frequency>'.  In practice, the source
    attribute is used to set the directory in which the netcdf files
    are stored:
       <self.basedir>/<self.source>/raw/ for original data in the
          original format downloaded from data source web site.  Netcdf
          files in this directory are the input to DataConverter
       <self.basedir>/<self.source>/std/ for the converted, standardised
          data, i.e., the output of DataConverter
       <self.basedir>/<self.source>/processed/ for any subsequent analysis
          on the data, e.g., time means etc.  DataConverter does not use this
          directory.

    self.data_source : string name of data source, e.g., 'ncepdoe',
    'olrcdr', etc.  Note that a particular data_source implies a
    particular latitude-longitude grid, e.g., 'ncepdoe' is a 73x144
    grid.  This is relevant for file sizes and self.outfile_frequency.

    self.level_type : string name of level type, e.g., 'plev', 'toa',
    'sfc', 'theta', etc.  This is determined by self.source.

    self.frequency : string denoting time frequency of data.  This is
    used in file names.  It is determined by self.source.  One of:
       'h' hourly
       '3h' 3-hourly
       '6h' 6-hourly
       'd' daily
       '5d' pentad (5-day)
       '7d' weekly (7-day)
       'm' monthly (calendar month)

    self.outfile_frequency : string denoting the time coverage of the
    output netcdf file(s).  It is determined by self.source.  One of:
       'year' for separate files for each year, e.g., 1979, 1980
       'month' for separate files for each calendar month, e.g., 197901, 197902
    
    self.var_name : string name of variable, e.g., 'uwnd'.

    self.raw_name : string name of variable in the raw input file.

    self.level : integer single level, e.g., 200 for 200 hPa.  Level naming
    convention:
       1000, 850, 200, etc. for pressure level data (hPa)
       350, 360, etc. for theta level data
       0 for top of atmosphere, e.g., olr
       1 for surface, e.g., sst
       Note there is no ambiguity between eg a 350 K theta level, and
       a 350 hPa pressure level, because of the self.level_type
       attribute, which is determined by self.source.

    self.cube : iris cube of data.  This is the most important
    attribute of DataConverter. The purpose of DataConverter is to
    convert self.cube to standard form and write it as a netcdf file.

    self.file_mask and self.mask.  Default values are False.  If the
    data is to be masked (e.g., SST data to be masked with a land-sea
    mask), then file_mask is the path name for the mask, and mask is
    the array with the mask.

    """
    
    def __init__(self,descriptor,verbose=True):
        """Initialise from descriptor dictionary."""
        self.descriptor=descriptor
        self.verbose=verbose
        self.source=descriptor['source']
        source_info(self)
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.level=descriptor['level']
        self.basedir=descriptor['basedir']
        self.file_mask=False # Default value, overwrite if exists
        self.mask=False # Ditto
        if descriptor['file_mask']:
            self.file_mask=os.path.join(self.basedir,self.source,'raw',descriptor['file_mask'])
            with iris.FUTURE.context(netcdf_promote=True):
                x1=iris.load(self.file_mask,callback=clean_callback)
            x2=x1.concatenate_cube()
            self.mask=x2.data # numpy array of mask data
            if len(self.mask.shape)!=3 and self.mask.shape[0]!=1:
                # Mask should be 2-D (eg lat,lon) but with a third time
                # dimension of length 1
                raise ValueError('Expecting 3-d mask of shape (1,?,?)')
            if self.source=='sstrey_sfc_7d':
                self.mask=1-self.mask # Switch the 1's and 0's
        if self.verbose:
            print(self)
        
    def __repr__(self):
        return 'DataConverter({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'DataConverter instance \n'+\
                'source: {0.source!s} \n'+\
                'var_name: {0.var_name!s} \n'+\
                'name: {0.name!s} \n'+\
                'level: {0.level!s} \n'
            if self.file_mask:
                ss+='file_mask: {0.file_mask!s} \n'+\
                     'mask.shape: {0.mask.shape!s} \n'
            ss+=h1b
            return ss.format(self)
        else:
            return self.__repr__()

    def read_cube(self):
        """Read cube from raw input file.

        Code is by necessity ad hoc as it caters for many different
        data sources with different input formats.

        """
        # Set time constraint for current time block
        time1,time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time = lambda cell: time1 <= cell <= time2)
        #
        # Set input file name(s)
        if self.source in ['erainterim_plev_6h']:
            self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+str(self.level)+'_'+str(self.year)+'_6.nc')
        elif self.source in ['ncepdoe_plev_6h','ncepdoe_plev_d','ncepncar_plev_d']:
            if self.var_name=='ta':
                self.filein1=os.path.join(self.basedir,self.source,'raw','air'+'.'+str(self.year)+'.nc')
            elif self.var_name=='zg':
                self.filein1=os.path.join(self.basedir,self.source,'raw','hgt'+'.'+str(self.year)+'.nc')
            else:
                self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+'.'+str(self.year)+'.nc')
        elif self.source in ['ncepncar_sfc_d',]:
            self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+'.sig995.'+str(self.year)+'.nc')
        elif self.source in ['olrcdr_toa_d','olrinterp_toa_d']:
            self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+'.day.mean.nc')
        elif self.source in ['sg579m031oi01_zlev_h','sg534m031oi01_zlev_h','sg532m031oi01_zlev_h','sg620m031oi01_zlev_h','sg613m031oi01_zlev_h',]:
            self.filein1=os.path.join(self.basedir,self.source,'raw','oi_zt_2m3h_SG'+self.source[2:5]+'.nc')
        elif self.source in ['sstrey_sfc_7d',]:
            if 1981<=self.year<=1989:
                self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+'.wkmean.1981-1989.nc')
            elif 1990<=self.year:
                self.filein1=os.path.join(self.basedir,self.source,'raw',self.var_name+'.wkmean.1990-present.nc')
            else:
                raise ValueError('Invalid year')
        elif self.source in ['trmm3b42v7_sfc_3h']:
            # Inconsistent file naming from NASA DISC
            # 1998-1999 and 2011-2016 files end in .7.nc
            # 2000-2010 files end in .7A.nc
            # Use '.7*.nc' to cover both
            self.filein1=os.path.join(self.basedir,self.source,'raw',str(self.year)+str(self.month).zfill(2),'3B42.'+str(self.year)+str(self.month).zfill(2)+'*.7*.nc')
        elif self.source in ['tropflux_sfc_d']:
            if self.var_name=='lhfd':
                self.filein1=os.path.join(self.basedir,self.source,'raw','lhf_tropflux_1d_'+str(self.year)+'.nc')
        else:
            raise ValueError('Data source not recognised')
        #
        # Set level constraint (set to False if none)
        if self.data_source in ['erainterim'] and self.level_type=='plev':
            level_constraint=iris.Constraint(p=self.level)
        elif self.data_source in ['ncepdoe','ncepncar'] and self.level_type=='plev':
            level_constraint=iris.Constraint(Level=self.level)
        elif self.source in ['ncepncar_sfc_d','olrcdr_toa_d','olrinterp_toa_d','sg579m031oi01_zlev_h','sg534m031oi01_zlev_h','sg532m031oi01_zlev_h','sg620m031oi01_zlev_h','sg613m031oi01_zlev_h','sstrey_sfc_7d','trmm3b42v7_sfc_3h','tropflux_sfc_d']:
            level_constraint=False
        else:
            raise ToDoError('Set an instruction for level_constraint.')
        #
        # Set raw_name of variable in raw input data
        self.raw_name=self.name
        if self.data_source in ['erainterim',]:
            if self.var_name in['omega']:
                self.raw_name='vertical_air_velocity_expressed_as_tendency_of_pressure'
        elif self.data_source in ['ncepdoe',]:
            if self.var_name in['omega']:
                self.raw_name=self.var_name
        elif self.data_source in ['ncepncar',]:
            if self.var_name in['uwnd','vwnd','omega']:
                self.raw_name=self.var_name
        elif self.data_source in ['olrinterp',]:
            self.raw_name='olr'
        elif self.data_source in ['sg579m031oi01','sg534m031oi01','sg532m031oi01','sg620m031oi01','sg613m031oi01',]:
            if self.var_name=='tsc':
                self.raw_name='cons_temp'
            elif self.var_name=='sa':
                self.raw_name='abs_salin'
        elif self.data_source in ['tropflux',]:
            if self.var_name=='lhfd':
                self.raw_name='lhf'
        #
        # Load cube
        with iris.FUTURE.context(netcdf_promote=True):
            self.cube=iris.load_cube(self.filein1,self.raw_name,callback=clean_callback)
        xx=self.cube.coord('time')
        xx.bounds=None # Hack for new netcdf4 ncepdoe which have physically implausible time bounds
        with iris.FUTURE.context(cell_datetime_objects=True):
            if level_constraint:
                self.cube=self.cube.extract(level_constraint & time_constraint)
            else:
                self.cube=self.cube.extract(time_constraint)
        #
        # Apply mask if appropriate
        if self.file_mask:
            print('Applying mask')
            if self.cube.dim_coords[0].name()!='time':
                raise ValueError('First dimension of cube needs to be time.')
            # Broadcast ntime copies of mask (1,?,?) to (ntime,?,?)
            ntime=self.cube.shape[0]
            ngrid=self.mask.shape[1]*self.mask.shape[2]
            x1=self.mask.reshape((1,ngrid))
            ones=np.ones((ntime,1))
            x2=np.dot(ones,x1)
            x3=x2.reshape((ntime,self.mask.shape[1],self.mask.shape[2]))
            # Apply mask
            self.cube.data=np.ma.array(self.cube.data,mask=x3)
        if self.verbose==2:
            ss=h2a+'read_cube. \n'+\
                'time1: {0!s} \n'+\
                'time2: {1!s} \n'+h2b
            print(ss.format(time1,time2))
        
    def format_cube(self):
        """Change cube to standard format.

        There a few standard format changes applied to all data sets,
        followed by changes specific to particular data sets.
        
        """
        #
        # Universal format changes
        self.cube.rename(self.name)
        self.cube.var_name=self.var_name
        self.cube.coord('time').bounds=None
        self.cube.attributes['data_source']=self.data_source
        self.cube.attributes['level_type']=self.level_type
        self.cube.attributes['frequency']=self.frequency
        self.cube.attributes['outfile_frequency']=self.outfile_frequency
        #
        # BoBBLE OI glider data from Ben Webber
        if self.data_source[:2]=='sg' and self.data_source[6:]=='031oi01':
            # Missing data is set to zero.  Change to 1e20 and mask
            missing_value=1e20
            x1=np.where(np.equal(self.cube.data,0),missing_value,self.cube.data)
            self.cube.data=np.ma.masked_equal(x1,missing_value)
            ## Reset time, as original hourly data time axis was in days
            ## since ..., but values were only stored to 2 dp.
            ## As 1 hour = 0.041666667 days, significant round off error
            ## New time axis is in hours since first time
            #tc=self.cube.coord('time')
            #time0_val=tc.points[0]
            #if time0_val!=int(time0_val):
            #    raise ValueError('Need a integer value to start with.')
            #time0_datetime=tc.units.num2date(time0_val)
            #new_time_units='hours since '+str(time0_datetime)
            #ntime=tc.points.shape[0]
            #print('ntime : {0!s}'.format(ntime))
            #print('time0_datetime : {0!s}'.format(time0_datetime))
            #new_time_vals=np.arange(ntime)
            #new_time_coord=iris.coords.DimCoord(new_time_vals,standard_name='time',units=new_time_units)
            #print('new_time_coord : {0!s}'.format(new_time_coord))
            #self.cube.remove_coord('time')
            #self.cube.add_dim_coord(new_time_coord,0)
            # Dec 2016.  New data set with time axis units of
            # 'seconds since 1970-01-01'.  However, this suffers from
            # round off error.
            # 5 Dec 2016.  Ben redid analysis and set time axis to
            # 'hours since 2016-1-1' with first time value being 4369 and the
            # last being 4800 (ntime=432)
            # 9 Jan 2017.  Ben supplied new OI data for SG532 with corrected
            #   salinity.  However, time axis had reverted back to
            #   'seconds since 1970-1-1' which suffers from round off error.
            # Overwrite
            tc=self.cube.coord('time')
            if tc.points.shape!=(432,):
                raise ValueError('Time axis not right')
            new_time_vals=conv_float32(np.arange(4369,4800+1))
            new_time_units='hours since 2016-1-1'
            new_time_coord=iris.coords.DimCoord(new_time_vals,standard_name='time',units=new_time_units)
            print('new_time_coord : {0!s}'.format(new_time_coord))
            self.cube.remove_coord('time')
            self.cube.add_dim_coord(new_time_coord,0)
            if self.var_name=='tsc':
                self.cube.units='degC'
            elif self.var_name=='lon':
                self.cube.units='degree_east'
        #
        # Reynolds SST weekly data.
        # Time stamp is at beginning of week.
        # Change so it is at the end of the week, by adding 3 (days).
        # NB This can bump a data point at the end of the year into the next
        # year, eg 1982-12-29 is changed to 1984-01-01.  This should not
        # matter as the next step with this data is to linearly interpolate
        # to daily data, which uses all data, not the individual yearly files.
        if self.source=='sstrey_sfc_7d':
            tcoord=self.cube.coord('time')
            time_units=tcoord.units
            if 'day' not in time_units.name:
                raise ValueError('Expecting time units in days.')
            time_val=tcoord.points+3
            tcoord2=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
            if self.verbose==2:
                print('tcoord: {0!s}'.format(tcoord))
                print('tcoord2: {0!s}'.format(tcoord2))
            self.cube.remove_coord('time')
            self.cube.add_dim_coord(tcoord2,0)
            print('Added 3 days to time coord so time is now in centre of 7-day mean.')
        #
        # Tropflux daily data.
        # Time stamp is at 12 UTC. Change to 00 UTC by subtracting 0.5 (days).
        # Also, longitude runs from 30.5 to 279.5 (19.5E), missing 10 deg long over Africa
        # Set it to run from 0.5 to 359.5 with no missing longitude
        if self.source=='tropflux_sfc_d':
            tcoord=self.cube.coord('time')
            time_units=tcoord.units
            if 'day' not in time_units.name:
                raise ValueError('Expecting time units in days.')
            # Cannot simply subtract 0.5 because of round off, leading to later errors
            # e.g., time value on 1 Jan 2016 is 24105.999999999534 not 24105.5
            # Use divmod to achieve same end
            #time_val=tcoord.points-0.5
            time_val=divmod(tcoord.points,1)[0]
            tcoord2=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
            if self.verbose==2:
                print('tcoord: {0!s}'.format(tcoord))
                print('tcoord2: {0!s}'.format(tcoord2))
            self.cube.remove_coord('time')
            self.cube.add_dim_coord(tcoord2,0)
            print('Subtracted 0.5 days from time coord so time is now at 00 UTC not 12 UTC.')
            # Longitude
            # 0.5-19.5E
            lon_constraint1=iris.Constraint(longitude = lambda cell: 360.5 <= cell <= 379.5)
            x1=self.cube.extract(lon_constraint1)
            # create cube of missing values at missing longitudes 20.5-29.5E
            shape=x1.shape[:-1]+(10,)
            x2=1e20*conv_float32(np.ones(shape))
            x2=np.ma.array(x2,mask=x2)
            # 30.5-359.5E
            lon_constraint3=iris.Constraint(longitude = lambda cell:  30.5 <= cell <= 359.5)
            x3=self.cube.extract(lon_constraint3)
            # Concatenate the three longitude bands
            x4=np.ma.concatenate((x1.data,x2,x3.data),2)
            # Create new longitude axis
            lon_val=conv_float32(np.arange(0.5,359.5+1e-6,1.0))
            lon_units=x1.coord('longitude').units
            loncoord=iris.coords.DimCoord(lon_val,standard_name='longitude',units=lon_units,circular=True)
            # Create iris cube
            kdim=0
            dim_coords=[]
            for xx in x1.dim_coords[:-1]:
                dim_coords.append([xx,kdim])
                kdim+=1
            dim_coords.append([loncoord,kdim])
            x5=iris.cube.Cube(x4,units=x1.units,attributes=x1.attributes,cell_methods=x1.cell_methods,dim_coords_and_dims=dim_coords)
            x5.rename(x1.name())
            x5.var_name=x1.var_name
            self.cube=x5

    def write_cube(self):
        """Write standardised iris cube to netcdf file."""
        # Set output file name
        if self.outfile_frequency=='year':
            self.fileout1=os.path.join(self.basedir,self.source,'std/',self.var_name+'_'+str(self.level)+'_'+str(self.year)+'.nc')
        elif self.outfile_frequency=='month':
            self.fileout1=os.path.join(self.basedir,self.source,'std/',self.var_name+'_'+str(self.level)+'_'+str(self.year)+str(self.month).zfill(2)+'.nc')
        else:
            raise ToDoError("Need to write code for this outfile_frequency.")
        # Write cube
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.cube,self.fileout1)
        if self.verbose==2:
            print('write_cube: {0.fileout1!s}'.format(self))

#==========================================================================

class TimeDomStats(object):

    """Time mean and other statistics of data over non-contiguous time domain.

    Called from several scripts, including mean.py.

    All statistics calculations will be done by this class, so add
    further functions as needed.

    Selected attributes:

    self.tdomainid : string id of the time domain.

    self.tdomain : instance of TimeDomain class for the time domain.

    self.filein1 : path name for input files for data to calculate
    statistics from.  Will likely contain wild card characters.

    self.fileout1 : path name for output file of calculated statistic.
    
    self.cube_event_means : iris cube list of individual event means,
    i.e., the time mean over each pair of (start,end) times in the
    time domain.

    self.cube_event_ntimes : list of integers with number of times
    that went into each even mean, e.g., list of numbers of days in
    each event is data is daily.

    self.time_mean : iris cube of time mean calculated over the whole
    time domain.

    # Following attributes are used in Monte Carlo simulation to
    # calculate null distribution of mean.

    self.nmc : integer number of Monte Carlo simulations to be
    performed for calculation of null distribution.

    self.percentiles_null : list of the percentiles of the null
    distribution to be saved.

    self.max_day_shift : integer maximum day shift to be used in
    randomisation of time domain times for use in calculation of null
    distribution.

    self.time_first : datetime.datetime object that is first (i.e.,
    lowest) allowable date for the randomised times in the Monte Carlo
    simulation.

    self.time_last : as self.time_first but the last (i.e., highest)
    allowable date.

    """

    def __init__(self,descriptor,verbose=False):
        """Initialise from descriptor dictionary."""
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.level=descriptor['level']
        self.source=descriptor['source']
        source_info(self)
        self.tdomainid=descriptor['tdomainid']
        self.filepre=descriptor['filepre']
        self.filein1=os.path.join(self.basedir,self.source,'std',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.wildcard+'.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.filein1,self.name)
        self.tdomain=TimeDomain(self.tdomainid,verbose=self.verbose)
        self.tdomain.read_ascii()
        self.tdomain.ascii2datetime()
        self.tdomain.f_nevents()
        self.fileout_mean=os.path.join(self.basedir,self.source,'processed',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.tdomainid+'.nc')
        self.fileout_lagged_mean=os.path.join(self.basedir,self.source,'processed',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.tdomainid+'_lag.nc')
        self.fileout_dc=os.path.join(self.basedir,self.source,'processed',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.tdomainid+'_dc.nc')
        if 'nmc' in descriptor:
            self.nmc=descriptor['nmc']
        if 'percentiles_null' in descriptor:
            self.percentiles_null=descriptor['percentiles_null']
        if 'max_day_shift' in descriptor:
            self.max_day_shift=descriptor['max_day_shift']
        if 'time_first' in descriptor:
            self.time_first=descriptor['time_first']
        if 'time_last' in descriptor:
            self.time_last=descriptor['time_last']
        if self.verbose:
            print(self)
        
    def __repr__(self):
        return 'TimeDomStats({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'TimeDomStats instance \n'+\
                'var_name: {0.var_name!s} \n'+\
                'level: {0.level!s} \n'+\
                'source: {0.source!s} \n'+\
                'tdomainid: {0.tdomainid!s} \n'+\
                'data_in: {0.data_in!s} \n'+\
                'filein1: {0.filein1!s} \n'+\
                'fileout_mean: {0.fileout_mean!s} \n'+\
                'fileout_lagged_mean: {0.fileout_lagged_mean!s} \n'+\
                'fileout_dc: {0.fileout_dc!s} \n'+h1b
            return ss.format(self)
        else:
            return 'Statistics of '+self.source+' '+self.var_name+str(self.level)+' over '+self.tdomainid

    def event_means(self):
        """Calculate time mean and ntime for each event in time domain.

        Create cube_event_means and cube_event_ntimes attributes.

        """
        # Check that time domain is of type 'event'
        self.tdomain.time_domain_type()
        if self.tdomain.type!='event':
            raise ValueError("Warning: time domain type is '{0.tdomain.type}'.  It must be 'event'.".format(self))
        # Loop over events in time domain
        cube_event_means=iris.cube.CubeList([])
        cube_event_ntimes=[]
        for eventc in self.tdomain.datetimes:
            time_beg=eventc[0]
            time_end=eventc[1]
            print('time_beg,time_end: {0!s}, {1!s}'.format(time_beg,time_end))
            time_constraint=iris.Constraint(time=lambda cell: time_beg <=cell<= time_end)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            x2=concatenate_cube(x1)
            ntime=x2.coord('time').shape[0]
            cube_event_ntimes.append(ntime)
            if ntime==1:
                x3=x2
            else:
                x3=x2.collapsed('time',iris.analysis.MEAN)
            cube_event_means.append(x3)
        self.cube_event_means=cube_event_means
        self.cube_event_ntimes=cube_event_ntimes
        self.units=x2.units

    def f_time_mean(self):
        """Calculate time mean over time domain and save to netcdf.

        Calculate this by a weighted (cube_event_ntimes) mean of the
        cube_event_means.  Hence, each individual time (e.g., day) in the
        original data has equal weighting.

        Create attribute time_mean.

        """
        # Contribution from first event mean
        ntime_total=0
        ntime=self.cube_event_ntimes[0]
        x1=self.cube_event_means[0]*float(ntime)
        ntime_total+=ntime
        # Contribution from remaining events
        if self.tdomain.nevents>1:
            for ievent in range(1,self.tdomain.nevents):
                ntime=self.cube_event_ntimes[ievent]
                x1+=self.cube_event_means[ievent]*float(ntime)
                ntime_total+=ntime
        # Calculate mean
        #time_mean=x1/ntime_total
        # iris bug. Sometimes (eg glider tsc all) the line above fails.
        # Work around.  Make a copy of iris cube, and just access data.
        x2=x1.copy()
        x2.data/=ntime_total
        time_mean=x2
        # Set attributes
        time_mean.rename(self.name)
        time_mean.var_name=self.var_name
        time_mean.units=self.units
        # Add cell method to describe time mean
        cm=iris.coords.CellMethod('point','time',comments='mean over time domain '+self.tdomain.idx)
        time_mean.add_cell_method(cm)
        self.time_mean=time_mean
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.time_mean,self.fileout_mean)

    def f_percentiles_null(self):
        """Calculate percenticles of null distribution of the time mean.

        Perform a Monte Carlo simulation of size self.nmc to create
        the null distribution of the time mean.  For each evaluation in the
        Monte Carlo simulation,

        
        The time mean is then calculated for each of the self.nmc
        randomised time domains, building up a null distribution of
        the mean of length self.nmc.

        The percentiles self.percentiles_null of this distribution are
        then extracted and saved as self.mean_percentiles_null.

        Creates attributes:

        self.mean_percentiles_null

        """
        if self.verbose:
            ss=h1a+'f_percentiles_null \n'+\
                'nmc: {0.nmc!s} \n'+\
                'percentiles_null: {0.percentiles_null!s} \n'+\
                'max_day_shift: {0.max_day_shift!s} \n'+\
                'time_first: {0.time_first!s} \n'+\
                'time_last: {0.time_last!s} \n'+h1b
            print(ss.format(self))
        self.tdomain.time_domain_type()
        zfill_length=len(str(self.nmc))
        # Loop over Monte Carlo simulations and create list of randomised time domains
        tdomains_mc=[]
        for imc in range(self.nmc):
            idxc=self.tdomainid+'_'+str(imc).zfill(zfill_length)
            print('imc,idxc: {0!s}, {1!s}'.format(imc,idxc))
            tdomain_rand=self.tdomain.f_randomise_times(idxc,self.max_day_shift,self.time_first,self.time_last)
            tdomains_mc.append(tdomain_rand)
        self.tdomains_mc=tdomains_mc

        # Loop over Monte Carlo simulations again and create list of time means from each randomised time domain

    def f_lagged_mean(self,lags=[0,]):
        """Calculate time-lagged means over time domain and save.

        self.tdomain must be of type 'single'.

        Input:

        lags : list of integer lags (default is [0,]). These are
        combined with self.timedelta to create the lags to calculate
        the mean over.  E.g., if lags=[-5,0,5] and self.timedelta is 1
        day, then lagged means will be calculated at lags of -5, 0,
        and 5 days.

        Create attributes:

        self.lags : lags

        self.nlags : length of list of lags

        self.lagged_mean : iris cube of lagged means.  Shape is
        (self.nlags,...)  where ... is shape of input cube.  The 'lag'
        axis is a time axis, relative to an arbitrary reference time
        of 1000-01-01 00:00:0.0

        """
        # Check that time domain is of type 'single'
        self.tdomain.time_domain_type()
        if self.tdomain.type!='single':
            raise ValueError("Warning: time domain type is '{0.tdomain.type}'.  It must be 'single'.".format(self))
        # Set lags and nlags attributes
        self.lags=lags
        self.nlags=len(lags)
        timelag_units='hours since 1000-01-01 00:00:0.0'
        # Loop over lags
        cubelist=iris.cube.CubeList([])
        for lagc in self.lags:
            timedelta_lagc=lagc*self.timedelta
            print('lagc,timedelta_lagc : {0!s}, {1!s}'.format(lagc,timedelta_lagc))
            datetimes_lagc=[xx[0]+timedelta_lagc for xx in self.tdomain.datetimes]
            time_constraints_lagc=[iris.Constraint(time=xx) for xx in datetimes_lagc]
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=[self.data_in.extract(xx) for xx in time_constraints_lagc]
            pdb.set_trace()
            x2=[xx.concatenate_cube() for xx in x1]
            x3=iris.cube.CubeList(x2)
            x4=x3.merge_cube()
            x5=x4.collapsed('time',iris.analysis.MEAN)
            x5.remove_coord('time')
            hours_lagc=timedelta_lagc.total_seconds()/3600
            timelag_coord=iris.coords.DimCoord([hours_lagc],standard_name='time',units=timelag_units)
            x5.add_aux_coord(timelag_coord)
            cubelist.append(x5)
        lagged_mean=cubelist.merge_cube()
        # Add cell method to describe time mean
        cm=iris.coords.CellMethod('point','time',comments='lagged mean over time domain '+self.tdomain.idx)
        lagged_mean.add_cell_method(cm)
        #
        self.lagged_mean=lagged_mean
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.lagged_mean,self.fileout_lagged_mean)

    def f_diurnal_cycle(self,double=True):
        """Calculate mean diurnal cycle.

        Time axis for mean diurnal cycle runs for one day (1 Jan in
        year 1, ie 01-01-01).

        if double is True, create a copy of the diurnal cycle for a
        second day, ie 2 Jan year 1.  This is to help in plotting
        later.

        Create attribute:

        self.mean_dc : iris cube of mean diurnal cycle calculated over
        time domain self.tdomain.

        Note: initial attempt at coding this function used
        PartialDateTime to extract data on each day for a given time
        of day (e.g. 0300 UTC) and then average.  This took about
        10-100 times longer to run than method here.
        
        """
        # Extract first day in first event of time domain
        xx=self.tdomain.datetimes[0]
        first_day=xx[0]
        last_day=xx[-1]
        print('first_day,last_day: {0!s}, {1!s}'.format(first_day,last_day))
        time1=datetime.datetime(first_day.year,first_day.month,first_day.day)
        time2=datetime.datetime(first_day.year,first_day.month,first_day.day,23,59,59)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        x1=x1.concatenate_cube()
        cube1=x1
        # NB the data attribute of an iris cube is a numpy array if there is
        # no missing data, but a numpy masked array if there is missing data
        # To deal with occasional missing data, convert the data attribute
        # to a numpy masked array, and turn the boolean mask into an array of
        # zeroes and ones.  The kount array then keeps track of the number
        # of non-missing data values contributing to the sum at each spatial
        # grid point
        x1ma=np.ma.array(x1.data)
        x1ma_mask_numerical=np.where(x1ma.mask,0,1)
        x1ma_times_mask=x1ma.data*x1ma_mask_numerical
        data_sum=x1ma_times_mask
        kount=np.zeros(x1.data.shape)
        kount+=x1ma_mask_numerical
        print('time1,time2: {0!s}, {1!s}'.format(time1,time2))
        #print('type(x1.data): {0!s}'.format(type(x1.data)))
        #print(kount.shape,x1ma_mask_numerical.shape)
        #
        # Create time coordinate for final diurnal cycle
        # Extract times of day for data on first day
        tc=x1.coord('time')
        time_units=tc.units
        x2=[tc.units.num2date(xx) for xx in tc.points]
        # Reset these times to year=1,month=1,day=1
        times_datetime=[datetime.datetime(1,1,1,xx.hour,xx.minute,xx.second) for xx in x2]
        times_val=[time_units.date2num(xx) for xx in times_datetime]
        time_coord=iris.coords.DimCoord(times_val,standard_name='time',units=time_units)
        print('times_datetime: {0!s}'.format(times_datetime))
        print('times_val: {0!s}'.format(times_val))
        print('time_coord: {0!s}'.format(time_coord))
        #
        # Loop over (any) remaining days in first event of time domain and add data to data_sum
        timedelta_day=datetime.timedelta(days=1)
        time1+=timedelta_day
        time2+=timedelta_day
        while time1<=last_day:
            time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            x1=x1.concatenate_cube()
            x1ma=np.ma.array(x1.data)
            x1ma_mask_numerical=np.where(x1ma.mask,0,1)
            x1ma_times_mask=x1ma.data*x1ma_mask_numerical
            data_sum+=x1ma_times_mask
            kount+=x1ma_mask_numerical
            print('time1,time2: {0!s}, {1!s}'.format(time1,time2))
            #print('type(x1.data): {0!s}'.format(type(x1.data)))
            #print(kount.shape,x1ma_mask_numerical.shape)
            time1+=timedelta_day
            time2+=timedelta_day
        # Loop over (any) remaining events in time domain and add data to data_sum
        for xx in self.tdomain.datetimes[1:]:
            first_day=xx[0]
            last_day=xx[-1]
            print('first_day,last_day: {0!s}, {1!s}'.format(first_day,last_day))
            time1=datetime.datetime(first_day.year,first_day.month,first_day.day)
            time2=datetime.datetime(first_day.year,first_day.month,first_day.day,23,59,59)
            while time1<=last_day:
                time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
                with iris.FUTURE.context(cell_datetime_objects=True):
                    x1=self.data_in.extract(time_constraint)
                x1=x1.concatenate_cube()
                x1ma=np.ma.array(x1.data)
                x1ma_mask_numerical=np.where(x1ma.mask,0,1)
                x1ma_times_mask=x1ma.data*x1ma_mask_numerical
                data_sum+=x1ma_times_mask
                kount+=x1ma_mask_numerical
                print('time1,time2: {0!s}, {1!s}'.format(time1,time2))
                #print('type(x1.data): {0!s}'.format(type(x1.data)))
                #print(kount.shape,x1ma_mask_numerical.shape)
                time1+=timedelta_day
                time2+=timedelta_day
        # Divide by kount
        #print('kount: {0!s}'.format(kount))
        print('kount.min,kount.max: {0!s}, {1!s}'.format(kount.min(),kount.max()))
        data_mean=data_sum/kount
        # Create iris cube using metadata from first day of input data
        x10=create_cube(data_mean,cube1)
        x10.remove_coord('time')
        x10.add_dim_coord(time_coord,0)
        # Add cell method to describe diurnal cycle
        cm=iris.coords.CellMethod('point','time',comments='mean diurnal cycle over time domain '+self.tdomain.idx)
        x10.add_cell_method(cm)
        # Create mean_dc attribute
        self.mean_dc=x10
        #
        if double:
            print('Create a double diurnal cycle (ie two identical days)')
            # Create a new time axis for 2 Jan year 1
            times_datetime=[datetime.datetime(1,1,2,xx.hour,xx.minute,xx.second) for xx in x2]
            times_val=[time_units.date2num(xx) for xx in times_datetime]
            time_coord=iris.coords.DimCoord(times_val,standard_name='time',units=time_units)
            print('times_datetime: {0!s}'.format(times_datetime))
            print('times_val: {0!s}'.format(times_val))
            print('time_coord: {0!s}'.format(time_coord))
            # Create copy of cube of diurnal cycle
            x11=self.mean_dc.copy()
            # Apply new time axis
            x11.remove_coord('time')
            x11.add_dim_coord(time_coord,0)
            # Create a cube list of the two diurnal cycles and concatenate
            x12=iris.cube.CubeList([self.mean_dc,x11])
            x13=x12.concatenate_cube()
            # Overwrite mean_dc attribute
            self.mean_dc=x13
            
        # Save diurnal cycle
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.mean_dc,self.fileout_dc)
            
#==========================================================================

class TimeFilter(object):
    
    """Time filter using rolling_window method of iris cube.

    Called from filter.py.

    Assumes input data has equally spaced time intervals

    Attributes:

    self.weights : 1-d numpy array of filter weights

    self.nn : integer value of order of filter.

    self.nweights : integer number of weights (length of self.weights
    array.  Must be odd.  Filtered output will be a nweight running
    mean of the input data, using self.weights.

    self.nn : Equal to (nweights-1)/2, e.g., if nweights=61, nn=30.
    Filtered output will be a 61-point weighted mean of the 30 input
    data points before the current time, the current time itself, and
    the 30 intput data points after the current time.

    self.data_out : iris cube of filtered output data.  Length of time
    dimension is typically a convenient block of time, e.g., 1 year
    for daily data.

    self.frequency : string to denote frequency of input (and output)
    data, e.g., 'd' for daily data.

    self.timeout1 : datetime object for start time of self.data_out.
    
    self.timeout2 : datetime object for end time of self.data_out.
    
    self.data_in : iris cube of input data to be filtered.  Length of
    time dimension is length of time dimension of self.data_out +
    2*self.nn (self.nn at the beginning, and self.nn at the end).
    
    self.timein1 : datetime object for start time of self.data_in.
    
    self.timein2 : datetime object for end time of self.data_in.

    self.filein1 : path name for file(s) of input data.
    
    self.fileout1 : path name for file of output (filtered) data.
    
    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.level=descriptor['level']
        self.filter=descriptor['filter']
        self.file_weights=descriptor['file_weights']
        self.f_weights()
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.source=descriptor['source']
        source_info(self)
        self.filepre=descriptor['filepre']
        self.filein1=os.path.join(self.basedir,self.source,'std',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.wildcard+'.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.filein1,self.name)
        xx=self.source.split('_')
        self.frequency=xx[2]
        if self.frequency=='d':
            self.timedelta=datetime.timedelta(days=self.nn)
        elif self.frequency=='h':
            self.timedelta=datetime.timedelta(hours=self.nn)
        else:
            raise ToDoError('data time interval is not days or hours - need more code!')
        if self.verbose:
            print(self)

    def __repr__(self):
        return 'TimeFilter({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'TimeFilter instance \n'+\
                'filter: {0.filter!s} \n'+\
                'nn: {0.nn!s} \n'+\
                'nweights: {0.nweights!s} \n'+\
                'weights: {0.weights!s} \n'+\
                'filein1: {0.filein1!s} \n'+\
                'frequency: {0.frequency!s} \n'+\
                'data_in: {0.data_in!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'TimeFilter instance'

    def f_weights(self):
        """Create array of file weights.

        Read in file weights from ASCII text file.  Must be in format:
        Line 0: information on filter (not actually used here)
        Remaining lines: each line has a single float filter weight.

        If nlines is number of lines in the filter file, there are
        nweights=nlines-1 filter weights.

        nweights must be odd.

        nn=(nfilter-1)/2

        Create nweights, nn and weights attributes.

        """
        # Read ASCII filter weights
        f1=open(self.file_weights)
        lines=f1.readlines()
        # Discard first (information) line
        lines2=lines[1:]
        self.nweights=len(lines2)
        # Check nweights is odd
        if divmod(self.nweights,2)[1]!=1:
            raise ValueError('Error: self.nweights must be odd.')
        self.nn=divmod(self.nweights,2)[0]
        weights=[float(xx) for xx in lines2]
        self.weights=np.array(weights)

    def time_filter(self,subtract=False):
        """Filter using the rolling_window cube method and save data.

        If subtract is False, the filtered data set is just returned.

        If subtract is True, then the filtered data is subtracted from
        the original data, and this subtracted data set is returned.
        This option is generally used to create high-pass filtered
        data.  For example, if the filter weights create an N-running
        mean low pass filter data set, this is then subtracted from
        the original data to create a high-pass filtered data set.

        """
        # Set start and end time of output data
        self.timeout1,self.timeout2=block_times(self,verbose=self.verbose)
        # Calculate start and end time of input data
        self.timein1=self.timeout1-self.timedelta
        self.timein2=self.timeout2+self.timedelta
        # Set output file name
        subtract_string=''
        if subtract:
            subtract_string='minus_'
        if self.outfile_frequency=='year':
            self.fileout1=self.filein1.replace(self.wildcard,subtract_string+self.filter+'_'+str(self.year))
        elif self.outfile_frequency=='month':
            self.fileout1=self.filein1.replace(self.wildcard,subtract_string+self.filter+'_'+str(self.year)+str(self.month).zfill(2))
        else:
            raise ValueError('outfile_frequency not recognised')
        if self.verbose==2:
            ss=h2a+'timein1: {0.timein1!s} \n'+\
                'timeout1: {0.timeout1!s} \n'+\
                'timeout2: {0.timeout2!s} \n'+\
                'timein2: {0.timein2!s} \n'+\
                'fileout1: {0.fileout1!s} \n'+h2b
            print(ss.format(self))
        # Extract input data
        time_constraint=iris.Constraint(time=lambda cell: self.timein1 <=cell<= self.timein2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x2=self.data_in.extract(time_constraint)
        self.data_current=x2.concatenate_cube()
        # Apply filter
        x3=self.data_current.rolling_window('time',iris.analysis.SUM,
                self.nweights,weights=self.weights)
        # Create a cube from this numpy array
        x4=create_cube(conv_float32(x3.data),x3)
        # Subtract filtered data from original data if required
        if subtract:
            # Re-read input data over same time range as output data
            time_constraint2=iris.Constraint(time=lambda cell: self.timeout1 <=cell<= self.timeout2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x2a=self.data_in.extract(time_constraint2)
            x2b=x2a.concatenate_cube()
            # Subtract filtered data
            x5=x2b.data-x4.data
            x4=create_cube(conv_float32(x5),x3)
        # Add a cell method to describe the time filter
        cm=iris.coords.CellMethod('mean','time',comments='time filter: '+subtract_string+self.filter)
        x4.add_cell_method(cm)
        # Set time bounds to None type
        x4.coord('time').bounds=None
        self.data_out=x4
        # Save data
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_out,self.fileout1)

#==========================================================================

class TimeAverage(object):
    
    """Time average data e.g., convert from 3-hourly to daily mean.

    Called from time_average.py.

    Used to change (reduce) time resolution of data, for subsequent
    data analysis.  Because the source attribute of a data set
    contains information on the time resolution, this effectively
    creates data with a different source.

    N.B. To calculate time mean statistics over a particular time
    domain, use the TimeDomStats class instead.

    N.B.  To "increase" time resolution of data, e.g., from weekly to
    daily, use the TimeInterpolate class instead.

    Selected attributes:

    self.source1 : input source, e.g., trmm3b42v7_sfc_3h 3-hourly data.

    self.source2 : output source, e.g., trmm3b42v7_sfc_d daily data.
    The data_source and level_type parts of self.source1 and
    self.source 2 should be identical.

    self.frequency : the frequency part of self.source2, indicating
    what time resolution the input data is to be converted to, e.g.,
    'd' for daily.

    self.year and self.month : year (and month, depending on value of
    self.outfile_frequency) of input data block.

    self.filein1 : path name for input files from self.source1.
    Probably contains wild card characters.

    self.data_in : iris cube list of all input data

    self.cube_in : input cube of data from self.source1 for the
    current time block.

    self.cube_out : output cube of data for current time block to be
    saved under self.source2.

    self.fileout1 : path name for output file under self.source2.

    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.level=descriptor['level']
        self.source1=descriptor['source1']
        self.source2=descriptor['source2']
        self.source=self.source2
        source_info(self)
        self.filein1=os.path.join(self.basedir,self.source1,'std',self.var_name+'_'+str(self.level)+'_'+self.wildcard+'.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.filein1,self.name)
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'TimeAverage({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'TimeAverage instance \n'+\
                'data_in: {0.data_in!s} \n'+\
                'source1: {0.source1!s} \n'+\
                'source2: {0.source2!s} \n'+\
                'frequency: {0.frequency!s} \n'+\
                'filein1: {0.filein1!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'Interpolate instance'

    def f_time_average(self,method=1):
        """Time average data."""
        # Extract input data for current block of time
        time1,time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time = lambda cell: time1 <= cell <= time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        self.cube_in=x1.concatenate_cube()
        time_units=self.cube_in.coord('time').units
        
        if self.frequency=='d':
            # Creating daily average data
            if method==1:
                print('f_time_average: Method 1')
                # Method 1 loops over days, creates a time constraint for
                # each day, extracts data for that day, then averages for that
                # day, then appends to a cube list, then finally merges the
                # cube list to create a cube of daily averaged data.  It is
                # very slow.
                timedelta_day=datetime.timedelta(days=1)
                timedelta_minute=datetime.timedelta(seconds=60)
                timec1=time1
                # Create empty CubeList
                x10=iris.cube.CubeList([])
                while timec1<time2:
                    # Extract data over current day
                    timec2=timec1+timedelta_day-timedelta_minute
                    print(timec1,timec2)
                    time_constraintc=iris.Constraint(time = lambda cell: timec1 <= cell <= timec2)
                    with iris.FUTURE.context(cell_datetime_objects=True):
                        x1=self.data_in.extract(time_constraintc)
                    x2=x1.concatenate_cube()
                    # Calculate daily mean
                    x3=x2.collapsed('time',iris.analysis.MEAN)
                    # Reset auxiliary time coordinate for current day at 00 UTC
                    timec_val=time_units.date2num(timec1)
                    timec_coord=iris.coords.DimCoord(timec_val,standard_name='time',units=time_units)
                    x3.remove_coord('time')
                    x3.add_aux_coord(timec_coord)
                    # Append current daily mean to cube list
                    x10.append(x3)
                    # Increment time
                    timec1+=timedelta_day
                x11=x10.merge_cube()
            elif method==2:
                print('f_time_average: Method 2')
                # Method 2 slices the input cube numpy array to get a
                # different numpy array for each time of day, then
                # adds them together, then divides by number of times
                # of day to get daily mean. Order(1000) faster than method 1
                # Method 2 depends on the time values being equally spaced,
                # and there being no missing data (in time)
                #
                # Find time resolution of input data
                xx=self.source1.split('_')
                source1_frequency=xx[2]
                if source1_frequency[-1]!='h':
                    raise ToDoError('Need to code up for input data other than hourly.')
                if source1_frequency=='h':
                    npd=24
                else:
                    npd=int(24/int(source1_frequency[:-1]))
                print('source1_frequency,npd: {0!s}, {1!s}'.format(source1_frequency,npd))
                # Check input data is well formed
                dim_coord_names=[xx.var_name for xx in self.cube_in.dim_coords]
                time_index=dim_coord_names.index('time')
                print('time_index: {0!s}'.format(time_index))
                if time_index!=0:
                    raise ToDoError('Code below only works if time is first dimension.')
                ntime=self.cube_in.shape[time_index]
                nday=int(ntime/npd)
                print('ntime,nday: {0!s}, {1!s}'.format(ntime,nday))
                if nday!=ntime/npd:
                    raise ValueError('Input data is not integer number of days.')
                # Slice numpy array, one slice per time of day.
                # Then calculate daily means
                x1=self.cube_in.data
                x2=[x1[ii:ntime:npd,...] for ii in range(npd)]
                x3=x2[0]
                for xx in x2[1:]:
                    x3+=xx
                x4=x3/npd
                # Create new time axis for daily mean data (00 UTC each day)
                timec=time1
                time_vals=[]
                timedelta_day=datetime.timedelta(days=1)
                while timec<time2:
                    time_vals.append(time_units.date2num(timec))
                    timec+=timedelta_day
                time_coord=iris.coords.DimCoord(time_vals,standard_name='time',units=time_units)
                # Create new cube of daily mean
                x11=create_cube(x4,self.cube_in,new_axis=time_coord)
                cm=iris.coords.CellMethod('point','time',comments='daily mean from f_time_mean method 2')
                x11.add_cell_method(cm)
            else:
                raise ValueError('Invalid method option.')
        else:
            raise ToDoError('Need code to average over something other than daily.')
        # Convert units for selected data sources
        if self.source1 in ['trmm3b42v7_sfc_3h',] and self.source2 in ['trmm3b42v7_sfc_d',]:
            print("Converting TRMM precipitation from 3-hourly in 'mm hr-1' to daily mean in 'mm day-1'")
            x11.convert_units('mm day-1')
        self.cube_out=x11
        # Save time averaged cube
        if self.outfile_frequency=='year':
            x2=str(self.year)
        elif self.outfile_frequency=='month':
            x2=str(self.year)+str(self.month).zfill(2)
        else:
            raise ValueError('outfile_frequency not recognised')
        self.fileout1=os.path.join(self.basedir,self.source2,'std',self.var_name+'_'+str(self.level)+'_'+x2+'.nc')
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.cube_out,self.fileout1)
        
#==========================================================================

class Interpolate(object):
    
    """Interpolate data using iris.analysis.interpolate.

    Called from interpolate.py.

    N.B.  To decrease time resolution of data, e.g., from 3-hourly to
    daily, use the TimeAverage class instead.

    Selected attributes:

    self.source1 : input source, e.g., sstrey_sfc_7d weekly data.

    self.source2 : output source, e.g., sstrey_sfc_d daily data.  The
    data_source and level_type parts of self.source1 and self.source 2
    should be identical.

    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.file_data_in=descriptor['file_data_in']
        self.file_data_out=descriptor['file_data_out']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.source1=descriptor['source1']
        self.source2=descriptor['source2']
        self.source=self.source2
        source_info(self)
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.file_data_in,self.name)
        # Get first time in input data
        x1=self.data_in[0].coord('time')[0]
        x2=x1.cell(0)[0]
        self.time1=x1.units.num2date(x2)
        # Get last time in input data
        x1=self.data_in[-1].coord('time')[-1]
        x2=x1.cell(0)[0]
        self.time2=x1.units.num2date(x2)
        self.time_units=x1.units
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'Interpolate({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'Interpolate instance \n'+\
                'file_data_in: {0.file_data_in!s} \n'+\
                'data_in: {0.data_in!s} \n'+\
                'time1: {0.time1!s} \n'+\
                'time2: {0.time2!s} \n'+\
                'time_units: {0.time_units!s} \n'+\
                'source1: {0.source1!s} \n'+\
                'source2: {0.source2!s} \n'+\
                'file_data_out: {0.file_data_out!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'Interpolate instance'

    def f_interpolate_time(self):
        """Interpolate over time."""
        # Ensure time interval to interpolate onto is within input time interval
        if self.time1_out<self.time1:
            self.time1_out=self.time1
            print('Correcting time1_out to be within bounds: {0.time1_out!s}'.format(self))
        if self.time2_out>self.time2:
            self.time2_out=self.time2
            print('Correcting time2_out to be within bounds: {0.time2_out!s}'.format(self))
        # Interpolate to daily data
        if self.frequency=='d':
            # Set times to interpolate onto
            self.time1_out_val=self.time_units.date2num(self.time1_out)
            self.time2_out_val=self.time_units.date2num(self.time2_out)
            if 'day' in self.time_units.name:
                time_diff=1
                self.sample_points=[('time',np.arange(self.time1_out_val,self.time2_out_val,time_diff))]
            else:
                raise ToDoError('Need code for time units that are not in days.')
            # Create cube (from cube list) of input data for interpolation
            # Times interval for this cube must completely contain the
            #   output interval (time_val1 to time_val2) for interpolation,
            #   if possible, otherwise there will be extrapolation at ends
            # Create a timedelta of default 25 days to account for this
            self.timedelta=datetime.timedelta(days=25)
            self.time1_in=self.time1_out-self.timedelta
            self.time2_in=self.time2_out+self.timedelta
            ss=h2a+'f_time_interpolate. \n'+\
                'timedelta: {0.timedelta!s} \n'+\
                'time1_in: {0.time1_in!s} \n'+\
                'time1_out: {0.time1_out!s} \n'+\
                'time2_out: {0.time2_out!s} \n'+\
                'time2_in: {0.time2_in!s} \n'+\
                'time1_out_val: {0.time1_out_val!s} \n'+\
                'time2_out_val: {0.time2_out_val!s} \n'+h2b
            if self.verbose==2:
                print(ss.format(self))
            time_constraint=iris.Constraint(time=lambda cell: self.time1_in <=cell<= self.time2_in)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            self.cube_in=x1.concatenate_cube()
            # Interpolate in time
            self.cube_out=iris.cube.Cube.interpolate(self.cube_in,self.sample_points,scheme=iris.analysis.Linear())
            # Add cell method to describe linear interpolation
            cm=iris.coords.CellMethod('point','time',comments='linearly interpolated from weekly to daily time dimension')
            self.cube_out.add_cell_method(cm)
            # Save interpolated data
            fileout=replace_wildcard_with_time(self,self.file_data_out)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.cube_out,fileout)
        else:
            raise ToDoError('Need code for interpolation to other than daily data.')
        

#==========================================================================

class Hovmoller(object):
    
    """Create a Hovmoller object.

    Called from hovmoller.py.

    Selected attributes:

    self.data_in : iris cube list of all input data.

    self.data_hov_current : iris cube of Hovmoller of current data
    (e.g., year).

    self.data_hov : iris cube list of all Hovmoller data.

    self.file_data_in : path name for file(s) of input data.  Contains a
    wild card * character, which will be replaced by, e.g., year
    numbers (if self.outfile_frequency is 'year').

    self.file_data_hov : path name for file(s) of Hovmoller data.
    Contains a wild card * character, which will be replaced by, e.g.,
    year numbers (if self.outfile_frequency is 'year').
    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.source=descriptor['source']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        source_info(self)
        self.level=descriptor['level']
        self.filepre=descriptor['filepre']
        self.band_name=descriptor['band_name']
        self.band_val1=descriptor['band_val1']
        self.band_val2=descriptor['band_val2']
        self.file_data_in=os.path.join(self.basedir,self.source,'std',self.var_name+'_'+str(self.level)+self.filepre+'_'+self.wildcard+'.nc')
        self.strhov='_hov_'+self.band_name[:3]+'_'+str(self.band_val1)+'_'+str(self.band_val2)
        self.file_data_hov=os.path.join(self.basedir,self.source,'processed',self.var_name+'_'+str(self.level)+self.filepre+self.strhov+'_'+self.wildcard+'.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.file_data_in,self.name)
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'Hovmoller({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'Hovmoller instance \n'+\
                'file_data_in: {0.file_data_in!s} \n'+\
                'data_in: {0.data_in!s} \n'+\
                'source: {0.source!s} \n'+\
                'band_name: {0.band_name!s} \n'+\
                'band_val1: {0.band_val1!s} \n'+\
                'band_val2: {0.band_val2!s} \n'+\
                'file_data_hov: {0.file_data_hov!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'Hovmoller instance'

    def f_hovmoller(self):
        """Create cube of Hovmoller data and save.

        Create data_hov_current attribute.
        
        """
        # Extract input data for current time block, and 
        # for dimension band_name, between band_val1 and band_val1
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        if self.band_name=='latitude':
            band_constraint=iris.Constraint(latitude=lambda cell: self.band_val1 <=cell<= self.band_val2)
        elif self.band_name=='longitude':
            band_constraint=iris.Constraint(longitude=lambda cell: self.band_val1 <=cell<= self.band_val2)
        else:
            raise ValueError('Invalid band_name.')
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint & band_constraint)
        x2=x1.concatenate_cube()
        self.cube=x2
        # Average over dimension band_name
        x3=x2.collapsed(self.band_name,iris.analysis.MEAN)
        # Add a cell method to further describe averaging
        str1='Mean over {0.band_name!s} {0.band_val1!s} to {0.band_val2!s}'.format(self)
        cm=iris.coords.CellMethod('mean',str1)
        if self.verbose:
            print(cm)
        x3.add_cell_method(cm)
        self.data_hov_current=x3
        # Save Hovmoller data
        fileout=replace_wildcard_with_time(self,self.file_data_hov)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_hov_current,fileout)

    def f_read_hovmoller(self):
        """Read previously calculated Hovmoller data.

        Create data_hov attribute.
        """
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_hov=iris.load(self.file_data_hov,self.name)

#==========================================================================

class Wind(object):
    
    """Create a Wind object.

    Called from wind.py.

    Wrapper for windspharm VectorWind object to handle file i/o.

    Attributes (following VectorWind):

    self.uwnd : iris cube of u wind
    self.vwnd : iris cube of v wind
    self.psi : iris cube of streamfunction
    self.chi : iris cube of velocity potential
    self.vrt : iris cube of relative vorticity
    self.div : iris cube of divergence
    self.wndspd : iris cube of wind speed

    self.flag_psi : Boolean flag to compute streamfunction
    or not.

    Similarly, self.flag_chi, self.flag_vrt, self.flag_div

    self.file_data : path name for file(s) of input data.  Contains a
    wild card * character, which will be replaced by, e.g., year
    numbers (if self.outfile_frequency is 'year').  Contains a dummy
    string'VARNAME' to be replaced by 'uwnd', 'vwnd', 'psi', etc.

    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.source=descriptor['source']
        source_info(self)
        self.level=descriptor['level']
        self.filepre=descriptor['filepre']
        self.file_data=os.path.join(self.basedir,self.source,'std','VAR_NAME_'+str(self.level)+self.filepre+'_'+self.wildcard+'.nc')
        # uwnd
        self.var_name_uwnd='uwnd'
        self.name_uwnd=var_name2long_name[self.var_name_uwnd]
        self.file_data_uwnd=self.file_data.replace('VAR_NAME',self.var_name_uwnd)
        # vwnd
        self.var_name_vwnd='vwnd'
        self.name_vwnd=var_name2long_name[self.var_name_vwnd]
        self.file_data_vwnd=self.file_data.replace('VAR_NAME',self.var_name_vwnd)
        # psi
        self.flag_psi=descriptor['flag_psi']
        if self.flag_psi:
            self.var_name_psi='psi'
            self.name_psi=var_name2long_name[self.var_name_psi]
            self.file_data_psi=self.file_data.replace('VAR_NAME',self.var_name_psi)
        # chi
        self.flag_chi=descriptor['flag_chi']
        if self.flag_chi:
            self.var_name_chi='chi'
            self.name_chi=var_name2long_name[self.var_name_chi]
            self.file_data_chi=self.file_data.replace('VAR_NAME',self.var_name_chi)
        # vrt
        self.flag_vrt=descriptor['flag_vrt']
        if self.flag_vrt:
            self.var_name_vrt='vrt'
            self.name_vrt=var_name2long_name[self.var_name_vrt]
            self.file_data_vrt=self.file_data.replace('VAR_NAME',self.var_name_vrt)
        # div
        self.flag_div=descriptor['flag_div']
        if self.flag_div:
            self.var_name_div='div'
            self.name_div=var_name2long_name[self.var_name_div]
            self.file_data_div=self.file_data.replace('VAR_NAME',self.var_name_div)
        # wndspd
        self.flag_wndspd=descriptor['flag_wndspd']
        if self.flag_wndspd:
            self.var_name_wndspd='wndspd'
            self.name_wndspd=var_name2long_name[self.var_name_wndspd]
            self.file_data_wndspd=self.file_data.replace('VAR_NAME',self.var_name_wndspd)
        #
        self.source=descriptor['source']
        source_info(self)
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_uwnd=iris.load(self.file_data_uwnd,self.name_uwnd)
            self.data_vwnd=iris.load(self.file_data_vwnd,self.name_vwnd)
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'Wind({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'Wind instance \n'+\
                'source: {0.source!s} \n'+\
                'file_data: {0.file_data!s} \n'+\
                'file_data_uwnd: {0.file_data_uwnd!s} \n'+\
                'file_data_vwnd: {0.file_data_vwnd!s} \n'
            if self.flag_psi:
                ss+='file_data_psi: {0.file_data_psi!s} \n'
            if self.flag_chi:
                ss+='file_data_chi: {0.file_data_chi!s} \n'
            if self.flag_vrt:
                ss+='file_data_vrt: {0.file_data_vrt!s} \n'
            if self.flag_div:
                ss+='file_data_div: {0.file_data_div!s} \n'
            if self.flag_wndspd:
                ss+='file_data_wndspd: {0.file_data_wndspd!s} \n'
            ss+=h1b
            return(ss.format(self))
        else:
            return 'Wind instance'

    def f_wind(self):
        """Calculate and save streamfunction etc.

        windspharm methods always return output with latitude running
        from north to south.  If input data runs south to north,
        switch the output data to also run from south to north, with
        lat_direction.
        """
        # Set current time range
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        # Read uwnd and vwnd for current time range
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_uwnd.extract(time_constraint)
            x2=self.data_vwnd.extract(time_constraint)
        self.uwnd=x1.concatenate_cube()
        self.vwnd=x2.concatenate_cube()
        # Find value of south2north
        lat_coord=self.uwnd.coord('latitude')
        lat_beg=lat_coord.points[0]
        lat_end=lat_coord.points[-1]
        if lat_beg<lat_end:
            self.south2north=True
        else:
            self.south2north=False
        print('lat_beg, lat_end, south2north: {0!s}, {1!s}, {2!s} '.format(lat_beg,lat_end,self.south2north))
        # Create VectorWind instance
        self.ww=VectorWind(self.uwnd,self.vwnd)
        # Both psi and chi
        if self.flag_psi and self.flag_chi:
            self.psi,self.chi=self.ww.sfvp()
            self.psi.var_name=self.var_name_psi
            self.chi.var_name=self.var_name_chi
            if self.south2north:
                self.psi=lat_direction(self.psi,'s2n')
                self.chi=lat_direction(self.chi,'s2n')
            fileout1=replace_wildcard_with_time(self,self.file_data_psi)
            fileout2=replace_wildcard_with_time(self,self.file_data_chi)
            print('fileout1: {0!s}'.format(fileout1))
            print('fileout2: {0!s}'.format(fileout2))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.psi,fileout1)
                iris.save(self.chi,fileout2)
        # or psi only
        elif self.flag_psi:
            self.psi=self.ww.streamfunction()
            self.psi.var_name=self.var_name_psi
            if self.south2north:
                self.psi=lat_direction(self.psi,'s2n')
            fileout=replace_wildcard_with_time(self,self.file_data_psi)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.psi,fileout)
        # or chi only
        elif self.flag_chi:
            self.chi=self.ww.velocitypotential()
            self.chi.var_name=self.var_name_chi
            if self.south2north:
                self.chi=lat_direction(self.chi,'s2n')
            fileout=replace_wildcard_with_time(self,self.file_data_chi)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.chi,fileout)
        # Both vrt and div
        if self.flag_vrt and self.flag_div:
            self.vrt,self.div=self.ww.vrtdiv()
            self.vrt.var_name=self.var_name_vrt
            self.div.var_name=self.var_name_div
            if self.south2north:
                self.vrt=lat_direction(self.vrt,'s2n')
                self.div=lat_direction(self.div,'s2n')
            fileout1=replace_wildcard_with_time(self,self.file_data_vrt)
            fileout2=replace_wildcard_with_time(self,self.file_data_div)
            print('fileout1: {0!s}'.format(fileout1))
            print('fileout2: {0!s}'.format(fileout2))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.vrt,fileout1)
                iris.save(self.div,fileout2)
        # or vrt only
        elif self.flag_vrt:
            self.vrt=self.ww.vorticity()
            self.vrt.var_name=self.var_name_vrt
            if self.south2north:
                self.vrt=lat_direction(self.vrt,'s2n')
            fileout=replace_wildcard_with_time(self,self.file_data_vrt)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.vrt,fileout)
        # or div only
        elif self.flag_div:
            self.div=self.ww.divergence()
            self.div.var_name=self.var_name_div
            if self.south2north:
                self.div=lat_direction(self.div,'s2n')
            fileout=replace_wildcard_with_time(self,self.file_data_div)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.div,fileout)
        # wndspd
        if self.flag_wndspd:
            self.wndspd=self.ww.magnitude()
            self.wndspd.var_name=self.var_name_wndspd
            if self.south2north:
                self.wndspd=lat_direction(self.wndspd,'s2n')
            fileout=replace_wildcard_with_time(self,self.file_data_wndspd)
            print('fileout: {0!s}'.format(fileout))
            with iris.FUTURE.context(netcdf_no_unlimited=True):
                iris.save(self.wndspd,fileout)

#==========================================================================

class AnnualCycle(object):
    
    """Calculate and subtract annual cycle.

    Called from anncycle.py

    Assumes input data has equally spaced time intervals.

    Attributes:

    self.frequency : string to denote frequency of input (and output)
    data, e.g., 'd' for daily data.

    self.year1 and self.year2 : integers.  The annual cycle will be
    calculated using input data from 0000:00 UTC 1 Jan self.year1 to
    2359:59 31 Dec self.year2.

    self.data_in : iris cube list of all input data.

    self.data_anncycle_raw : iris cube of 'raw' annual cycle.  The
    data for e.g. 5 Jan is a simple mean of all the input data for 5
    Jan.

    self.data_anncycle_smooth : iris cube of smoothed annual cycle.
    Calculated from mean plus first self.nharm annual harmonics of
    self.data_anncycle_raw.

    self.nharm : integer number of annual harmonics to retain for
    smoothed annual cycle.

    self.data_anncycle_rm : iris cube list of anomaly data with annual
    cycle subtracted.
    
    self.file_data_in : path name for file(s) of input data.  Contains a
    wild card ???? character, which will be replaced by, e.g., year
    numbers (if self.outfile_frequency is 'year').
    
    self.file_anncycle_raw : path name for file of raw annual cycle.
    
    self.file_anncycle_smooth : path name for file of smoothed annual
    cycle.
    
    self.file_anncycle_rm : path name for file(s) of data with
    smoothed annual cycle subtracted.  Contains a wild card ????
    character, which will be replaced by, e.g., year numbers (if
    self.outfile_frequency is 'year').

    self.time1: If False, calculate anomalies from annual cycle from
    beginning of data set.  If datetime.datetime object, calculate
    anomalies from annual cycle from this time.
    
    self.time2: If False, calculate anomalies from annual cycle up to
    end of data set.  If datetime.datetime object, calculate anomalies
    from annual cycle up to this time.
    
    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.source=descriptor['source']
        source_info(self)
        self.basedir=descriptor['basedir']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        self.level=descriptor['level']
        self.year1=descriptor['year1']
        self.year2=descriptor['year2']
        self.time1=descriptor['time1']
        self.time2=descriptor['time2']
        self.file_data_in=os.path.join(self.basedir,self.source,'std',
              self.var_name+'_'+str(self.level)+'_'+self.wildcard+'.nc')
        self.file_anncycle_raw=os.path.join(self.basedir,self.source,
              'processed',self.var_name+'_'+str(self.level)+'_ac_raw_'+\
              str(self.year1)+'_'+str(self.year2)+'.nc')
        self.file_anncycle_smooth=os.path.join(self.basedir,self.source,
              'processed',self.var_name+'_'+str(self.level)+'_ac_smooth_'+\
              str(self.year1)+'_'+str(self.year2)+'.nc')
        self.file_anncycle_smooth_leap=self.file_anncycle_smooth.replace('.','_leap.')
        self.file_anncycle_rm=os.path.join(self.basedir,self.source,'std',
              self.var_name+'_'+str(self.level)+'_rac_'+self.wildcard+'.nc')
        self.nharm=descriptor['nharm']
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in=iris.load(self.file_data_in,self.name)
        # Get first time in input data if self.time1 not externally set
        if not self.time1:
            x1=self.data_in[0].coord('time')[0]
            x2=x1.cell(0)[0]
            self.time1=x1.units.num2date(x2)
        # Get last time in input data if self.time2 not externally set
        if not self.time2:
            x1=self.data_in[-1].coord('time')[-1]
            x2=x1.cell(0)[0]
            self.time2=x1.units.num2date(x2)
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'AnnualCycle({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'AnnualCycle instance \n'+\
                'year1: {0.year1!s} \n'+\
                'year2: {0.year2!s} \n'+\
                'time1: {0.time1!s} \n'+\
                'time2: {0.time2!s} \n'+\
                'nharm: {0.nharm!s} \n'+\
                'file_data_in: {0.file_data_in!s} \n'+\
                'data_in: {0.data_in!s} \n'+\
                'file_anncycle_raw: {0.file_anncycle_raw!s} \n'+\
                'file_anncycle_smooth: {0.file_anncycle_smooth!s} \n'+\
                'file_anncycle_smooth_leap: {0.file_anncycle_smooth_leap!s} \n'+\
                'file_anncycle_rm: {0.file_anncycle_rm!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'AnnualCycle instance'

    def f_anncycle_raw_old(self):
        """Create raw annual cycle and write to file.

        This will be assigned to year 1, i.e. AD 1.  NB year 1 is not
        a leap year.

        Code is rather slow as it loops over each day of the year, and
        reads in all data (e.g., 1 Jan's) for that day and averages
        over them.  Superceded by new version of f_anncycle_raw, but
        have left the code here as it has some good iris pointers.

        Create data_anncycle_raw attribute.
        
        """
        raise DeprecationWarning('Use f_anncycle_raw instead')
        if self.frequency!='d':
            raise ToDoError('Annual cycle presently only coded up for daily data.')
        # Set first day as 1 Jan year 1 (time coord will be relative to this)
        time_first=datetime.datetime(year=1,month=1,day=1)
        # Set current time to be 1 Jan year 1.
        time_sample=datetime.datetime(year=1,month=1,day=1)
        # Set last time to loop over to be 31 Dec year 1.
        time_sample_last=datetime.datetime(year=1,month=12,day=31)
        #time_sample_last=datetime.datetime(year=1,month=1,day=3)
        # Time increment for annual cycle is 1 day
        timedelta=datetime.timedelta(days=1)
        # Create empty CubeList for annual cycle
        x10=iris.cube.CubeList([])
        # Create time constraint to only use data between self.year1 and self.year2
        pdt1=PartialDateTime(year=self.year1)
        pdt2=PartialDateTime(year=self.year2)
        time_constraint_range=iris.Constraint(time=lambda cell: pdt1<=cell<=pdt2)
        # Loop over days of year
        while time_sample<=time_sample_last:
            # Extract all data for current day of year
            pdtc=PartialDateTime(month=time_sample.month,day=time_sample.day)
            time_constraintc=iris.Constraint(time=pdtc)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint_range & time_constraintc)
            x2=x1.merge_cube()
            # Calculate mean for current day of year
            x3=x2.collapsed('time',iris.analysis.MEAN)
            # Add a cell method to describe this action in more detail
            cm=iris.coords.CellMethod('mean','time',comments='raw annual cycle '+str(self.year1)+'-'+str(self.year2))
            x3.add_cell_method(cm)
            # Remove now inappropriate time coordinate
            x3.remove_coord('time')
            # Add auxiliary time coordinate for current day of year
            time_diff=time_sample-time_first
            time_val=time_diff.days
            time_units='days since '+str(time_first)
            tcoord=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
            x3.add_aux_coord(tcoord)
            # Append mean for current day of year to annual cycle
            x10.append(x3)
            print('time_sample,time_val: {0!s}, {1!s}'.format(time_sample,time_val))
            # Increment day of year
            time_sample+=timedelta
        x11=x10.merge_cube()
        self.data_anncycle_raw=x11
        # Save raw annual cycle
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_anncycle_raw,self.file_anncycle_raw)

    def f_anncycle_raw(self):
        """Create raw annual cycle and write to file.

        The raw annual cycle consists of a value for each time (e.g.,
        day) of the year by averaging over all years, e.g., creates a
        value for 1 Jan by averaging over all the 1 Jan days from all
        the year.
        
        This will be assigned to year 1, i.e. AD 1.  NB year 1 is not
        a leap year.

        Create data_anncycle_raw attribute.
        
        """
        if self.frequency!='d':
            raise ToDoError('Annual cycle presently only coded up for daily data.')
        kyear=0
        # First year
        yearc=self.year1
        kyear+=1
        print('kyear: {0!s}'.format(kyear))
        # Extract 1 Jan to 28 Feb of current year
        time1=datetime.datetime(yearc,1,1)
        time2=datetime.datetime(yearc,2,28)
        print(time1,time2)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        janfeb=x1.concatenate_cube()
        janfeb.remove_coord('time')
        # Extract 1 Mar to 31 Dec of current year
        time1=datetime.datetime(yearc,3,1)
        time2=datetime.datetime(yearc,12,31)
        print(time1,time2)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        mardec=x1.concatenate_cube()
        mardec.remove_coord('time')
        #
        # Loop over remaining years and add contributions
        for yearc in range(self.year1+1,self.year2+1):
            kyear+=1
            print('kyear: {0!s}'.format(kyear))
            # Do not use 1978 for olrinterp as missing data from 16 Mar 1978
            # to 31 Dec 1978.  Just use data from 1979 onwards to calculate
            # annual cycle
            if self.data_source=='olrinterp' and yearc==1978:
                raise ValueError('Cannot use 1978 for olrinterp because of missing data.')
            # Extract 1 Jan to 28 Feb of current year
            time1=datetime.datetime(yearc,1,1)
            time2=datetime.datetime(yearc,2,28)
            print(time1,time2)
            time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            janfebc=x1.concatenate_cube()
            janfebc.remove_coord('time')
            # Extract 1 Mar to 31 Dec of current year
            time1=datetime.datetime(yearc,3,1)
            time2=datetime.datetime(yearc,12,31)
            print(time1,time2)
            time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            mardecc=x1.concatenate_cube()
            mardecc.remove_coord('time')
            # Add contributions for current year
            janfeb=iris.analysis.maths.add(janfeb,janfebc)
            mardec=iris.analysis.maths.add(mardec,mardecc)
        # Divide by number of years to calculate mean
        # Bizarrely, this converts units of temperature from degC to kelvin!
        # So save units and copy them back.
        x1=janfeb.units
        janfeb=iris.analysis.maths.divide(janfeb,kyear)
        mardec=iris.analysis.maths.divide(mardec,kyear)
        janfeb.units=x1
        mardec.units=x1
        # Create and add a time coordinate for year 1, and attributes
        time_first=datetime.datetime(year=1,month=1,day=1)
        time_units='days since '+str(time_first)
        # janfeb
        time_val=np.arange(59)
        tcoord=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
        janfeb.add_dim_coord(tcoord,0)
        janfeb.rename(janfebc.name())
        janfeb.var_name=janfebc.var_name
        janfeb.attributes=janfebc.attributes
        # mardec
        time_val=np.arange(59,365)
        tcoord=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
        mardec.add_dim_coord(tcoord,0)
        mardec.rename(mardecc.name())
        mardec.var_name=mardecc.var_name
        mardec.attributes=mardecc.attributes
        # Create a cubelist then a single cube
        x3=iris.cube.CubeList([janfeb,mardec])
        x4=x3.concatenate_cube()
        # Add a cell method to describe the action of creating annual cycle
        cm=iris.coords.CellMethod('mean','time',comments='raw annual cycle '+str(self.year1)+'-'+str(self.year2))
        x4.add_cell_method(cm)
        # Set data_anncycle_raw attribute
        self.data_anncycle_raw=x4
        # Save raw annual cycle
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_anncycle_raw,self.file_anncycle_raw)

    def f_read_anncycle_raw(self):
        """Read previously created raw annual cycle.

        Create data_anncycle_raw attribute.
        
        """
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_anncycle_raw=iris.load_cube(self.file_anncycle_raw,self.name)

    def f_anncycle_smooth(self):
        """Calculate smoothed annual cycle.

        This is created from the mean and first self.nharm annual
        harmonics of the raw annual cycle.

        f(t_i)= \overline f + \Sum_{k=1}^\{nharm} A_k \cos \omega_k t
                                                + B_k \sin \omega_k t

        \omega_k = 2\pi k / T

        A_k = 2/N \Sum_{k=1}^\{N} f(t_i) \cos \omega_k t_i

        B_k = 2/N \Sum_{k=1}^\{N} f(t_i) \sin \omega_k t_i

        Create data_mean, data_anncycle_smooth,
        data_anncycle_smooth_leap attributes.
        
        """
        # Calculate annual mean
        print('data_anncycle_raw.shape: {0!s}'.format(self.data_anncycle_raw.shape))
        self.data_mean=self.data_anncycle_raw.collapsed('time',iris.analysis.MEAN)
        print('data_mean.shape: {0!s}'.format(self.data_mean.shape))
        # Calculate cosine and sine harmonics
        # Create a (ntime,1) array of itime
        ntime=self.data_anncycle_raw.coord('time').shape[0]
        itime=np.array(np.arange(ntime))
        itime=itime.reshape(itime.shape+(1,))
        print('itime.shape: {0!s}'.format(itime.shape))
        # Create a (1,nharm) array of iharm
        iharm=np.array(np.arange(1,self.nharm+1))
        iharm=iharm.reshape(iharm.shape+(1,))
        iharm=iharm.transpose()
        print('iharm: {0!s}'.format(iharm))
        print('iharm.shape: {0!s}'.format(iharm.shape))
        # Create (ntime,nharm) arrays of cosine and sine waves
        argument=(2*np.pi/float(ntime))*np.dot(itime,iharm)
        cosine=np.cos(argument)
        sine=np.sin(argument)
        print('argument: {0!s}'.format(argument))
        print('cosine: {0!s}'.format(cosine))
        print('cosine.shape: {0!s}'.format(cosine.shape))
        # Duplicate data array with an nharm-length axis and reshape to
        # (nlat,nlon,...,ntime,nharm) in x6
        # NB the nlat,nlon,... could be just nlat,nlon, or eg nlat,lon,nlev
        x1=self.data_anncycle_raw.data
        x2=x1.reshape(x1.shape+(1,))
        print('x2.shape: {0!s}'.format(x2.shape))
        x3=np.ones((1,self.nharm))
        print('x3.shape: {0!s}'.format(x3.shape))
        x4=np.dot(x2,x3)
        print('x4.shape: {0!s}'.format(x4.shape))
        x5=list(x4.shape)
        x5.remove(ntime)
        x5.remove(self.nharm)
        shape1=tuple(x5)+(ntime,)+(self.nharm,)
        # x4 is (ntime,nlat,nlon,...,nharm)
        # Cannot do a simple np.reshape to shape1 (nlat,nlon,...,ntime,nharm)
        # as data is not assigned properly.
        # Create a new array of zeros, then overwrite the data in this
        x6=np.zeros(shape1)
        if len(shape1)>4:
            raise ValueError('Code not able to cope with lat,lon,lev data!')
        else:
            for ii in range(ntime):
                for kk in range(self.nharm):
                    # The :,: below refers to the lat,lon dimensions
                    # If data is lat,lon,lev need to rewrite code
                    x6[:,:,ii,kk]=x4[ii,:,:,kk]
        #x6=x4.reshape(shape1)
        print('x6.shape: {0!s}'.format(x6.shape))
        # Multiply data by cosine and sine waves:
        # f(t_i) cos/sin omega_k t_i
        x7cos=x6*cosine
        x7sin=x6*sine
        # Sum over time axis (index is -2)
        x8cos=x7cos.sum(axis=-2)
        x8sin=x7sin.sum(axis=-2)
        # Multiply by 2/ntime to get (nlat,nlon,...,nharm) array of a and b
        # A_k and B_k coefficients
        a_coeff=2.0*x8cos/float(ntime)
        b_coeff=2.0*x8sin/float(ntime)
        print('a_coeff.shape: {0!s}'.format(a_coeff.shape))
        # Matrix multiply the a,b coefficients by the (nharm,ntime) arrays of
        # cosine and sine harmonic time series to get the contributions to the
        # smoothed annual cycle in a (nlat,nlon,...,ntime) array
        x10cos=np.dot(a_coeff,cosine.transpose())
        x10sin=np.dot(b_coeff,sine.transpose())
        print('x10cos.shape: {0!s}'.format(x10cos.shape))
        # Reshape so it can be added to mean by broadcasting
        shape2=(ntime,)+tuple(x5)
        x11cos=np.zeros(shape2)
        x11sin=np.zeros(shape2)
        for ii in range(ntime):
            x11cos[ii,:,:]=x10cos[:,:,ii]
            x11sin[ii,:,:]=x10sin[:,:,ii]
        print('x11cos.shape: {0!s}'.format(x11cos.shape))
        # Add cosine and sine contributions and time mean to get smoothed
        # annual cycle
        x12=self.data_mean.data+x11cos+x11sin
        # Create a cube from this numpy array
        x13=create_cube(conv_float32(x12),self.data_anncycle_raw)
        # Add a cell method to describe the smoothed annual cycle
        cm=iris.coords.CellMethod('mean','time',comments='smoothed annual cycle: mean + '+str(self.nharm)+' harmonics')
        x13.add_cell_method(cm)
        # Set data_anncycle_smooth attribute
        self.data_anncycle_smooth=x13
        # Save smoothed annual cycle
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_anncycle_smooth,self.file_anncycle_smooth)
        #
        # Create alternate smoothed annual cycle for leap year, with 29 Feb
        # represented by a copy of 28 Feb.
        # Use year 4 for this, as this is a leap year.
        time_units_leap='days since 0004-01-01 00:00:0.0'
        # 1 Jan to 28 Feb of smoothed annual cycle
        time1=datetime.datetime(1,1,1)
        time2=datetime.datetime(1,2,28)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            janfeb=x13.extract(time_constraint)
        time_val=janfeb.coord('time').points
        tcoord2=iris.coords.DimCoord(time_val,standard_name='time',var_name='time',units=time_units_leap)
        janfeb.remove_coord('time')
        janfeb.add_dim_coord(tcoord2,0)
        # 29 Feb only of smoothed annual cycle. Copy of 28 Feb data.
        time1=datetime.datetime(1,2,28)
        time_constraint=iris.Constraint(time=time1)
        with iris.FUTURE.context(cell_datetime_objects=True):
            feb29=x13.extract(time_constraint)
        feb29=iris.util.new_axis(feb29,'time') # Promote singleton aux coord to dim coord
        time_val=np.array([59,]) # Julian day number for 29 Feb
        tcoord2=iris.coords.DimCoord(time_val,standard_name='time',var_name='time',units=time_units_leap)
        feb29.remove_coord('time')
        feb29.add_dim_coord(tcoord2,0)
        # 1 Mar to 31 Dec of smoothed annual cycle
        time1=datetime.datetime(1,3,1)
        time2=datetime.datetime(1,12,31)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            mardec=x13.extract(time_constraint)
        time_val=mardec.coord('time').points+1
        tcoord2=iris.coords.DimCoord(time_val,standard_name='time',var_name='time',units=time_units_leap)
        mardec.remove_coord('time')
        mardec.add_dim_coord(tcoord2,0)
        # Create a cubelist then a single cube
        x1=iris.cube.CubeList([janfeb,feb29,mardec])
        x2=x1.concatenate_cube()
        # Set data_anncycle_smooth_leap attribute
        self.data_anncycle_smooth_leap=x2
        # Save smoothed annual cycle for leap year
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_anncycle_smooth_leap,self.file_anncycle_smooth_leap)

    def f_read_anncycle_smooth(self):
        """Read previously calculated smoothed annual cycle.

        Create data_anncycle_smooth and data_anncycle_smooth_leap
        attributes.
        
        """
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_anncycle_smooth=iris.load_cube(self.file_anncycle_smooth,self.name)
            self.data_anncycle_smooth_leap=iris.load_cube(self.file_anncycle_smooth_leap,self.name)

    def f_subtract_anncycle(self):
        """Subtract smoothed annual cycle from input data to create anomaly data.

        Read in input data and process and write output data (anomaly
        with smoothed annual cycle subtracted) in chunks of
        outfile_frequency, e.g., 'year'.

        Makes allowance for input data not having to start on 1 Jan or
        end on 31 Dec.

        Create data_anncycle_rm attribute.
        
        """
        # Set initial value of current year
        yearc=self.time1.year
        # Set final value of current year
        year_end=self.time2.year
        #
        if self.outfile_frequency=='year':
            # Loop over years
            while yearc<=year_end:
                # If current year is leap year, use
                # self.data_anncycle_smooth_leap, otherwise use
                # self.data_anncycle_smooth.
                if divmod(yearc,4)[1]==0:
                    leap=True
                    smooth_year_number=4
                    anncycle_smooth=self.data_anncycle_smooth_leap
                else:
                    leap=False
                    smooth_year_number=1
                    anncycle_smooth=self.data_anncycle_smooth
                print('yearc,leap: {0!s}, {1!s}'.format(yearc,leap))
                # Set start and end times for input and anncycle data
                time_beg_in=datetime.datetime(yearc,1,1)
                if time_beg_in<self.time1:
                    time_beg_in=self.time1
                time_end_in=datetime.datetime(yearc,12,31)
                if time_end_in>self.time2:
                    time_end_in=self.time2
                time_beg_anncycle=datetime.datetime(smooth_year_number,
                                                    time_beg_in.month,
                                                    time_beg_in.day)
                time_end_anncycle=datetime.datetime(smooth_year_number,
                                                    time_end_in.month,
                                                    time_end_in.day)
                print(time_beg_in,time_end_in,time_beg_anncycle,
                      time_end_anncycle)
                # Extract input data for this (potentially partial) year
                time_constraint=iris.Constraint(time=lambda cell: time_beg_in<=cell<=time_end_in)
                with iris.FUTURE.context(cell_datetime_objects=True):
                    x1=self.data_in.extract(time_constraint)
                data_in=x1.concatenate_cube()
                # Extract anncycle data for this (potentially partial) year
                time_constraint=iris.Constraint(time=lambda cell: time_beg_anncycle<=cell<=time_end_anncycle)
                with iris.FUTURE.context(cell_datetime_objects=True):
                    data_anncycle=anncycle_smooth.extract(time_constraint)
                # Remove time coordinates from input and anncycle cubes so they
                # can be subtracted
                tcoord=data_in.coord('time')
                data_in.remove_coord('time')
                data_anncycle.remove_coord('time')
                # Subtract smoothed annual cycle for current period to create anomaly
                data_anom=iris.analysis.maths.subtract(data_in,data_anncycle)
                # Add back time coordinate and update metadata
                data_anom.add_dim_coord(tcoord,0)
                data_anom.rename(data_in.name())
                data_anom.var_name=data_in.var_name
                data_anom.attributes=data_in.attributes
                # Add a cell method to describe the smoothed annual cycle
                # There is a small list of allowed method names (1st argument)
                # 'point' seems the most appropriate, actions on each point?!
                cm=iris.coords.CellMethod('point','time',comments='anomaly: subtracted smoothed annual cycle: '+str(self.year1)+'-'+str(self.year2)+': mean + '+str(self.nharm)+' harmonics')
                data_anom.add_cell_method(cm)
                # Save anomaly data (anncycle subtracted)
                self.year=yearc
                fileout=replace_wildcard_with_time(self,self.file_anncycle_rm)
                print('fileout: {0!s}'.format(fileout))
                with iris.FUTURE.context(netcdf_no_unlimited=True):
                    iris.save(data_anom,fileout)
                # Increment current year
                yearc+=1
            # Set data_anncycle_rm attribute
            with iris.FUTURE.context(netcdf_promote=True):
                self.data_anncycle_rm=iris.load(self.file_anncycle_rm,self.name)
        else:
            raise ToDoError('Need code for other outfile_frequency values.')

    def f_read_subtract_anncycle(self):
        """Read previously calculatedanomaly data (annual cycle subtracted).

        Create data_anncycle_rm attribute.
        
        """
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_anncycle_rm=iris.load(self.file_anncycle_rm,self.name)
        
#==========================================================================

class GliderMission(object):
    
    """Analyse a glider mission

    Called from, e.g., glider_interp_lon.py

    Selected attributes:

    self.mission : integer id for mission

    self.gliderids : list of integer glider ids for mission

    self.time1 : datetime.datetime object for 00 UTC on day of
    deployment of first glider in mission

    self.time2 : datetime.datetime object for 00 UTC on day after
    recovery of last glider in mission

    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.mission=descriptor['mission']
        self.source_wildcard=descriptor['source_wildcard']
        self.var_name=descriptor['var_name']
        self.name=var_name2long_name[self.var_name]
        if self.mission==31:
            self.gliderids=[579,534,532,620,613]
            self.time1=datetime.datetime(2016,6,30)
            self.time2=datetime.datetime(2016,7,21)
        else:
            raise ValueError('Invalid mission')
        # self.gliders is a dictionary of glider objects
        self.gliders={}
        for gliderid in self.gliderids:
            self.gliders[gliderid]=Glider(gliderid,self.descriptor,verbose=self.verbose)
        self.data_oi_pad_all={}
        self.data_oi_interp_lon={}
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'GliderMission({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'GliderMission instance \n'+\
                'mission: {0.mission!s} \n'+\
                'gliderids: {0.gliderids!s} \n'+\
                'source_wildcard: {0.source_wildcard!s} \n'+\
                'time1: {0.time1!s} \n'+\
                'time2: {0.time2!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'GliderMission instance'

    def f_interp_oi_lon(self,lon1,lon2,delta_lon):
        """Interpolate padded OI data from individual gliders in longitude."""

        x1a=iris.cube.CubeList([]) # var_name
        x2a=iris.cube.CubeList([]) # longitude
        kount=0
        for gliderid in self.gliderids:
            print('gliderid: {0!s}'.format(gliderid))
            # Read and pad OI var_name
            self.gliders[gliderid].f_read_oi(self.var_name,verbose=False)
            self.gliders[gliderid].f_oi_pad(self.var_name,self.time1,self.time2,verbose=False)
            # Read and pad OI longitude
            self.gliders[gliderid].f_read_oi('lon',verbose=False)
            self.gliders[gliderid].f_oi_pad('lon',self.time1,self.time2,verbose=False)
            # Combine padded OI data from all gliders into single iris cube
            glider_coord=iris.coords.DimCoord(kount,var_name='glider')
            # Variable
            x1b=self.gliders[gliderid].data_oi_pad[self.var_name]
            x1b.add_aux_coord(glider_coord)
            x1b.data.fill_value=1e20 # Reset or merge_cube will fail
            x1a.append(x1b)
            # Longitude
            x2b=self.gliders[gliderid].data_oi_pad['lon']
            x2b.add_aux_coord(glider_coord)
            x2b.data.fill_value=1e20 # Reset or merge_cube will fail
            x2a.append(x2b)
            kount+=1
        # Merge cube list of individual gliders to cube of all gliders
        x1c=x1a.merge_cube()
        x2c=x2a.merge_cube()
        # Create data_oi_pad_all attribute dictionary entries
        self.data_oi_pad_all[self.var_name]=x1c
        self.data_oi_pad_all['lon']=x2c
        # Create longitude axis to interpolate onto
        delta=1e-6
        xnew=np.arange(lon1,lon2+delta,delta_lon)
        print('xnew: {0!s}'.format(xnew))
        nlon=len(xnew)
        lon_coord=iris.coords.DimCoord(xnew,standard_name='longitude',var_name='longitude',units='degree_east')
        print('lon_coord: {0!s}'.format(lon_coord))
        # Interpolation
        xx=self.data_oi_pad_all['lon'].data
        yy=self.data_oi_pad_all[self.var_name].data
        # As the longitudes (x values) of each glider are different at each
        # time, depth etc, have to do a separate interpolation over longitude
        # at each time, depth etc.  Ugly nested loops
        if len(yy.shape)!=3:
            raise ValueError('Need some more code if not 3-d data')
        # Create empty new 3-d array to fill with interpolated values
        shape=(nlon,)+yy.shape[1:]
        ndim1=shape[1]
        ndim2=shape[2]
        print('nlon,ndim1,ndim2: {0!s},{1!s},{2!s}'.format(nlon,ndim1,ndim2))
        x3=np.zeros(shape)
        # Loop over other dimensions and fill x3 with interpolated values
        missing_value=1e20
        nglider=len(self.gliderids)
        for idim1 in range(ndim1):
            print('idim1: {0!s}'.format(idim1))
            for idim2 in range(ndim2):
                xx1=xx[:,idim1,idim2]
                yy1=yy[:,idim1,idim2]
                # Get rid of masked data
                xxc=[xx1.data[ii] for ii in range(nglider) if not xx1.mask[ii]]
                yyc=[yy1.data[ii] for ii in range(nglider) if not xx1.mask[ii]]
                n_data_points=len(xxc)
                #print('idim1,idim2,n_data_points: {0!s},{1!s},{2!s}'.format(idim1,idim2,n_data_points))
                #print('xxc: {0!s}'.format(xxc))
                #print('yyc: {0!s}'.format(yyc))
                if n_data_points>1:
                    # Interpolate
                    ynew=np.interp(xnew,xxc,yyc,left=missing_value,right=missing_value)
                elif n_data_points==1:
                    # Only 1 data point.  
                    # Take the data at the 1 point and
                    # extend it over neighbouring points.
                    xspread=3*delta_lon
                    ynew=np.where(abs(xxc[0]-xnew)<xspread,yyc[0],missing_value)
                else:
                    # Zero data points.
                    # Fill ynew with missing values
                    ynew=missing_value*np.ones((nlon,))
                #print('ynew: {0!s}'.format(ynew))
                x3[:,idim1,idim2]=ynew
        # Create iris cube
        oldcube=self.data_oi_pad_all[self.var_name]
        # Create a list of two-lists, each of form [dim_coord,index]
        dim_coords=[[lon_coord,0]]
        kdim=1
        for xx in oldcube.dim_coords[1:]:
            dim_coords.append([xx,kdim])
            kdim+=1
        x4=iris.cube.Cube(conv_float32(x3),units=oldcube.units,attributes=oldcube.attributes,cell_methods=oldcube.cell_methods,dim_coords_and_dims=dim_coords)
        x4.rename(oldcube.name())
        x4.var_name=oldcube.var_name
        # Mask missing values
        x4.data=np.ma.masked_greater(x4.data,missing_value/10)
        # Create data_oi_interp_lon attribute dictionary entry
        self.data_oi_interp_lon[self.var_name]=x4
        # Save interpolated data
        source_out=self.source_wildcard.replace('???','all')
        #
        ########################################################
        # Need to sort out code with '2016' below to handle year properly
        # in gliderMission and Glider classes
        ########################################################
        #
        fileout=os.path.join(self.basedir,source_out,'std',self.var_name+'_all_2016.nc')
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_oi_interp_lon[self.var_name],fileout)
        
        
#==========================================================================

class Glider(object):

    """Glider object."""

    def __init__(self,gliderid,descriptor,verbose=False):
        self.descriptor=descriptor
        self.gliderid=gliderid
        self.basedir=descriptor['basedir']
        self.mission=descriptor['mission']
        self.source_wildcard=descriptor['source_wildcard']
        self.data_oi={}
        self.data_oi_pad={}
        self.verbose=verbose

    def __repr__(self):
        return 'Glider ({0.gliderid!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'Glider instance \n'+\
                'gliderid: {0.gliderid!s} \n'+\
                'mission: {0.mission!s} \n'+\
                'source_wildcard: {0.source_wildcard!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'Glider instance'

    def f_read_oi(self,var_name,verbose=False):
        """Read OI field for individual glider.

        Create attribute data_oi.
        """
        name=var_name2long_name[var_name]
        self.source=self.source_wildcard.replace('???',str(self.gliderid))
        file1=os.path.join(self.basedir,self.source,'std',var_name+'_all_????.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            x1=iris.load(file1,name)
        x2=x1.concatenate_cube()
        # Set data_oi attribute dictionary entry
        self.data_oi[var_name]=x2
        if verbose:
            print('source {0.source!s}'.format(self))

    def f_oi_pad(self,var_name,time1,time2,verbose=False):
        """Pad OI data with missing values back to time1 and forward to time2.

        Create attribute data_oi_pad.
        """
        name=var_name2long_name[var_name]
        source_info(self)
        # Get first and last time of data: time1a,time2a
        tcoord=self.data_oi[var_name].coord('time')
        time_units=tcoord.units
        time1a=tcoord.units.num2date(tcoord.points[0])
        time2a=tcoord.units.num2date(tcoord.points[-1])
        time1a_val=time_units.date2num(time1a)
        time2a_val=time_units.date2num(time2a)
        if verbose:
            print('time1a,time2a: {0!s}, {1!s}'.format(time1a,time2a))
            print('time1a_val,time2a_val: {0!s}, {1!s}'.format(time1a_val,time2a_val))
        # Process times to pad to: time1,time2
        time1_val=time_units.date2num(time1)
        time2_val=time_units.date2num(time2)
        if verbose:
            print('time1,time2: {0!s}, {1!s}'.format(time1,time2))
            print('time1_val,time2_val: {0!s}, {1!s}'.format(time1_val,time2_val))
        if time1>time1a or time2<time2a:
            raise ValueError('time1 or time2 not set correctly')
        # Find index of time coordinate
        missing_value=1e20
        kount=0
        for dimc in self.data_oi[var_name].dim_coords:
            if dimc.standard_name=='time':
                tcoord_index=kount
                kount+=1
        if verbose:
            print('tcoord_index: {0!s}'.format(tcoord_index))
        #
        # Create padding array with missing values between time1 and time1a
        ntime=int((time1a-time1)/self.timedelta)
        shape=self.data_oi[var_name].shape
        shape1=shape[:tcoord_index]+(ntime,)+shape[tcoord_index+1:]
        if verbose:
            print('ntime: {0!s}'.format(ntime))
            print('shape: {0!s}'.format(shape))
            print('shape1: {0!s}'.format(shape1))
        x1=missing_value*np.ones(shape1)
        tcoord1_vals=[time_units.date2num(time1+ii*self.timedelta) for ii in range(ntime)]
        tcoord1_vals=conv_float32(np.array(tcoord1_vals))
        tcoord1=iris.coords.DimCoord(tcoord1_vals,standard_name='time',var_name='time',units=time_units)        
        if verbose:
            print('tcoord1_vals: {0!s}'.format(tcoord1_vals))
            print('tcoord1: {0!s}'.format(tcoord1))
        pad_before=create_cube(conv_float32(x1),self.data_oi[var_name],new_axis=tcoord1)
        #
        # Create padding array with missing values between time2a and time2'
        ntime=int((time2-time2a)/self.timedelta)
        shape=self.data_oi[var_name].shape
        shape1=shape[:tcoord_index]+(ntime,)+shape[tcoord_index+1:]
        if verbose:
            print('ntime: {0!s}'.format(ntime))
            print('shape: {0!s}'.format(shape))
            print('shape1: {0!s}'.format(shape1))
        x1=missing_value*np.ones(shape1)
        tcoord1_vals=[time_units.date2num(time2a+(ii+1)*self.timedelta) for ii in range(ntime)]
        tcoord1_vals=conv_float32(np.array(tcoord1_vals))
        tcoord1=iris.coords.DimCoord(tcoord1_vals,standard_name='time',var_name='time',units=time_units)        
        if verbose:
            print('tcoord1_vals: {0!s}'.format(tcoord1_vals))
            print('tcoord1: {0!s}'.format(tcoord1))
        pad_after=create_cube(conv_float32(x1),self.data_oi[var_name],new_axis=tcoord1)
        #
        # Create cube list of padding before, data, padding after
        x4=iris.cube.CubeList([pad_before,self.data_oi[var_name],pad_after])
        x5=x4.concatenate_cube()
        # Mask missing values
        x5.data=np.ma.masked_greater(x5.data,missing_value/10)
        # Set data_oi_pad attribute dictionary entry
        self.data_oi_pad[var_name]=x5
        
#==========================================================================

class CubeDiagnostics(object):

    """CubeDiagnostics object.

    An object to calculate physical diagnostics of data, e.g.,
    calculate the mixed layer depth using a particular method, from a
    cube of conservative temperature.

    NB There are no generic diagnostics here, such as calculation of
    the x-derivative (which could be applied to a cube of any
    spatially varying quantity).  Instead, reserve this class to
    contain methods that only work on specific physical quantities,
    such as the above example.

    Other examples might be calculating density from pressure and
    temperature using the ideal gas law.

    An instance of the class can have attributes such as self.tsc
    (conservative_temperature), self.rho (air_density), etc.  These
    will be iris cubes.  Each iris cube should have the same dimensions.
    
    After creating an instance of the class, call method f_read_data()
    to lazy read whatever variables (e.g., 'tsc', or 'rho' and 'ta')
    will be needed later.

    Then loop over relevant time blocks in main programme, and call
    desired method from this class, e.g., f_mld().  These methods,
    e.g., f_mld() then read and set attributes named after the
    variable name, e.g., self.tsc is an iris cube of tsc, and
    calculate diagnostics, e.g., self.mld is an iris cube of mixed
    layer depth.

    It is anticipated that this class will grow over time and will
    eventually be very large, adding more methods as needed.

    """

    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.basedir=descriptor['basedir']
        self.source=descriptor['source']
        source_info(self)
        self.level=descriptor['level']
        # Empty dictionaries to fill later
        self.filein={}
        self.data_in={}
        self.file_data_out=os.path.join(self.basedir,self.source,'std','VAR_NAME_'+str(self.level)+'_'+self.wildcard+'.nc')
        if self.verbose:
            print(self)        

    def __repr__(self):
        return 'CubeDiagnostics (verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'CubeDiagnostics instance \n'+\
                'source: {0.source!s} \n'+\
                'file_data_out: {0.file_data_out!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'CubeDiagnostics instance'

    def f_read_data(self,var_name,level,verbose=False):
        """Lazy read cube(s) of var_name at level for current time block.

        Add entry to the dictionary attributes self.filein and
        self.data_in.
        """
        name=var_name2long_name[var_name]
        self.filein[var_name+'_'+str(level)]=os.path.join(self.basedir,self.source,'std',var_name+'_'+str(level)+'_'+self.wildcard+'.nc')
        with iris.FUTURE.context(netcdf_promote=True):
            self.data_in[var_name+'_'+str(level)]=iris.load(self.filein[var_name+'_'+str(level)],name)
        if verbose:
            ss=h2a+'f_read_data \n'+\
                'var_name: {0!s} \n'+\
                'filein: {1.filein!s} \n'+\
                'data_in: {1.data_in!s} \n'+h2b
            print(ss.format(var_name,self))
            

    def f_mld(self,method=1,zzsfc=1.0,deltatsc=1.0):
        """Calculate mixed layer depth.

        Assumes tsc (conservative temperature) has already been
        loaded, in self.data_in['tsc_all'].

        Choose <method> to calculate mixed layer depth.

        Method 1 (default).  For each profile, takes (conservative)
        temperature at the "surface" depth. (default is <zzsfc> of 1.0
        m).  Finds by linear interpolation the depth z* (zz_star) at which
        the temperature is T* (tsc_star) = <deltatsc> (default of 1.0 degC)
        below the temperature at <zzsfc>.  Saves this as attribute
        self.mltt and saves to file.

        Linear interpolation details.  For each profile, start from
        the surface and go deeper to find z_a (zz_a), the depth at
        which the temperature first falls below T*.  Temperature at
        z_a is T_a (tsc_a).  The depth immediately before z_a is then
        z_b (zz_b_, with temperature T_b (tsc_b).  T* must be in the
        range T_a < T* < T_b, and the desired z* is in the range z_b <
        z* < z_a.  From linear interpolation:

        z* = T*(z_a - z_b) - (T_b*z_a - T_a*z_b)
             ----------------------------------
                       (T_a - T_b)
                       
        
        """
        # Read in tsc for current time block and assign to tsc attribute
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in['tsc_all'].extract(time_constraint)
        self.tsc=x1.concatenate_cube()
        # Make a copy of tsc and work on this copy
        tsc=self.tsc.copy()
        # Check that tsc has a depth axis
        dim_coord_names=[xx.var_name for xx in tsc.dim_coords]
        if 'depth' not in dim_coord_names:
            raise ValueError('depth must be a coordinate.')
        # Transpose tsc cube such that depth is first axis
        ndim=len(dim_coord_names)
        indices=list(range(ndim))
        depth_index=dim_coord_names.index('depth')
        indices.pop(depth_index)
        indices_new=[depth_index,]+indices
        tsc.transpose(new_order=indices_new)
        # If tsc has > 2 dimensions, reshape tsc data to 2-D (nz,ngrid) numpy array
        shape1=tsc.shape
        nz=shape1[0]
        if ndim==1:
            ngrid=None
            tsc_data=tsc.data
        elif ndim==2:
            ngrid=shape1[1]
            tsc_data=tsc.data
        else:
            ngrid=shape1[1]
            for idim in range(2,ndim):
                ngrid*=shape1[idim]
            shape2=(nz,ngrid)
            tsc_data=numpy.reshape(tsc.data,shape2)
        print('ndim, shape1: {0!s}, {1!s}'.format(ndim,shape1))
        print('nz, ngrid: {0!s}, {1!s}'.format(nz,ngrid))
        #
        if method==1:
            # Method 1.
            # Extract 'surface temperature' as temperature at smallest depth.
            lev_coord=self.tsc.coord('depth')
            if lev_coord.points[0]>lev_coord.points[-1]:
                raise ValueError('Depth coordinate must be increasing, not decreasing.')
            lev_con=iris.Constraint(depth=zzsfc)
            tsfc=tsc.extract(lev_con)
            # Create field of T* = (surface temperature minus deltatsc)
            tsc_star=tsfc.data-deltatsc
            # Find the indices of z_a, the first depth at which the temperature
            #   is less than T*
            # z_b is then the depth immediately above this.
            # T_a and T_b are the temperatures at these two depths.  They
            #   bracket T*
            indices_za=np.argmax(tsc_data<tsc_star,axis=0)
            indices_zb=indices_za-1
            tsc_a=tsc_data[indices_za,list(np.arange(ngrid))]
            tsc_b=tsc_data[indices_zb,list(np.arange(ngrid))]
            zz_a=lev_coord.points[indices_za]
            zz_b=lev_coord.points[indices_zb]
            # Linearly interpolate to find the depth z* at which the
            #   temperature is T*
            zz_star=(tsc_star*(zz_a-zz_b)-(tsc_b*zz_a-tsc_a*zz_b)) / (tsc_a-tsc_b)
            if self.verbose:
                ii=15
                print('Sample data from first grid point')
                print('deltatsc: {0!s}'.format(deltatsc))
                print('ii: {0!s}'.format(ii))
                print('tsfc.data[ii]: {0!s}'.format(tsfc.data[ii]))
                print('tsc_star[ii]: {0!s}'.format(tsc_star[ii]))
                print('indices_za[ii]: {0!s}'.format(indices_za[ii]))
                print('indices_zb[ii]: {0!s}'.format(indices_zb[ii]))
                print('tsc_a[ii]: {0!s}'.format(tsc_a[ii]))
                print('tsc_b[ii]: {0!s}'.format(tsc_b[ii]))
                print('zz_a[ii]: {0!s}'.format(zz_a[ii]))
                print('zz_b[ii]: {0!s}'.format(zz_b[ii]))
                print('zz_star[ii]: {0!s}'.format(zz_star[ii]))
        else:
            raise ValueError('Invalid method')
        # Reshape zz_star to shape of original array (minus depth axis)
        if ndim>=2:
            zz_star.reshape(shape1[1:])
        # Create a list of two-lists, each of form [dim_coord,index]
        kdim=0
        dim_coords=[]
        for xx in tsc.dim_coords[1:]:
            dim_coords.append([xx,kdim])
            kdim+=1
        # Create iris cube of zz_star
        var_name='mltt'
        long_name=var_name2long_name[var_name]
        self.mltt=iris.cube.Cube(zz_star,units=lev_coord.units,dim_coords_and_dims=dim_coords)
        self.mltt.rename(long_name)
        self.mltt.var_name=var_name
        # Add cell method to describe calculation of mixed layer
        cm=iris.coords.CellMethod('point','depth',comments='depth where temp is temp (at depth '+str(zzsfc)+') minus '+str(deltatsc))
        self.mltt.add_cell_method(cm)
        # Save cube
        level=0
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.mltt,fileout)

    def f_sea_water_potential_density(self,pref=0):
        """Calculate sea water potential density using Gibbs TEOS-10 routine.

        Assumes tsc (conservative temperature) and sa (absolute
        salinity) hav already been loaded, in
        self.data_in['tsc_LEVEL'] and self.data_in['sa_LEVEL'].

        Uses gsw.rho_CT(tsc,sa,pref) where pref is reference sea water
        pressure in dbar.

        Choose <method> to calculate mixed layer depth.

        Method 1 (default).  Assumes do not have explicit values of
        swp, but tsc and sa are on a grid with a depth axis (in m).
        Depth in m is sufficiently close to sea water pressure in dbar
        to use directly, when only considering shallow depths.

        Method 2.  Assumes swp has already been loaded, in
        self.data_in['swp_LEVEL'], and uses this in the calculation of
        sea water potential density.
        """
        # Read in tsc,sa for current time block and assign to tsc,sa attributes
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in['tsc_'+str(self.level)].extract(time_constraint)
            x2=self.data_in['sa_'+str(self.level)].extract(time_constraint)
        self.tsc=x1.concatenate_cube()
        self.sa=x2.concatenate_cube()
        tsc=self.tsc.data
        sa=self.sa.data
        # Get sea water pressure
        #if method==1:
        #    # Use depth (m) as surrogate for sea water pressure (dbar)
        #    lev_coord=self.tsc.coord('depth')
        #    if lev_coord.units!='m':
        #        raise ValueError('Depth coordinate must be in m.')
        #    swp=lev_coord.points
        #elif method==2:
        #    # Explicitly read sea water pressure data
        #    with iris.FUTURE.context(cell_datetime_objects=True):
        #        x3=self.data_in['swp_'+str(self.level)].extract(time_constraint)
        #    self.swp=x3.concatenate_cube()
        #    if self.swp.units!='dbar':
        #        raise ValueError('Sea water pressure must be in dbar.')
        #    swp=self.swp.data
        # Calculate seawater potential density
        swpd=gsw.rho_CT_exact(sa,tsc,pref)
        # Create iris cube of swp
        var_name='swpd'
        self.swpd=create_cube(conv_float32(swpd),self.tsc,new_var_name=var_name)
        self.swpd.units='kg m-3'
        # Add cell method to describe calculation of sea water potential density
        cm=iris.coords.CellMethod('point','depth',comments='swpd calculated using gsw.rho_CT_exact in f_sea_water_potential_density: pref='+str(pref))
        self.swpd.add_cell_method(cm)
        # Save cube
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.swpd,fileout)

    def f_specific_humidity(self):
        """Calculate specific humidity from relative humidity and temperature.

        Assumes ta (air temperature) and rhum (relative humidity) on
        pressure level self.level have already been loaded, in
        self.data_in['ta_LEVEL'] and self.data_in['rhum_LEVEL'].

        Use Clausius-Clapeyron relation (COARE algorithm) to calculate
        specific humidity.  Follows the saturation specific humidity
        computation in the COARE Fortran code v2.5b.

        Reference is
        https://woodshole.er.usgs.gov/operations/sea-mat/air_sea-html/qsat.html

        """
        # Read in ta,rhum for current time block and assign to ta,rhum attributes
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in['ta_'+str(self.level)].extract(time_constraint)
            x2=self.data_in['rhum_'+str(self.level)].extract(time_constraint)
        self.ta=x1.concatenate_cube()
        self.rhum=x2.concatenate_cube()
        # air temperature must be in degC
        self.ta.convert_units('celsius')
        # relative humidity must be a fraction (not percentage)
        self.rhum.convert_units('1')
        # Calculate specific humidity
        ta=self.ta.data
        rhum=self.rhum.data
        pa=self.level
        # vapour pressure in hPa
        ew=6.1121*(1.0007+3.46e-6*pa)*np.exp((17.502*ta)/(240.97+ta))
        # saturated specific humidity in kg kg-1
        shum_sat=0.62197*(ew/(pa-0.378*ew))
        # specific humidity in kg kg-1
        shum=rhum*shum_sat
        # Create iris cube of shum
        var_name='shum'
        self.shum=create_cube(conv_float32(shum),self.rhum,new_var_name=var_name)
        self.shum.units='1'
        # Add cell method to describe calculation of sea water potential density
        cm=iris.coords.CellMethod('point','pressure',comments='specific humidity calculated from relative humidity and temperature')
        self.shum.add_cell_method(cm)
        # Save cube
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.shum,fileout)
        if self.verbose:
            ta_min=self.ta.data.min()
            ta_max=self.ta.data.max()
            rhum_min=self.rhum.data.min()
            rhum_max=self.rhum.data.max()
            shum_min=self.shum.data.min()
            shum_max=self.shum.data.max()
            ss=h1a+'f_specific_humidity\n'+\
                'level: {0.level!s}\n'+\
                'ta min: {1!s}\n'+\
                'ta max: {2!s}\n'+\
                'rhum min: {3!s}\n'+\
                'rhum max: {4!s}\n'+\
                'shum min: {5!s}\n'+\
                'shum max: {6!s}\n'+h1b
            print(ss.format(self,ta_min,ta_max,rhum_min,rhum_max,shum_min,shum_max))

    def f_vrtbudget(self,level_below,level,level_above):
        """Calculate and save terms in vorticity budget at pressure level.

        Assumes input data has already been loaded for current time
        block, using f_read_data(), in:
        
        self.data_in['uwnd_<level_below>']
        self.data_in['uwnd_<level>']
        self.data_in['uwnd_<level_above>']
        self.data_in['vwnd_<level_below>']
        self.data_in['vwnd_<level>']
        self.data_in['vwnd_<level_above>']
        self.data_in['vrt_<level_below>']
        self.data_in['vrt_<level>']
        self.data_in['vrt_<level_above>']
        self.data_in['omega_<level>']
        self.data_in['div_<level>']

        Calculate and create attributes:

        self.south2north: True if latitude coordinate of input data
        (taken as self.data_in['uwnd_<level>']) runs from south to
        north, False otherwise.  The windspharm methods (horizontal
        gradients and calculation of ff and beta) need cubes to run
        from north to south, and convert them if they do not.  Cube
        output from windspharm running from north to south can then
        not be combined (added) with cubes running from south to
        north.  So convert cube output from windspharm to run from
        south to north if self.south2north is True.

        self.dvrtdt (d zeta/dt) by centred differences

        self.m_uwnd_dvrtdx (-u d zeta/dx)
        self.m_vwnd_dvrtdy (-v d zeta/dy)
        self.m_omega_dvrtdp (-omega d zeta/dp)

        self.m_vrt_div (-zeta D)
        self.m_ff_div (-fD)

        self.m_beta_vwnd (-beta v)

        self.m_domegadx_dvwnddp (-d omega/dx * dv/dp)
        self.domegady_duwnddp (d omega/dy * du/dp)

        self.source_dvrtdt (-u d zeta/dx -v d zeta/dy -omega d zeta/dp
                            -zeta D -fD -beta v
                           - d omega/dx * dv/dp +d omega/dy * du/dp)

        self.res_dvrtdt : d zeta/dt = source + residual
                          residual = d zeta/dt - source
                          residual is the missing source needed to create
                            the actual vorticity tendency

        """
        # Read data for current time block
        self.time1,self.time2=block_times(self,verbose=self.verbose)
        time_constraint=iris.Constraint(time=lambda cell: self.time1 <=cell<= self.time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in['uwnd_'+str(level_below)].extract(time_constraint)
            x2=self.data_in['uwnd_'+str(level)].extract(time_constraint)
            x3=self.data_in['uwnd_'+str(level_above)].extract(time_constraint)
            x4=self.data_in['vwnd_'+str(level_below)].extract(time_constraint)
            x5=self.data_in['vwnd_'+str(level)].extract(time_constraint)
            x6=self.data_in['vwnd_'+str(level_above)].extract(time_constraint)
            x7=self.data_in['vrt_'+str(level_below)].extract(time_constraint)
            x8=self.data_in['vrt_'+str(level)].extract(time_constraint)
            x9=self.data_in['vrt_'+str(level_above)].extract(time_constraint)
            x10=self.data_in['omega_'+str(level)].extract(time_constraint)
            x11=self.data_in['div_'+str(level)].extract(time_constraint)
        self.uwnd_level_below=x1.concatenate_cube()
        self.uwnd_level=x2.concatenate_cube()
        self.uwnd_level_above=x3.concatenate_cube()
        self.vwnd_level_below=x4.concatenate_cube()
        self.vwnd_level=x5.concatenate_cube()
        self.vwnd_level_above=x6.concatenate_cube()
        self.vrt_level_below=x7.concatenate_cube()
        self.vrt_level=x8.concatenate_cube()
        self.vrt_level_above=x9.concatenate_cube()
        self.omega_level=x10.concatenate_cube()
        self.div_level=x11.concatenate_cube()
        #
        # Find value of south2north
        lat_coord=self.uwnd_level.coord('latitude')
        lat_beg=lat_coord.points[0]
        lat_end=lat_coord.points[-1]
        if lat_beg<lat_end:
            self.south2north=True
        else:
            self.south2north=True
        print('lat_beg, lat_end, south2north: {0!s}, {1!s}, {2!s} '.format(lat_beg,lat_end,self.south2north))
        #
        ### Calculate dvrtdt
        # Try to read vrt for timestep before beginning of current time block
        # First create dummy of missing_value to substitute if no data
        time_constraint=iris.Constraint(time=self.time1)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x20=self.data_in['vrt_'+str(level)].extract(time_constraint)
        x21=x20.concatenate_cube()
        shape=x21.data.shape
        print('shape: {0!s}'.format(shape))
        missing_value=1e20
        dummy=missing_value*np.ones(shape)
        lmask1=False
        lmask2=False
        #
        time_before=self.time1-self.timedelta
        print('time_before: {0!s}'.format(time_before))
        time_constraint=iris.Constraint(time=time_before)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x12=self.data_in['vrt_'+str(level)].extract(time_constraint)
        if len(x12)==0:
            self.vrt_level_time_before=dummy
            lmask1=True
            print('Data at time_before not available, creating dummy data')
        else:
            x12a=x12.concatenate_cube()
            self.vrt_level_time_before=x12a.data
        # Try to read vrt for timestep after end of current time block
        timedelta_second=datetime.timedelta(seconds=1)
        time_after=self.time2+timedelta_second+self.timedelta
        print('time_after: {0!s}'.format(time_after))
        time_constraint=iris.Constraint(time=time_after)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x13=self.data_in['vrt_'+str(level)].extract(time_constraint)
        if len(x13)==0:
            self.vrt_level_time_after=dummy
            lmask2=True
            print('Data at time_after not available, creating dummy data')
        else:
            x13a=x13.concatenate_cube()
            self.vrt_level_time_after=x13a.data
        # Find index of time dimension
        dim_coord_names=[xx.var_name for xx in self.vrt_level.dim_coords]
        time_index=dim_coord_names.index('time')
        print('time_index: {0!s}'.format(time_index))
        vrt=self.vrt_level.data.copy()
        vrt_time_minus1=np.roll(vrt,1,axis=time_index)
        vrt_time_plus1=np.roll(vrt,-1,axis=time_index)
        # Don't know how to assign to an arbitrary index.
        if time_index==0:
            vrt_time_minus1[0,...]=self.vrt_level_time_before
            vrt_time_plus1[-1,...]=self.vrt_level_time_after
        else:
            raise ToDoError('Code up for time index other than 0')
        # Find dvrtdt from centered differences
        deltatime_seconds=self.timedelta.total_seconds()
        print('deltatime_seconds: {0!s}'.format(deltatime_seconds))
        dvrtdt=(vrt_time_plus1-vrt_time_minus1)/(2*deltatime_seconds)
        # Mask missing values if needed
        if lmask1:
            dvrtdt[0,...]=dummy
            dvrtdt=np.ma.masked_equal(dvrtdt,missing_value)
        if lmask2:
            dvrtdt[-1,...]=dummy
            dvrtdt=np.ma.masked_equal(dvrtdt,missing_value)
        # Create iris cube
        dvrtdt=conv_float32(dvrtdt)
        dvrtdt=create_cube(dvrtdt,self.vrt_level)
        var_name='dvrtdt'
        long_name=var_name2long_name[var_name]
        dvrtdt.rename(long_name) # not a standard_name
        dvrtdt.var_name=var_name
        vrt_tendency_units='s-2'
        dvrtdt.units=vrt_tendency_units
        self.dvrtdt=dvrtdt
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.dvrtdt,fileout)
        #
        ### Calculate m_uwnd_dvrtdx and m_vwnd_dvrtdx
        ww=VectorWind(self.uwnd_level,self.vwnd_level)
        dvrtdx,dvrtdy=ww.gradient(self.vrt_level)
        if self.south2north:
            dvrtdx=lat_direction(dvrtdx,'s2n')
            dvrtdy=lat_direction(dvrtdy,'s2n')
        m_uwnd_dvrtdx=-1*self.uwnd_level*dvrtdx
        m_vwnd_dvrtdy=-1*self.vwnd_level*dvrtdy
        # m_uwnd_dvrtdx attributes
        var_name='m_uwnd_dvrtdx'
        long_name=var_name2long_name[var_name]
        m_uwnd_dvrtdx.rename(long_name) # not a standard_name
        m_uwnd_dvrtdx.var_name=var_name
        m_uwnd_dvrtdx.units=vrt_tendency_units
        self.m_uwnd_dvrtdx=m_uwnd_dvrtdx
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_uwnd_dvrtdx,fileout)
        # m_vwnd_dvrtdy attributes
        var_name='m_vwnd_dvrtdy'
        long_name=var_name2long_name[var_name]
        m_vwnd_dvrtdy.rename(long_name) # not a standard_name
        m_vwnd_dvrtdy.var_name=var_name
        m_vwnd_dvrtdy.units=vrt_tendency_units
        self.m_vwnd_dvrtdy=m_vwnd_dvrtdy
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_vwnd_dvrtdy,fileout)
        #
        ### Calculate m_omega_dvrtdp
        deltap=100*(level_below-level_above) # Pa
        print('deltap: {0!s}'.format(deltap))
        dvrtdp=(self.vrt_level_below-self.vrt_level_above)/deltap
        m_omega_dvrtdp=-1*self.omega_level*dvrtdp
        # Attributes
        var_name='m_omega_dvrtdp'
        long_name=var_name2long_name[var_name]
        m_omega_dvrtdp.rename(long_name) # not a standard_name
        m_omega_dvrtdp.var_name=var_name
        m_omega_dvrtdp.units=vrt_tendency_units
        self.m_omega_dvrtdp=m_omega_dvrtdp
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_omega_dvrtdp,fileout)
        #
        ### Calculate m_vrt_div
        m_vrt_div=-1*self.vrt_level*self.div_level
        # Attributes
        var_name='m_vrt_div'
        long_name=var_name2long_name[var_name]
        m_vrt_div.rename(long_name) # not a standard_name
        m_vrt_div.var_name=var_name
        m_vrt_div.units=vrt_tendency_units
        self.m_vrt_div=m_vrt_div
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_vrt_div,fileout)
        #
        ### Calculate m_ff_div
        ff=ww.planetaryvorticity()
        if self.south2north:
            ff=lat_direction(ff,'s2n')
        m_ff_div=-1*ff*self.div_level
        m_ff_div.data=conv_float32(m_ff_div.data)
        # Attributes
        var_name='m_ff_div'
        long_name=var_name2long_name[var_name]
        m_ff_div.rename(long_name) # not a standard_name
        m_ff_div.var_name=var_name
        m_ff_div.units=vrt_tendency_units
        self.m_ff_div=m_ff_div
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_ff_div,fileout)
        #
        ### Calculate m_beta_vwnd
        dummy,beta=ww.gradient(ff)
        if self.south2north:
            beta=lat_direction(beta,'s2n')
        m_beta_vwnd=-1*beta*self.vwnd_level
        # Attributes
        var_name='m_beta_vwnd'
        long_name=var_name2long_name[var_name]
        m_beta_vwnd.rename(long_name) # not a standard_name
        m_beta_vwnd.var_name=var_name
        m_beta_vwnd.units=vrt_tendency_units
        self.m_beta_vwnd=m_beta_vwnd
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_beta_vwnd,fileout)
        #
        ### Calculate m_domegadx_dvwnddp
        domegadx,domegady=ww.gradient(self.omega_level)
        if self.south2north:
            domegadx=lat_direction(domegadx,'s2n')
            domegady=lat_direction(domegady,'s2n')
        dvwnddp=(self.vwnd_level_below-self.vwnd_level_above)/deltap
        m_domegadx_dvwnddp=-1*domegadx*dvwnddp
        m_domegadx_dvwnddp.data=conv_float32(m_domegadx_dvwnddp.data)
        # Attributes
        var_name='m_domegadx_dvwnddp'
        long_name=var_name2long_name[var_name]
        m_domegadx_dvwnddp.rename(long_name) # not a standard_name
        m_domegadx_dvwnddp.var_name=var_name
        m_domegadx_dvwnddp.units=vrt_tendency_units
        self.m_domegadx_dvwnddp=m_domegadx_dvwnddp
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.m_domegadx_dvwnddp,fileout)
        #
        ### Calculate domegady_duwnddp
        duwnddp=(self.uwnd_level_below-self.uwnd_level_above)/deltap
        domegady_duwnddp=domegady*duwnddp
        domegady_duwnddp.data=conv_float32(domegady_duwnddp.data)
        # Attributes
        var_name='domegady_duwnddp'
        long_name=var_name2long_name[var_name]
        domegady_duwnddp.rename(long_name) # not a standard_name
        domegady_duwnddp.var_name=var_name
        domegady_duwnddp.units=vrt_tendency_units
        self.domegady_duwnddp=domegady_duwnddp
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.domegady_duwnddp,fileout)
        #
        ### Calculate source_dvrtdt (total source)
        source_dvrtdt=self.m_uwnd_dvrtdx+self.m_vwnd_dvrtdy+self.m_omega_dvrtdp+self.m_vrt_div+self.m_ff_div+self.m_beta_vwnd+self.m_domegadx_dvwnddp+self.domegady_duwnddp
        source_dvrtdt.data=conv_float32(source_dvrtdt.data)
        # Attributes
        var_name='source_dvrtdt'
        long_name=var_name2long_name[var_name]
        source_dvrtdt.rename(long_name) # not a standard_name
        source_dvrtdt.var_name=var_name
        source_dvrtdt.units=vrt_tendency_units
        self.source_dvrtdt=source_dvrtdt
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.source_dvrtdt,fileout)
        #
        ### Calculate res_dvrtdt (residual)
        res_dvrtdt=self.dvrtdt-self.source_dvrtdt
        res_dvrtdt.data=conv_float32(res_dvrtdt.data)
        # Attributes
        var_name='res_dvrtdt'
        long_name=var_name2long_name[var_name]
        res_dvrtdt.rename(long_name) # not a standard_name
        res_dvrtdt.var_name=var_name
        res_dvrtdt.units=vrt_tendency_units
        self.res_dvrtdt=res_dvrtdt
        fileout=self.file_data_out.replace('VAR_NAME',var_name)
        fileout=replace_wildcard_with_time(self,fileout)
        print('fileout: {0!s}'.format(fileout))
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.res_dvrtdt,fileout)

#==========================================================================

class Planet(object):

    """Planetary parameters.

    Default is Planet Earth.

    Attributes:

    Specified base attributes

    self.arad : Planet radius (m)
    self.gg : Gravitational acceleration at surface (m s-2)
    self.omega :  Sidereal rotation rate (s-1)
    self.gascon : Specific gas constant of atmosphere (J kg-1 K-1)
    self.cp : Specific heat capacity of atmosphere at constant pressure (J kg-1 K-1)

    Derived attributes

    self.f0 : Coriolis parameter at 45 degrees lat (s-1)
    self.beta0 : beta at equator (m-1 s-1)
    self.circum : planet circumference (m)
    """

    def __init__(self,title='Earth',verbose=True):
        """Initialise."""
        self.title=title
        self.verbose=verbose
        if self.title=='Earth':
            self.arad=6370000
            self.gg=9.81
            self.omega=7.292e-5
            self.gascon=287
            self.cp=1004
        else:
            raise ToDoError('Code up for other planets.')
        self.f0=np.sqrt(2)*self.omega
        self.beta0=2*self.omega/self.arad
        self.circum=2*np.pi*self.arad
        if self.verbose:
            print(self)
            
    def __repr__(self):
        return 'Planet(title={0.title!r}, verbose={0.verbose!r})'.format(self)
    
    def __str__(self):
        if self.verbose:
            ss=h1a+'Planet instance \n'+\
                'title: {0.title!s} \n'+\
                'arad: {0.arad!s} m \n'+\
                'gg: {0.gg!s} m s-2 \n'+\
                'omega: {0.omega!s} s-1 \n'+\
                'gascon: {0.gascon!s} J kg-1 K-1 \n'+\
                'cp: {0.cp!s} J kg-1 K-1 \n'+\
                'f0: {0.f0!s} s-1 \n'+\
                'beta0: {0.beta0!s} m-1 s-1 \n'+\
                'circum: {0.circum!s} m \n'+h1b
            return ss.format(self)
        else:
            return self.__repr__()    

#==========================================================================

class XYGrid(object):

    """Regularly spaced 2-d rectangular grid.
    
    Basic grid is initially an empty (zeroes) 2-d numpy array.  Populate
    the grid with numbers externally.

    Two alternative sets of axes are used: (x,y) and (lon,lat).  All
    calculations on the grid are done using the (x,y) axes.  The
    (lon,lat) axes are only for presentational purposes.

    Attributes:
    
    self.nx=self.nlon, self.ny=self.nlat : integer number of grid
    points in x and y directions.
    
    self.deltax, self.deltay : float spacing for x and y grid points
    
    self.deltalon, self.deltalat : equivalent float spacing for lon
    and lat grid points
    
    self.x1, self.y1 : float values of first points on x and y axes
    
    self.lon1, self.lat1 : equivalent float values of first points on
    lon and lat axes

    self.xaxis, self.yaxis : numpy arrays of x and y values
    
    self.lonaxis, self.lataxis : equivalent numpy arrays of lon and
    lat values
    
    self.xperiodic, self.yperiodic : Boolean flags for whether x and y
    axes are periodic (wrapping around).
    
    """
    
    def __init__(self,descriptor,verbose=True):
        """Initialise."""
        self.nx=self.nlon=descriptor['nx']
        self.ny=self.nlat=descriptor['ny']
        self.type_init=descriptor['type_init']
        self.var_name=descriptor['var_name']
        self.units=descriptor['units']
        if self.type_init=='lonlat':
            self.deltalon=descriptor['deltalon']
            self.deltalat=descriptor['deltalat']
            self.lon1=descriptor['lon1']
            self.lat1=descriptor['lat1']
        elif self.type_init=='xy':
            self.deltax=descriptor['deltax']
            self.deltay=descriptor['deltay']
            self.x1=descriptor['x1']
            self.y1=descriptor['y1']
        else:
            raise ValueError('Invalid initial type.')
        self.xperiodic=descriptor['xperiodic']
        self.yperiodic=descriptor['yperiodic']
        self.verbose=verbose
        # Create initial 2-d grid of zeros in numpy array
        self.grid=np.zeros((self.ny,self.nx))
        self.planet=Planet()
        if self.type_init=='lonlat':
            # Create x-y equivalents
            self.deltax=(self.deltalon/360)*self.planet.circum
            self.deltay=(self.deltalat/360)*self.planet.circum
            self.x1=(self.lon1/360)*self.planet.circum
            self.y1=(self.lat1/360)*self.planet.circum
        elif self.type_init=='xy':
            # Create lon-lat equivalents
            self.deltalon=(self.deltax/self.planet.circum)*360
            self.deltalat=(self.deltay/self.planet.circum)*360
            self.lon1=(self.x1/self.planet.circum)*360
            self.lat1=(self.y1/self.planet.circum)*360
        # Create lon-axis
        self.lon2=self.lon1+(self.nlon-1)*self.deltalon
        epsilon=self.deltalon/100
        self.lonaxis=np.arange(self.lon1,self.lon2+epsilon,self.deltalon)
        # Create lat-axis
        self.lat2=self.lat1+(self.nlat-1)*self.deltalat
        self.lataxis=np.arange(self.lat1,self.lat2+epsilon,self.deltalat)
        # Create x-axis
        self.x2=self.x1+(self.nx-1)*self.deltax
        epsilon=self.deltax/100
        self.xaxis=np.arange(self.x1,self.x2+epsilon,self.deltax)
        # Create y-axis
        self.y2=self.y1+(self.ny-1)*self.deltay
        self.yaxis=np.arange(self.y1,self.y2+epsilon,self.deltay)
        if self.verbose:
            print(self)
            
    def __repr__(self):
        return 'XYGrid({0.type_init!r},verbose={0.verbose!r})'.format(self)
    
    def __str__(self):
        if self.verbose:
            ss=h1a+'XYGrid instance \n'+\
                'type_init: {0.type_init!s} \n'+\
                'nlon,nlat: {0.nlon!s},{0.nlat!s} \n'+\
                'deltalon,deltalat: {0.deltalon!s},{0.deltalat!s} \n'+\
                'lon1,lon2: {0.lon1!s},{0.lon2!s} \n'+\
                'lat1,lat2: {0.lat1!s},{0.lat2!s} \n'+\
                'nx,ny: {0.nx!s},{0.ny!s} \n'+\
                'deltax,deltay: {0.deltax!s},{0.deltay!s} \n'+\
                'x1,x2: {0.x1!s},{0.x2!s} \n'+\
                'y1,y2: {0.y1!s},{0.y2!s} \n'+\
                'xperiodic,yperiodic: {0.xperiodic!s},{0.yperiodic!s} \n'+h1b
            return ss.format(self)
        else:
            return self.__repr__()

    def x_gradient(self):
        """Calculate x-gradient of self.grid.

        Creates attribute self.x_gradient."""

    def y_gradient(self):
        """Calculate y-gradient of self.grid.

        Creates attribute self.y_gradient."""

    def create_xy_cube(self):
        """Create iris cube from x-y grid.

        Creates attribute self.xy_cube."""

    def create_lonlat_cube(self):
        """Create iris cube from lon-lat grid.

        Creates attribute self.lonlat_cube."""
        loncoord=iris.coords.DimCoord(self.lonaxis,standard_name='longitude',units='degree_east',circular=self.xperiodic)
        latcoord=iris.coords.DimCoord(self.lataxis,standard_name='latitude',units='degree_north')
        dim_coords=[[latcoord,0],[loncoord,1]]
        newcube=iris.cube.Cube(self.grid,units=self.units,dim_coords_and_dims=dim_coords)
        long_name=var_name2long_name[self.var_name]
        newcube.rename(long_name)
        newcube.var_name=self.var_name
        self.lonlatcube=newcube
        
