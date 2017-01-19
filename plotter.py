"""Plot labels etc. """

import iris
import matplotlib as mpl


months_Jan={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
months_1_Jan={1:'1 Jan', 2:'1 Feb', 3:'1 Mar', 4:'1 Apr', 5:'1 May', 6:'1 Jun', 7:'1 Jul', 8:'1 Aug', 9:'1 Sep', 10:'1 Oct', 11:'1 Nov', 12:'1 Dec'}
months_J={1:'J', 2:'F', 3:'M', 4:'A', 5:'M', 6:'J', 7:'J', 8:'A', 9:'S', 10:'O', 11:'N', 12:'D'}


def lonlat2string(val,lonlat):
    """Return a string of form e.g., '5$^\{circ}$N'.

    Use to create longitude and latitude axes labels for plotting.

    Inputs:

    val is a float or integer, e.g., 5, -112.5

    lonlat is a string, either 'lon' or 'lat'

    Outputs:

    lonlatstring is a string, e.g., '5$^\{circ}$S'.

    """
    # Check lonlat is valid
    if lonlat not in ['lon','lat']:
        raise UserWarning("lonlat not valid. Must be 'lon' or 'lat'.")
    # Remove decimal point if integer value, eg 5.0 becomes 5
    #   and take absolute value
    if int(val)==val:
        xx=abs(int(val))
    else:
        xx=abs(val)
    # Set sign
    if lonlat=='lon':
        if val<0:
            sign='W'
        elif 0<=val<=180:
            sign='E'
        else:
            xx=360-xx
            sign='W'
    else:
        if val<0:
            sign='S'
        else:
            sign='N'
    # Create string
    lonlatstring=str(xx)+'$^\circ$'+sign
    return lonlatstring

def reverse_colormap(cmap,name='my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse=[]
    k=[]   
    for key in cmap._segmentdata:    
        k.append(key)
        channel=cmap._segmentdata[key]
        data=[]
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
    LinearL=dict(zip(k,reverse))
    my_cmap_r=mpl.colors.LinearSegmentedColormap(name,LinearL) 
    return my_cmap_r

def fixed_aspect_ratio(axc,ratio=1):
    """Set the physical aspect ratio of an axis.

    axc is a matplotlib axis, with x and y limits already set.
    
    ratio is the desired x/y ratio.

    Redundant.  Use panel-plots to set location of axes on figure
    """
    raise DeprecationWarning('Do not use.  Use panel-plots instead.')
    xvals,yvals=axc.get_xlim(),axc.get_ylim()
    xrange=abs(xvals[1]-xvals[0])
    yrange=abs(yvals[1]-yvals[0])
    axc.set_aspect(ratio*(xrange/yrange),adjustable='box')

def conv_time_units(cube1,cube2):
    """Convert time axis of cube1 to have same time units as cube 2.

    This is useful when plotting both cube1 and cube2 on the same
    subplot in matplotlib.
    """
    time_coord1=cube1.coord('time')
    time_units1=time_coord1.units
    #
    time_coord2=cube2.coord('time')
    time_units2=time_coord2.units
    #
    new_time_vals=[time_units2.date2num(time_units1.num2date(xx)) for xx in time_coord1.points]
    new_time_coord=iris.coords.DimCoord(new_time_vals,standard_name='time',units=time_units2)
    #
    coord_names=[dimc.standard_name for dimc in cube1.dim_coords]
    time_index=coord_names.index('time')
    cube1.remove_coord('time')
    cube1.add_dim_coord(new_time_coord,time_index)
