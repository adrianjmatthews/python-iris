"""It's object oriented all the way down.

May 2016.  Moving from cdat to iris, and procedural to object oriented
programming.

data_analysis will ultimately replace much of ajm_functions.

Programming style:

#-----------------------------------------------

Class methods typically set an attribute(s) of the class instance.
They typically do not return an argument.

#-----------------------------------------------

Printed output and information

The __init__ method of each class has a 'verbose' keyword argument.

verbose=False (or 0) suppresses printed output

verbose=True (or 1) prints limited output (typically a statement that a
particular class attribute has been created)

verbose=2 prints extended output (typically the value of that class
attribute).

#-----------------------------------------------

"""

# All import statements here, before class definitions
from __future__ import division, print_function, with_statement # So can run this in python2
import iris
from iris.time import PartialDateTime
from iris.experimental.equalise_cubes import equalise_attributes
import datetime
import os.path
import mypaths
import pdb
import numpy as np
import matplotlib.pyplot as plt

h1a='<<<=============================================================\n'
h1b='=============================================================>>>\n'
h2a='<<<---------------------------\n'
h2b='--------------------------->>>\n'

# var_name, standard_name pairs
# var_name is used for file names and as the variable name in netcdf files
# standard_name is the string used by iris to extract a cube from a netcdf file
var_name2standard_name={
    'uwnd':'eastward_wind',
    'vwnd':'northward_wind',
    'wwnd':'upward_air_velocity',
    'wndspd':'wind_speed',
    'pv':'ertel_potential_vorticity',
    'div':'divergence_of_wind',
    'vrt':'atmosphere_relative_vorticity',
    'chi':'atmosphere_horizontal_velocity_potential',
    'omega':'lagrangian_tendency_of_air_pressure',
    'psi':'atmosphere_horizontal_streamfunction',
    'ke':'specific_kinetic_energy_of_air',
    'zg':'geopotential_height',
    'mslp':'air_pressure_at_sea_level',
    'psfc':'surface_air_pressure',
    'ta':'air_temperature',
    }

#==========================================================================

def source_info(self):
    """Create attributes based on the source attribute."""
    # Split source attribute string using underscores as separators
    xx=self.source.split('_')
    if len(xx)!=3:
        raise UserWarning("source attribute '{0.source!s}' must have three parts separated by underscores".format(self))
    self.data_source=xx[0]
    self.level_type=xx[1]
    self.frequency=xx[2]
    # Check data_source attribute is valid
    valid_data_sources=['ncepdoe',]
    if self.data_source not in valid_data_sources:
        raise UserWarning('data_source {0.data_source!s} not vaild'.format(self))
    # Set outfile_frequency attribute depending on source information
    if self.data_source in ['ncepdoe',]:
        self.outfile_frequency='year'
    # Printed output
    if self.verbose:
        ss=h2a+'source_info.  Created attributes: \n'+\
            'data source: {0.data_source!s} \n'+\
            'level_type: {0.level_type!s} \n'+\
            'frequency: {0.frequency!s} \n'+\
            'outfile_frequency {0.outfile_frequency!s} \n'+h2b
        print(ss.format(self))
                
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
              'platform','precision','source','statistic','title','var_desc']
    for attribute in att_list:
        if attribute in cube.attributes:
            del cube.attributes[attribute]

    # A cube also has some "attributes" that are not in the attributes dictionary
    # Set these to empty strings
    cube.long_name=''

    # Set cell methods to empty tuple
    cube.cell_methods=()

#==========================================================================

def create_cube(array,oldcube):
    """Create an iris cube from a numpy array and attributes from an old cube.

    Return newcube.
    """
    # Check that array and oldcube have same dimensions
    if array.shape!=oldcube.shape:
        raise UserWarning('Shape of array and oldcube must match.')
    # Create a list of two-lists, each of form [dim_coord,index]
    kdim=0
    dim_coords=[]
    for xx in oldcube.dim_coords:
        dim_coords.append([xx,kdim])
        kdim+=1
    # Create cube
    newcube=iris.cube.Cube(array,standard_name=oldcube.standard_name,var_name=oldcube.var_name,units=oldcube.units,attributes=oldcube.attributes,cell_methods=oldcube.cell_methods,dim_coords_and_dims=dim_coords)
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

    """
    
    def __init__(self,idx,basedir=mypaths.DIR_TDOMAIN_DEFAULT,verbose=True):
        self.idx=idx
        self.basedir=basedir
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

        Create a lines attribute.
        """
        file1=open(self.filename)
        self.lines=file1.readlines()
        file1.close()
        if self.verbose:
            ss='read_ascii:  {0.idx!s}: Created attribute "lines". \n'
            if self.verbose==2:
                ss+='   lines: {0.lines!s} \n'
            print(ss.format(self))

    def write_ascii(self):
        """Write the ascii file for a newly created time domain."""
        if os.path.isfile(self.filename):
            raise UserWarning('Warning: time domain already exists.  Not overwriting.')
        else:
            # Write ascii strings
            file1=open(self.filename,'w')
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
        """Convert ascii representation of timedomain to PartialDateTime representation.
        Create a partial_date_times attribute.

        """
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
        """Convert datetime representation of timedomain to ascii representation.
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


#==========================================================================

class DataConverter(object):
    """Converter to iris-friendly standard format netcdf files.

    Using iris, read in data from different sources.  Convert to
    standard cube format and file format.  Write to new netcdf file.

    Code is flexible to allow for new input data sources.

    Time: Several options for output file(s):
       If individual files are larger ~1GB, file access significantly slowed
       All data output in single file (e.g., from 1979-2016)
       Data output in separate files for each year, e.g., 1979, 1980, etc.
       Data output in separate files for each month, e.g., Jan 1979, etc.
       olr 144x73 grid, daily, 50 years = 770 MB single file, ok
       era-interim 512x256 grid, 6 hourly = 765 MB per year, ok
       trmm 1440x~400 grid, 3 hourly = 550 MB per month, ok

    Attributes:

    self.basedir : string name of base directory for all data

    self.source : string name of data source, e.g., 'ncepdoe_plev_d',
    'ncepdoe_plev_6h', 'ncepdoe_sfc_6h', 'olrinterp_toa_d' etc.  There
    is a different source for each different combination of source
    data set (e.g., NCEP-DOE reanalysis), type of level (e.g.,
    pressure level), and frequency of input data (e.g., daily).  The
    string should be in the format
    '<data_source>_<level_type>_<frequency>'.  In practice, the source
    attribute is used to set the directory in which the netcdf files
    are stored:
       <self.basedir>/<self.source>/raw_input/ for original data in the
          original format downloaded from data source web site.  Netcdf
          files in this directory are the input to DataConverter
       <self.basedir>/<self.source>/std/ for the converted, standardised
          data, i.e., the output of DataConverter
       <self.basedir>/<self.source>/processed/ for any subsequent analysis
          on the data, e.g., time means etc.  DataConverter does not use this
          directory.

    self.data_source : string name of data source, e.g., 'ncepdoe',
    'olrinterp', etc.  Note that a particular data_source implies a
    particular latitude-longitude grid, e.g., 'ncepdoe' is a 73x144
    grid.  This is relevant for file sizes and self.outfile_frequency.

    self.level_type : string name of level type, e.g., 'plev', 'toa',
    'sfc', 'theta', etc.  This is determined by self.source.

    self.frequency : string denoting time frequency of data.  This is
    used in file names.  It is determined by self.source.  One of:
       '3h' 3-hourly
       '6h' 6-hourly
       'd' daily
       'p' pentad (5-day)
       'w' weekly (7-day)
       'm' monthly (calendar month)

    self.outfile_frequency : string denoting the time coverage of the
    output netcdf file(s).  It is determined by self.source.  One of:
       'year' for separate files for each year, e.g., 1979, 1980
       'month' for separate files for each calendar month, e.g., 197901, 197902
    
    self.var_name : string name of variable, e.g., 'uwnd'.

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

    """
    
    def __init__(self,descriptor,verbose=True):
        """Initialise from descriptor dictionary."""
        self.descriptor=descriptor
        self.verbose=verbose
        self.source=descriptor['source']
        source_info(self)
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        self.level=descriptor['level']
        self.basedir=descriptor['basedir']
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
                'level: {0.level!s} \n'+h1b
            return ss.format(self)
        else:
            return self.__repr__()

    def source_info(self):
        """Create attributes based on the source attribute."""
        # Split source attribute string using underscores as separators
        xx=self.source.split('_')
        if len(xx)!=3:
            raise UserWarning("source attribute '{0.source!s}' must have three parts separated by underscores".format(self))
        self.data_source=xx[0]
        self.level_type=xx[1]
        self.frequency=xx[2]
        # Check data_source attribute is valid
        valid_data_sources=['ncepdoe',]
        if self.data_source not in valid_data_sources:
            raise UserWarning('data_source {0.data_source!s} not vaild'.format(self))
        # Set outfile_frequency attribute depending on source information
        if self.data_source in ['ncepdoe',]:
            self.outfile_frequency='year'
        # Printed output
        if self.verbose:
            ss=h2a+'source_info.  Created attributes: \n'+\
                'data source: {0.data_source!s} \n'+\
                'level_type: {0.level_type!s} \n'+\
                'frequency: {0.frequency!s} \n'+\
                'outfile_frequency {0.outfile_frequency!s} \n'+h2b
            print(ss.format(self))
                
    def read_cube(self):
        """Read cube from raw input file.

        Code is by necessity ad hoc as it caters for many different
        data sources with different input formats.
        """
        # Set input file name and time constraint for current year
        if self.outfile_frequency=='year':
            if self.data_source in ['ncepdoe',]:
                self.filein1=os.path.join(self.basedir,self.source,'raw_input',self.var_name+'.'+str(self.year)+'.nc')
                pdt1=PartialDateTime(year=self.year,month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
                pdt2=PartialDateTime(year=self.year,month=12,day=31,hour=23,minute=59,second=59,microsecond=999999)
                time_constraint=iris.Constraint(time = lambda cell: pdt1 <= cell <= pdt2)
        else:
            raise UserWarning("Need to write code for outfile_frequency other than 'year'.")
        # Set level constraint
        if self.data_source in ['ncepdoe',] and self.level_type=='plev':
            level_constraint=iris.Constraint(Level=self.level)
        # Load cube
        self.cube=iris.load_cube(self.filein1,self.name,callback=clean_callback)
        xx=self.cube.coord('time')
        xx.bounds=None # Hack for new netcdf4 ncepdoe which have physically implausible time bounds
        with iris.FUTURE.context(cell_datetime_objects=True):
            self.cube=self.cube.extract(level_constraint & time_constraint)
        if self.verbose==2:
            ss=h2a+'read_cube. \n'+\
                'pdt1: {0!s} \n'+\
                'pdt2: {1!s} \n'+h2b
            print(ss.format(pdt1,pdt2))
        
    def format_cube(self):
        """Change cube to standard format."""
        # Set self.cube.var_name to self.var_name
        if self.cube.var_name!=self.var_name:
            self.cube.varn_name=self.var_name
            print('format_cube: Changed {0.cube.var_name!s} to {0.var_name!s}'.format(self))

    def write_cube(self):
        """Write cube to netcdf file."""
        # Set output file name
        if self.outfile_frequency=='year':
            self.fileout1=os.path.join(self.basedir,self.source,'raw_std/',self.var_name+'_'+str(self.level)+'_'+str(self.year)+'.nc')
        else:
            raise UserWarning("Need to write code for outfile_frequency other than 'year'.")
        # Write cube
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.cube,self.fileout1)
        if self.verbose==2:
            print('write_cube: {0.fileout1!s}'.format(self))

#==========================================================================

class TimeDomStats(object):
    """Time mean and other statistics of data over non-contiguous time domain.

    """

    def __init__(self,descriptor,verbose=False):
        """Initialise from descriptor dictionary."""
        self.ntimemin=5
        self.__dict__.update(descriptor)
        self.descriptor=descriptor
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        self.level=descriptor['level']
        self.source=descriptor['source']
        self.tdomainid=descriptor['tdomainid']
        self.filein1=descriptor['filein1']
        self.fileout1=descriptor['fileout1']
        if 'ntimemin' in descriptor:
            self.ntimemin=descriptor['ntimemin']
        else:
            self.ntimemin=5
        self.verbose=verbose
        self.tdomain=TimeDomain(self.tdomainid,verbose=self.verbose)
        self.tdomain.read_ascii()
        self.tdomain.ascii2partial_date_time()
        self.tdomain.f_nevents()
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
                'filein1: {0.filein1!s} \n'+\
                'fileout1: {0.fileout1!s} \n'+\
                'ntimemin:{0.ntimemin!s} \n'+h1b
            return ss.format(self)
        else:
            return 'Statistics of '+self.source+' '+self.var_name+str(self.level)+' over '+self.tdomainid

    def event_means(self):
        """Calculate time mean and ntime for each event in time domain.

        Create cube_event_means and cube_event_ntimes attributes."""
        # Check that time domain is of type 'event'
        self.tdomain.time_domain_type()
        if self.tdomain.type!='event':
            raise UserWarning("Warning: time domain type is '{0.tdomain.type}'.  It must be 'event'.".format(self))
        # Load list of cubes
        x1=iris.load(self.filein1,self.name)
        # Loop over events in time domain
        cube_event_means=[]
        cube_event_ntimes=[]
        for eventc in self.tdomain.partial_date_times:
            # Create time constraint
            time_beg=eventc[0]
            time_end=eventc[1]
            print('time_beg: {0!s}'.format(time_beg))
            print('time_end: {0!s}'.format(time_end))
            time_constraint=iris.Constraint(time=lambda cell: time_beg <=cell<= time_end)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x2=x1.extract(time_constraint)
            x3=x2.concatenate_cube()
            ntime=x3.coord('time').shape[0]
            cube_event_ntimes.append(ntime)
            x4=x3.collapsed('time',iris.analysis.MEAN)
            cube_event_means.append(x4)
        self.cube_event_means=cube_event_means
        self.cube_event_ntimes=cube_event_ntimes

    def f_time_mean(self):
        """Calculate time mean over time domain and save to netcdf.

        Calculate this by a weighted (cube_event_ntimes) mean of the
        cube_event_means.  Hence, each individual time (e.g., day) in the
        original data has equal weighting.

        Create attribute time_mean"""
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
        time_mean=x1/float(ntime_total)
        time_mean.standard_name=self.name
        self.time_mean=time_mean
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.time_mean,self.fileout1)
            
#==========================================================================

class TimeFilter(object):
    """Time filter.

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
        self.filter=descriptor['filter']
        self.file_weights=descriptor['file_weights']
        self.f_weights()
        self.timeout1=descriptor['times'][0]
        self.timeout2=descriptor['times'][1]
        self.filein1=descriptor['filein1']
        self.fileout1=descriptor['fileout1']
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        self.source=descriptor['source']
        # Calculate start and end time of input data
        # Time interval of data is encoded in source e.g., ncep_plev_d
        xx=self.source.split('_')
        self.frequency=xx[2]
        if self.frequency=='d':
            timedelta=datetime.timedelta(days=self.nn)
        elif self.frequency=='h':
            timedelta=datetime.timedelta(hours=self.nn)
        else:
            raise UserWarning('data time interval is not days or hours - need more code!')
        self.timein1=self.timeout1-timedelta
        self.timein2=self.timeout2+timedelta
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
                'fileout1: {0.fileout1!s} \n'+\
                'frequency: {0.frequency!s} \n'+\
                'timein1: {0.timein1!s} \n'+\
                'timeout1: {0.timeout1!s} \n'+\
                'timeout2: {0.timeout2!s} \n'+\
                'timein2: {0.timein2!s} \n'+h1b
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
            raise UserWarning('Error: self.nweights must be odd.')
        self.nn=divmod(self.nweights,2)[0]
        weights=[float(xx) for xx in lines2]
        self.weights=np.array(weights)


    def time_filter(self):
        """Filter using the rolling_window cube method and save data."""

        # Read in input data
        x1=iris.load(self.filein1,self.name)
        time_constraint=iris.Constraint(time=lambda cell: self.timein1 <=cell<= self.timein2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x2=x1.extract(time_constraint)
        self.data_in=x2.concatenate_cube()
        # Apply filter
        self.data_out=self.data_in.rolling_window('time',
              iris.analysis.MEAN,self.nweights,weights=self.weights)
        # Save data
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_out,self.fileout1)

#==========================================================================

class AnnualCycle(object):
    """Calculate and subtract annual cycle.

    Assumes input data has equally spaced time intervals.

    Attributes:

    self.frequency : string to denote frequency of input (and output)
    data, e.g., 'd' for daily data.

    self.year1 and self.year2 : integers.  The annual cycle will be
    calculated using input data from 0000:00 UTC 1 Jan self.year1 to
    2359:59 31 Dec self.year2.

    self.data_in : iris cube list of all input data

    self.data_anncycle_raw : iris cube of 'raw' annual cycle.  The
    data for e.g. 5 Jan is a simple mean of all the input data for 5
    Jan.

    self.data_anncycle_smooth : iris cube of smoothed annual cycle.
    Calculated from mean plus first self.nharm annual harmonics of
    self.data_anncycle_raw.

    self.nharm : integer number of annual harmonics to retain for
    smoothed annual cycle.

    self.year_current : integer current year for which annual cycle is
    being removed.

    self.data_anncycle_rm : iris cube of output data with annual cycle
    removed.  Usually the time dimension will run for one complete
    cycle of self.outfile_frequency, e.g. 1 year, from 1 Jan to 31
    Dec.  However, the first and last years of input data may be
    incomplete, and allowance is made for this.
    
    self.filein1 : path name for file(s) of input data.
    
    self.file_anncycle_raw : path name for file of raw annual cycle.
    
    self.file_anncycle_smooth : path name for file of smoothed annual
    cycle.
    
    self.file_anncycle_rm : (partial) path name for file of data with
    smoothed annual cycle subtracted.  This path name is then split
    and e.g., year numbers spliced in for output to individual years.
    
    """
    def __init__(self,descriptor,verbose=False):
        self.descriptor=descriptor
        self.verbose=verbose
        self.filein1=descriptor['filein1']
        self.file_anncycle_raw=descriptor['file_anncycle_raw']
        self.file_anncycle_smooth=descriptor['file_anncycle_smooth']
        x1=self.file_anncycle_smooth.split('.')
        self.file_anncycle_smooth_leap=x1[0]+'_leap.'+x1[1]
        self.file_anncycle_rm=descriptor['file_anncycle_rm']
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        self.source=descriptor['source']
        source_info(self)
        self.year1=descriptor['year1']
        self.year2=descriptor['year2']
        self.nharm=descriptor['nharm']
        self.data_in=iris.load(self.filein1,self.name)
        # Get first time in input data
        x1=self.data_in[0].coord('time')[0]
        x2=x1.cell(0)[0]
        self.time1=x1.units.num2date(x2)
        # Get last time in input data
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
                'nharm: {0.nharm!s} \n'+\
                'filein1: {0.filein1!s} \n'+\
                'data_in: {0.data_in!s} \n'+\
                'time1: {0.time1!s} \n'+\
                'time2: {0.time2!s} \n'+\
                'file_anncycle_raw: {0.file_anncycle_raw!s} \n'+\
                'file_anncycle_smooth: {0.file_anncycle_smooth!s} \n'+\
                'file_anncycle_smooth_leap: {0.file_anncycle_smooth_leap!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'AnnualCycle instance'

    def f_anncycle_raw_old(self):
        """Create raw annual cycle and write to file.

        This will be assigned to year 1, i.e. AD 1.  NB year 1 is not
        a leap year.

        Code is rather slow as it loops over each day of the year, and
        reads in all data (eg 1 Jan's) for that day and averages over
        them.  Superceded by new version of f_anncycle_raw, but have
        left the code here as it has some good iris pointers.

        Create data_anncycle_raw attribute.
        """

        raise UserWarning('Use f_anncycle_raw instead')

        if self.frequency!='d':
            raise UserWarning('Annual cycle presently only coded up for daily data.')
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
            # Add auxilliary time coordinate for current day of year
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

        This will be assigned to year 1, i.e. AD 1.  NB year 1 is not
        a leap year.

        Create data_anncycle_raw attribute.
        """

        if self.frequency!='d':
            raise UserWarning('Annual cycle presently only coded up for daily data.')
        kyear=0
        # First year
        yearc=self.year1
        kyear+=1
        # Extract 1 Jan to 28 Feb of current year
        time1=datetime.datetime(yearc,1,1)
        time2=datetime.datetime(yearc,2,28)
        print(kyear,time1,time2)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        janfeb=x1.concatenate_cube()
        janfeb.remove_coord('time')
        # Extract 1 Mar to 31 Dec of current year
        time1=datetime.datetime(yearc,3,1)
        time2=datetime.datetime(yearc,12,31)
        time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
        with iris.FUTURE.context(cell_datetime_objects=True):
            x1=self.data_in.extract(time_constraint)
        mardec=x1.concatenate_cube()
        mardec.remove_coord('time')
        #
        # Loop over remaining years and add contributions
        #for yearc in range(self.year1+1,self.year2+1):
        for yearc in range(self.year1+1,self.year1+2):
            kyear+=1
            # Extract 1 Jan to 28 Feb of current year
            time1=datetime.datetime(yearc,1,1)
            time2=datetime.datetime(yearc,2,28)
            print(kyear,time1,time2)
            time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            janfebc=x1.concatenate_cube()
            janfebc.remove_coord('time')
            # Extract 1 Mar to 31 Dec of current year
            time1=datetime.datetime(yearc,3,1)
            time2=datetime.datetime(yearc,12,31)
            time_constraint=iris.Constraint(time=lambda cell: time1<=cell<=time2)
            with iris.FUTURE.context(cell_datetime_objects=True):
                x1=self.data_in.extract(time_constraint)
            mardecc=x1.concatenate_cube()
            mardecc.remove_coord('time')
            # Add contributions for current year
            janfeb=iris.analysis.maths.add(janfeb,janfebc)
            mardec=iris.analysis.maths.add(mardec,mardecc)
        # Divide by number of years to calculate mean
        janfeb=iris.analysis.maths.divide(janfeb,kyear)
        mardec=iris.analysis.maths.divide(mardec,kyear)
        # Create and add a time coordinate for year 1, and attributes
        time_first=datetime.datetime(year=1,month=1,day=1)
        time_units='days since '+str(time_first)
        # janfeb
        time_val=np.arange(59)
        tcoord=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
        janfeb.add_dim_coord(tcoord,0)
        janfeb.standard_name=janfebc.standard_name
        janfeb.var_name=janfebc.var_name
        janfeb.attributes=janfebc.attributes
        # mardec
        time_val=np.arange(59,365)
        tcoord=iris.coords.DimCoord(time_val,standard_name='time',units=time_units)
        mardec.add_dim_coord(tcoord,0)
        mardec.standard_name=mardecc.standard_name
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
        """Read raw annual cycle.

        Create data_anncycle_raw attribute.
        """

        self.data_anncycle_raw=iris.load_cube(self.file_anncycle_raw,self.name)

    def f_anncycle_smooth(self):
        """Calculate smoothed annual cycle.

        This is created from the mean and first self.nharm annual
        harmonics of the raw annual cycle.

        f(t_i)= \overline f + \Sum_{k=1}^\{N/2} A_k \cos \omega_k t
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
            raise UserWarning('Code not able to cope with lat,lon,lev data!')
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
        #pdb.set_trace(); print('@@@ Stop here.')
        with iris.FUTURE.context(netcdf_no_unlimited=True):
            iris.save(self.data_anncycle_smooth,self.file_anncycle_smooth)
        #
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
        """Read smoothed annual cycle.

        Create data_anncycle_smooth and data_anncycle_smooth_leap
        attributes.  """

        self.data_anncycle_smooth=iris.load_cube(self.file_anncycle_smooth,self.name)
        self.data_anncycle_smooth_leap=iris.load_cube(self.file_anncycle_smooth_leap,self.name)

    def f_subtract_anncycle(self):
        """Subtract smoothed annual cycle from input data.

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
                print(yearc,leap)
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
                data_anom.standard_name=data_in.standard_name
                data_anom.var_name=data_in.var_name
                data_anom.attributes=data_in.attributes
                # Add a cell method to describe the smoothed annual cycle
                # There is a small list of allowed method names (1st argument)
                # 'point' seems the most appropriate, actions on each point?!
                cm=iris.coords.CellMethod('point','time',comments='smoothed annual cycle: '+str(self.year1)+'-'+str(self.year2)+': mean + '+str(self.nharm)+' harmonics')
                data_anom.add_cell_method(cm)
                # Set data_anncycle_rm attribute
                self.data_anncycle_rm=data_anom
                # Save anomaly data (anncycle subtracted)
                fileout=self.file_anncycle_rm.replace('*',str(yearc))
                print('fileout: {0!s}'.format(fileout))
                with iris.FUTURE.context(netcdf_no_unlimited=True):
                    iris.save(self.data_anncycle_rm,fileout)
                # Increment current year
                yearc+=1
        else:
            raise UserWarning('Need code for other outfile_frequency values.')

    def f_read_subtract_anncycle(self):
        """Read anomaly data (annual cycle subtracted).


        """
        self.data_anncycle_rm=iris.load_cube(self.file_anncycle_rm,self.name)
        
