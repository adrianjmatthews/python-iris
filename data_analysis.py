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
import iris
from iris.time import PartialDateTime
import datetime
import os.path
import mypaths
import pdb

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
        self.source=descriptor['source']
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        self.level=descriptor['level']
        self.basedir=descriptor['basedir']
        self.verbose=verbose
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
        self.cube=iris.load_cube(self.filein1,self.name,callback=self.clean_callback)
        xx=self.cube.coord('time')
        xx.bounds=None # Hack for new netcdf4 ncepdoe which have physically implausible time bounds
        with iris.FUTURE.context(cell_datetime_objects=True):
            self.cube=self.cube.extract(level_constraint & time_constraint)
        if self.verbose==2:
            ss=h2a+'read_cube. \n'+\
                'pdt1: {0!s} \n'+\
                'pdt2: {1!s} \n'+h2b
            print(ss.format(pdt1,pdt2))
        
    def clean_callback(self,cube,field,filename):
        """Deletes some attributes on iris load.
        
        Problem.  iris concatenate and merge (to create a single cube
        from a cube list) is very picky and will fail if there are any
        mismatching metadata between the cubes.  This function removes
        attributes from the time coordinate and basic metadata that
        typically fall foul of this.  These attributes are not useful anyway.
        
        Usage: as an argument in iris load  (...,callback=clean_callback).

        Do this cleaning here in preprocessing so subsequent analysis
        does not have to contend with this.
        
        """
        # Delete the problem attribute from the time coordinate:
        del cube.coord('time').attributes['actual_range']
        # Or set the attributes dictionary of the time coordinate to empty:
        #cube.coord('time').attributes = {}
        
        # Similarly delete some of the main attributes
        for attribute in ['actual_range','history','unpacked_valid_range']:
            if attribute in cube.attributes:
                del cube.attributes[attribute]

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
        iris.save(self.time_mean,self.fileout1)
            
#==========================================================================

class Filter(object):
    """Time filter.

    Assumes input data has equally spaced time intervals

    Attributes:

    self.weights : 1-d numpy array of filter weights

    self.nn : integer value of order of filter.

    self.nweights : integer number of weights (length of self.weights
    array.  Equal to 2*self.nn+1, and is therefore odd.

    self.deltat_value: float value of fixed, constant time interval of
    input (and output filtered) data, e.g., 1.0, 3.0.

    self.deltat_units: string units of time interval, e.g., 'days',
    'hours'.

    self.data_out : iris cube of output data.  Length of time
    dimension is typically a convenient block of time, e.g., 1 year
    for daily data.

    self.timeout1 : datetime object for start time of self.data_out.
    
    self.timeout2 : datetime object for end time of
    self.data_out.
    
    self.data_in : iris cube of input data to be filtered.  Length of
    time dimension is length of time dimension of self.data_out +
    2*self.nn (self.nn at the beginning, and self.nn at the end).
    
    self.timein1 : datetime object for start time of self.data_in.
    
    self.timein2 : datetime object for end time of
    self.data_in.
    
    """
    def __init__(self,weights,times,descriptor,verbose=False):
        self.weights=weights
        self.timeout1=times[0]
        self.timeout2=times[1]
        self.descriptor=descriptor
        self.filein1=descriptor['filein1']
        self.fileout1=descriptor['fileout1']
        self.var_name=descriptor['var_name']
        self.name=var_name2standard_name[self.var_name]
        # Find nweights, nn
        self.nweights=self.weights.shape[0]
        self.nn=(self.nweights-1)/2
        if self.nn!=(self.nweights-1)/2.:
            raise UserWarning('nweights must be odd.')
        # Find time interval of input data
        self.cubes=iris.load(self.filein1,self.name)
        time_coord=self.cubes[0].coord('time')
        self.deltat_units=str(time_coord.units).split[0]
        self.deltat_value=time_coord.points[1]-time_coord.points[0]
        # Calculate start and end time of input data
        if self.delta_units=='days':
            timedelta=datetime.delta(days=self.deltat_value)
        elif self.delta_units=='hours':
            timedelta=datetime.delta(hours=self.deltat_value)
        else:
            raise UserWarning('delta_units is not days or hours - need more code!')
        self.timein1=self.timeout1-timedelta
        self.timein2=self.timeout2+timedelta
        

    def __repr__(self):
        return 'Filter({0.descriptor!r},verbose={0.verbose!r})'.format(self)

    def __str__(self):
        if self.verbose==2:
            ss=h1a+'Filter instance \n'+\
                'filein1: {0.filein1!s} \n'+\
                'fileout1: {0.fileout1!s} \n'+\
                'timein1: {0.timein1!s} \n'+\
                'timeout1: {0.timeout1!s} \n'+\
                'timeout2: {0.timeout2!s} \n'+\
                'timein2: {0.timein2!s} \n'+\
                'nn: {0.nn!s} \n'+\
                'nweights: {0.nweights!s} \n'+\
                'time interval: {0.deltat_value!s} {0.deltat_units!s} \n'+h1b
            return(ss.format(self))
        else:
            return 'Filter instance'


    def filter(self):
        """Filter using the rolling_window cube method."""

