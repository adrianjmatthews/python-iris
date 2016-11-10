"""Time filter data using data_analysis.TimeFilter."""

import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='ppt'; LEVEL=1; SOURCE='trmm3b42v7_sfc_d'

FILTER='rm11_n5'

FILEPRE='' # e.g., '', '_rac',

YEARS=[2016,]

VERBOSE=2

PLOT=False

#==========================================================================

descriptor={}
descriptor['file_weights']='/gpfs/home/e058/home/data/weights/w_'+FILTER+'.txt'
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['basedir']='/gpfs/afm/matthews/data/'
descriptor['filter']=FILTER
descriptor['filein1']=descriptor['basedir']+descriptor['source']+'/std/'+\
                       descriptor['var_name']+'_'+str(descriptor['level'])+\
                       FILEPRE+'_????.nc'

# Create instance of TimeFilter object
aa=da.TimeFilter(descriptor,verbose=VERBOSE)

for year in YEARS:
    print('#### {0!s}\n'.format(year))
    aa.year=year
    aa.timeout1=datetime.datetime(year,1,1,0,0)
    aa.timeout2=datetime.datetime(year,12,31,23,59)
    aa.time_filter()

if PLOT:
    print('# Plot')
    time_constraint=iris.Constraint(time = lambda cell: aa.timeout1 <= cell <= aa.timeout2)
    tol=0.1
    lon0=0.0
    lon_constraint=iris.Constraint(longitude = lambda cell: lon0-tol <= cell <= lon0+tol)
    lat0=55.0
    lat_constraint=iris.Constraint(latitude = lambda cell: lat0-tol <= cell <= lat0+tol)
    with iris.FUTURE.context(cell_datetime_objects=True):
        x1=aa.data_in.extract(time_constraint & lon_constraint & lat_constraint)
    x1=x1.concatenate_cube()
    x2=aa.data_out.extract(lon_constraint & lat_constraint)
    qplt.plot(x1,label='in')
    qplt.plot(x2,label='out')
    plt.legend()
    plt.axis('tight')
    qplt.show()
    
