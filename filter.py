import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

FILTER='rm61_n30'

FILE_WEIGHTS='/gpfs/home/e058/home/data/weights/w_'+FILTER+'.txt'

VERBOSE=2

PLOT=True

#=====================================================================

descriptor={}
descriptor['file_weights']=FILE_WEIGHTS
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['basedir']='/gpfs/afm/matthews/data/'
descriptor['filter']=FILTER
descriptor['filein1']=descriptor['basedir']+descriptor['source']+'/raw_std/'+\
                       descriptor['var_name']+'_'+str(descriptor['level'])+\
                       '_*.nc'

for year in [1983,]:
    print('################## {0!s}\n'.format(year))
    time1=datetime.datetime(year=year,month=1,day=1,hour=0,minute=0,second=0)
    time2=datetime.datetime(year=year,month=12,day=31,hour=23,minute=59,second=59)
    descriptor['times']=(time1,time2)
    descriptor['fileout1']=descriptor['basedir']+descriptor['source']+\
                        '/anom_std/'+descriptor['var_name']+\
                        '_'+str(descriptor['level'])+'_'+str(year)+'_'+\
                        descriptor['filter']+'.nc'

    aa=da.TimeFilter(descriptor,verbose=VERBOSE)
    aa.time_filter()

if PLOT:
    print('# Plot')
    time_constraint=iris.Constraint(time = lambda cell: time1 <= cell <= time2)
    tol=0.1
    lon0=0.0
    lon_constraint=iris.Constraint(longitude = lambda cell: lon0-tol <= cell <= lon0+tol)
    lat0=55.0
    lat_constraint=iris.Constraint(latitude = lambda cell: lat0-tol <= cell <= lat0+tol)
    with iris.FUTURE.context(cell_datetime_objects=True):
        x1=aa.data_in.extract(time_constraint & lon_constraint & lat_constraint)
    x2=aa.data_out.extract(lon_constraint & lat_constraint)
    qplt.plot(x1,label='in')
    qplt.plot(x2,label='out')
    plt.legend()
    plt.axis('tight')
    qplt.show()
    
