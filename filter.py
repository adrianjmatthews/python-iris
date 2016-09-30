import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

FILTER='rm61_n30'

FILE_WEIGHTS='/gpfs/home/e058/home/data/weights/w_'+FILTER+'.txt'

VERBOSE=2

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

