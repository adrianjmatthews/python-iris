import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

YEAR1=1979; YEAR2=2015

NHARM=3

VERBOSE=2

PLOT=True

descriptor={}
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['year1']=YEAR1
descriptor['year2']=YEAR2
descriptor['nharm']=NHARM
descriptor['basedir']='/gpfs/afm/matthews/data/'
descriptor['filein1']=descriptor['basedir']+descriptor['source']+'/raw_std/'+\
                       descriptor['var_name']+'_'+str(descriptor['level'])+\
                       '_*.nc'
descriptor['file_anncycle_raw']=descriptor['basedir']+descriptor['source']+\
          '/processed/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_ac_raw_'+str(YEAR1)+'_'+str(YEAR2)+'.nc'
descriptor['file_anncycle_smooth']=descriptor['basedir']+descriptor['source']+\
          '/processed/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_ac_smooth_'+str(YEAR1)+'_'+str(YEAR2)+'.nc'

aa=da.AnnualCycle(descriptor,verbose=VERBOSE)

# Either create and save raw annual cycle or read in previously calculated one
#aa.f_anncycle_raw()
#aa.f_read_anncycle_raw()

# Either create and save smoothed annual cycle or read in previously calculated one
#aa.f_anncycle_smooth()
aa.f_read_anncycle_smooth()

if PLOT:
    timecon=iris.Constraint(time=datetime.datetime(1,1,15))
    latcon=iris.Constraint(latitude=10.0)
    with iris.FUTURE.context(cell_datetime_objects=True):
        x1=aa.data_anncycle_smooth.extract(timecon)
    qplt.contourf(x1, coords=['longitude','latitude'])
    plt.gca().coastlines()
    plt.show()
