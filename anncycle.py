import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='uwnd'; LEVEL=200; SOURCE='ncepdoe_plev_d'

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
descriptor['file_anncycle_rm']=descriptor['basedir']+descriptor['source']+\
          '/anom_std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_rac_*.nc'

# Create instance of AnnualCycle object
aa=da.AnnualCycle(descriptor,verbose=VERBOSE)

# Either create and save raw annual cycle or read in previously calculated one
#aa.f_anncycle_raw()
aa.f_read_anncycle_raw()

# Either create and save smoothed annual cycle or read in previously calculated one
#aa.f_anncycle_smooth()
#aa.f_read_anncycle_smooth()

# Subtract smooothed annual cycle
#aa.f_subtract_anncycle()
# Or read it in
#aa.f_read_subtract_anncycle()


if PLOT:
    timecon=iris.Constraint(time=datetime.datetime(1,1,15))
    #timecon=iris.Constraint(time=lambda cell: datetime.datetime(1980,10,1)<=cell<=datetime.datetime(1981,6,30))
    latcon=iris.Constraint(latitude=5.0)
    loncon=iris.Constraint(longitude=90.0)
    with iris.FUTURE.context(cell_datetime_objects=True):
        x1=aa.data_anncycle_raw.extract(timecon)
        #
        #x1=aa.data_anncycle_smooth.extract(timecon)
        #
        #x1=aa.data_anncycle_smooth.extract(latcon & loncon)
        #x2=aa.data_anncycle_raw.extract(latcon & loncon)
        #
        #x1=aa.data_anncycle_rm.extract(timecon & latcon & loncon)
        #x2=aa.data_in.extract(timecon & latcon & loncon)
    qplt.contourf(x1)
    plt.gca().coastlines()
    plt.colorbar()
    #qplt.plot(x1)
    #qplt.plot(x2)
    plt.show()
