"""Calculate and subtract annual cycle using data_analysis.AnnualCycle."""

import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

# YEAR1-YEAR2 are complete years over which to calculate annual cycle
#VAR_NAME='uwnd'; LEVEL=200; SOURCE='ncepdoe_plev_d'; YEAR1=1979; YEAR2=2015
#VAR_NAME='wndspd'; LEVEL=1; SOURCE='ncepncar_sfc_d'; YEAR1=1979; YEAR2=2015
VAR_NAME='psi'; LEVEL=850; SOURCE='ncepncar_plev_d'; YEAR1=1979; YEAR2=2015
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrinterp_toa_d'; YEAR1=1979; YEAR2=2015
#VAR_NAME='sst'; LEVEL=1; SOURCE='sstrey_sfc_d'; YEAR1=1982; YEAR2=2015

NHARM=3

VERBOSE=2

PLOT=False

descriptor={}
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['year1']=YEAR1
descriptor['year2']=YEAR2
descriptor['nharm']=NHARM
descriptor['basedir']='/gpfs/afm/matthews/data/'
descriptor['file_data_in']=descriptor['basedir']+descriptor['source']+\
          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_*.nc'
descriptor['file_anncycle_raw']=descriptor['basedir']+descriptor['source']+\
          '/processed/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_ac_raw_'+str(YEAR1)+'_'+str(YEAR2)+'.nc'
descriptor['file_anncycle_smooth']=descriptor['basedir']+descriptor['source']+\
          '/processed/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_ac_smooth_'+str(YEAR1)+'_'+str(YEAR2)+'.nc'
descriptor['file_anncycle_rm']=descriptor['basedir']+descriptor['source']+\
          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_rac_*.nc'

# Create instance of AnnualCycle object
aa=da.AnnualCycle(descriptor,verbose=VERBOSE)

# Either create and save raw annual cycle or read in previously calculated one
aa.f_anncycle_raw()
#aa.f_read_anncycle_raw()

# Either create and save smoothed annual cycle or read in previously calculated one
aa.f_anncycle_smooth()
#aa.f_read_anncycle_smooth()

# Either create and save anomaly data (smooothed annual cycle subtracted),
# or read in previously calculated anomaly data
aa.f_subtract_anncycle()
#aa.f_read_subtract_anncycle()

if PLOT:
    #timecon=iris.Constraint(time=datetime.datetime(1,1,15))
    timecon=iris.Constraint(time=lambda cell: datetime.datetime(1981,1,1)<=cell<=datetime.datetime(1981,6,30))
    latcon=iris.Constraint(latitude=45.0)
    loncon=iris.Constraint(longitude=90.0)
    with iris.FUTURE.context(cell_datetime_objects=True):
        #x1=aa.data_anncycle_raw.extract(timecon)
        #
        #x1=aa.data_anncycle_smooth.extract(timecon)
        #
        #x1=aa.data_anncycle_smooth.extract(latcon & loncon)
        #x2=aa.data_anncycle_raw.extract(latcon & loncon)
        #
        x1=aa.data_anncycle_rm.extract(timecon & latcon & loncon)
        x1=x1.concatenate_cube()
        x2=aa.data_in.extract(timecon & latcon & loncon)
        x2=x2.concatenate_cube()

    #qplt.contourf(x1)
    #plt.gca().coastlines()

    qplt.plot(x1)
    qplt.plot(x2)
    
    plt.show()
