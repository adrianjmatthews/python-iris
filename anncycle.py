import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

YEAR1=1979; YEAR2=1981

VERBOSE=2

descriptor={}
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['year1']=YEAR1
descriptor['year2']=YEAR2
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
aa.f_read_anncycle_raw()

