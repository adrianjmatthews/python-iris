import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'; TDOMAINID='djf8283_8384'

VERBOSE=2

descriptor={}
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['tdomainid']=TDOMAINID
descriptor['basedir']='/gpfs/afm/matthews/data/'
descriptor['filein1']=descriptor['basedir']+descriptor['source']+'/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+'_*.nc'
descriptor['fileout1']=descriptor['basedir']+descriptor['source']+'/processed/'+descriptor['var_name']+'_'+str(descriptor['level'])+'_'+descriptor['tdomainid']+'.nc'

aa=da.TimeDomStats(descriptor,verbose=VERBOSE)

aa.event_means()
aa.f_time_mean()

#print('# Plot')
#qplt.contourf(aa.cube_event_means)
#plt.show()
