"""Calculate time mean statistics using data_analysis.TimeDomStats."""

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

#VAR_NAME='vwnd'; LEVEL=850; SOURCE='erainterim_plev_6h'#; TDOMAINID='jan7912'
VAR_NAME='swpd'; LEVEL='all'; SOURCE='sg613m031oi01_zlev_h'; TDOMAINID='boballsg'
#VAR_NAME='ppt'; LEVEL=1; SOURCE='trmm3b42v7_sfc_d'#; TDOMAINID='jan98'
#VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'; TDOMAINID='djf8283_8384'
#VAR_NAME='uwnd'; LEVEL=200; SOURCE='ncepdoe_plev_d'; TDOMAINID='djf8283'

FILEPRE='' # e.g., '', '_rac',

VERBOSE=2

PLOT=False

#==========================================================================

descriptor={}

descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['tdomainid']=TDOMAINID
descriptor['basedir']=BASEDIR
descriptor['filepre']=FILEPRE

# Create instance of TimeDomStats object
aa=da.TimeDomStats(descriptor,verbose=VERBOSE)

# Calculate event means and time mean
aa.event_means()
aa.f_time_mean()

if PLOT:
    print('# Plot')
    #qplt.contourf(aa.time_mean)
    #plt.gca().coastlines()

    qplt.plot(aa.time_mean)
    
    plt.show()
