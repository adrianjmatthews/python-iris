"""Calculate time mean statistics using data_analysis.TimeDomStats."""

import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

BASEDIR='/gpfs/afm/matthews/data/'

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'; TDOMAINID='djf8283_8384'

FILEPRE='' # e.g., '', '_rac',

VERBOSE=2

PLOT=True

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
    qplt.contourf(aa.time_mean)
    plt.show()
