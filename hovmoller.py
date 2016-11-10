"""Create a Hovmoller field using data_analysis.Hovmoller."""

import data_analysis as da
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

BASEDIR='/gpfs/afm/matthews/data/'

VAR_NAME='wndspd'; LEVEL=1; SOURCE='ncepncar_sfc_d'
#VAR_NAME='sst'; LEVEL=1; SOURCE='sstrey_sfc_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrinterp_toa_d'

FILEPRE='_rac' # e.g., '', '_rac', '_rac_f20_200'

BAND_NAME='longitude' # Dimension to average over: 'latitude' or 'longitude'
BAND_VAL1=80
BAND_VAL2=95

CREATE=True
YEAR_BEG=2016; YEAR_END=2016

PLOT=True

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['file_data_in']=BASEDIR+SOURCE+'/std/'+VAR_NAME+'_'+\
          str(LEVEL)+FILEPRE+'_????.nc'
strhov='_hov_'+BAND_NAME[:3]+'_'+str(BAND_VAL1)+'_'+str(BAND_VAL2)
descriptor['file_data_hov']=BASEDIR+SOURCE+'/processed/'+VAR_NAME+'_'+\
          str(LEVEL)+FILEPRE+strhov+'_????.nc'
descriptor['band_name']=BAND_NAME
descriptor['band_val1']=BAND_VAL1
descriptor['band_val2']=BAND_VAL2

# Create instance of Hovmoller object
aa=da.Hovmoller(descriptor,verbose=VERBOSE)

if CREATE:
    for year in range(YEAR_BEG,YEAR_END+1):
        print('### year={0!s}'.format(year))
        aa.year=year
        aa.time1=datetime.datetime(aa.year,1,1,0,0)
        aa.time2=datetime.datetime(aa.year,12,31,23,59)
        aa.f_hovmoller()
else:
    aa.f_read_hovmoller()
    aa.data_hov_current=aa.data_hov[-1]

if PLOT:
    x1=aa.data_hov_current
    qplt.contourf(x1)
    
    plt.show()
