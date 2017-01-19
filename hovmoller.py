"""Create a Hovmoller field using data_analysis.Hovmoller."""

import datetime

import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

#VAR_NAME='vwnd'; LEVEL=1; SOURCE='ncepncar_sfc_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrinterp_toa_d'
#VAR_NAME='sst'; LEVEL=1; SOURCE='sstrey_sfc_d'
VAR_NAME='ppt'; LEVEL=1; SOURCE='trmm3b42v7_sfc_d'

FILEPRE='' # e.g., '', '_rac', '_rac_f20_200', '_rac_rm5_n5'

BAND_NAME='latitude' # Dimension to average over: 'latitude' or 'longitude'
BAND_VAL1=7.875
BAND_VAL2=8.125

YEAR_BEG=2016; YEAR_END=2016
#MONTH1=MONTH2=-999 # Set both MONTH1 and MONTH2 to same (irrelevant) value if outfile_frequency is 'year'
MONTH1=7; MONTH2=7 # Set month ranges if outfile_frequency is less than 'year'

CREATE=True

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['filepre']=FILEPRE
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
        for month in range(MONTH1,MONTH2+1):
            print('### year={0!s} month={1!s}'.format(year,month))
            aa.year=year
            aa.month=month
            aa.f_hovmoller()
else:
    aa.f_read_hovmoller()
    aa.data_hov_current=aa.data_hov[-1]

if PLOT:
    x1=aa.data_hov_current
    qplt.contourf(x1)
    
    plt.show()
