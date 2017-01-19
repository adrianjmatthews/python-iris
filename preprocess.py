"""Preprocess data using data_analysis.DataConverter."""

import datetime

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

FILE_MASK=False # Default value

#VAR_NAME='vrt'; LEVEL=850; SOURCE='erainterim_plev_6h'
VAR_NAME='omega'; LEVEL=850; SOURCE='ncepdoe_plev_6h'
#VAR_NAME='vwnd'; LEVEL=200; SOURCE='ncepdoe_plev_d'
#VAR_NAME='vwnd'; LEVEL=1; SOURCE='ncepncar_sfc_d'
#VAR_NAME='omega'; LEVEL=850; SOURCE='ncepncar_plev_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrcdr_toa_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrinterp_toa_d'
#VAR_NAME='sa'; LEVEL='all'; SOURCE='sg532m031oi01_zlev_h'
#VAR_NAME='sst'; LEVEL=1; SOURCE='sstrey_sfc_7d'; FILE_MASK='lsmask.nc'
#VAR_NAME='lhfd'; LEVEL=1; SOURCE='tropflux_sfc_d'
#VAR_NAME='ppt'; LEVEL=1; SOURCE='trmm3b42v7_sfc_3h'

YEAR_BEG=1981; YEAR_END=2016

MONTH1=MONTH2=-999 # Set both MONTH1 and MONTH2 to same value if outfile_frequency is 'year'
#MONTH1=9; MONTH2=9 # Set month ranges if outfile_frequency is less than 'year'

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['file_mask']=FILE_MASK

aa=da.DataConverter(descriptor,verbose=VERBOSE)

for year in range(YEAR_BEG,YEAR_END+1):
    for month in range(MONTH1,MONTH2+1):
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.read_cube()
        aa.format_cube()
        aa.write_cube()

if PLOT:
    time1=aa.cube.coord('time').points[-1]
    time_constraint=iris.Constraint(time=time1)
    x1=aa.cube.extract(time_constraint)

    #x1=aa.cube
    
    qplt.contourf(x1)
    plt.gca().coastlines()
    
    plt.show()
