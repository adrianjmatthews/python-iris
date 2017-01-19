"""Calculate psi, chi, vrt, div, wndspeed from uwnd, vwnd using data_analysis.Wind."""

import datetime

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

LEVEL=850; SOURCE='erainterim_plev_6h'
#LEVEL=925; SOURCE='ncepdoe_plev_6h'
#LEVEL=200; SOURCE='ncepdoe_plev_d'
#LEVEL=700; SOURCE='ncepncar_plev_d'

FILEPRE='' # e.g., '', '_rac', '_rac_f20_200'

FLAG_PSI=True
FLAG_CHI=False
FLAG_VRT=False
FLAG_DIV=False
FLAG_WNDSPD=False

YEAR_BEG=1979; YEAR_END=1979
MONTH1=MONTH2=-999 # Set both MONTH1 and MONTH2 to same (irrelevant) value if outfile_frequency is 'year'
#MONTH1=1; MONTH2=12 # Set month ranges if outfile_frequency is less than 'year'

PLOT=True

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['source']=SOURCE
descriptor['level']=LEVEL
descriptor['filepre']=FILEPRE
#descriptor['file_data']=BASEDIR+SOURCE+'/std/VAR_NAME_'+\
#          str(LEVEL)+FILEPRE+'_????.nc'
descriptor['flag_psi']=FLAG_PSI
descriptor['flag_chi']=FLAG_CHI
descriptor['flag_vrt']=FLAG_VRT
descriptor['flag_div']=FLAG_DIV
descriptor['flag_wndspd']=FLAG_WNDSPD

# Create instance of Wind object
aa=da.Wind(descriptor,verbose=VERBOSE)

# Overwrite irrelevant MONTH1,MONTH2 if outfile_frequency is 'year'
if aa.outfile_frequency=='year':
    MONTH1=MONTH2=-999

for year in range(YEAR_BEG,YEAR_END+1):
    for month in range(MONTH1,MONTH2+1):
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.f_wind()
    
if PLOT:
    x1=aa.psi
    x2=x1.extract(iris.Constraint(time=x1.coord('time').points[0]))
    qplt.contourf(x2)
    plt.gca().coastlines()
    
    plt.show()
