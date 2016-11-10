"""Calculate psi, chi, vrt, div, wndspeed from uwnd, vwnd using data_analysis.Wind."""

import data_analysis as da
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime
import iris

BASEDIR='/gpfs/afm/matthews/data/'

LEVEL=850; SOURCE='ncepncar_plev_d'

FILESUF='' # e.g., '', '_rac', '_rac_f20_200'

FLAG_PSI=True
FLAG_CHI=False
FLAG_VRT=False
FLAG_DIV=False
FLAG_WNDSPD=False

YEAR_BEG=1979; YEAR_END=2016

PLOT=True

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['level']=LEVEL
descriptor['source']=SOURCE
descriptor['file_data']=BASEDIR+SOURCE+'/std/VAR_NAME_'+\
          str(LEVEL)+FILESUF+'_*.nc'
descriptor['flag_psi']=FLAG_PSI
descriptor['flag_chi']=FLAG_CHI
descriptor['flag_vrt']=FLAG_VRT
descriptor['flag_div']=FLAG_DIV
descriptor['flag_wndspd']=FLAG_WNDSPD

# Create instance of Wind object
aa=da.Wind(descriptor,verbose=VERBOSE)

for year in range(YEAR_BEG,YEAR_END+1):
    print('### year={0!s}'.format(year))
    aa.year=year
    aa.time1=datetime.datetime(aa.year,1,1,0,0)
    aa.time2=datetime.datetime(aa.year,12,31,23,59)
    aa.f_wind()
    

if PLOT:
    x1=aa.psi
    x2=x1.extract(iris.Constraint(time=x1.coord('time').points[0]))
    qplt.contourf(x2)
    plt.gca().coastlines()
    
    plt.show()
