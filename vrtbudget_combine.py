"""Combine terms previously calculated from vorticity budget."""

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

LEVEL=850; SOURCE='ncepdoe_plev_d'
FILEPRE='_rac_h20_n241'

#YEAR=2010
#YEAR=range(1979,1980+1)
MONTH=[-999] # Dummy value if outfile_frequency is 'year'
#MONTH=range(1,12+1) # If outfile_frequency is less than 'year' 

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['verbose']=VERBOSE
descriptor['basedir']=BASEDIR
descriptor['source']=SOURCE
descriptor['level']=LEVEL
descriptor['filepre']=FILEPRE

# Create instance of CubeDiagnostics object
aa=da.CubeDiagnostics(**descriptor)

# Lazy read data: tsc
aa.f_read_data('m_uwnd_dvrtdx',LEVEL)
aa.f_read_data('m_vwnd_dvrtdy',LEVEL)
aa.f_read_data('m_vrt_div',LEVEL)
aa.f_read_data('m_ff_div',LEVEL)
aa.f_read_data('m_domegadx_dvwnddp',LEVEL)
aa.f_read_data('domegady_duwnddp',LEVEL)

iter_year=da.iter_generator(YEAR)
iter_month=da.iter_generator(MONTH)
for year in iter_year:
    for month in iter_month:
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.f_vrtbudget_combine()

if PLOT:
    time1=aa.vrt_stretch.coord('time').points[3]
    time_constraint=iris.Constraint(time=time1)
    x1=aa.vrt_stretch.extract(time_constraint)

    qplt.contourf(x1)
    plt.gca().coastlines()
    
    plt.show()
