"""Calculate vorticity budget terms."""

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

#SOURCE='erainterim_plev_6h'
#LEVEL_BELOW=875; LEVEL=850; LEVEL_ABOVE=825
SOURCE='ncepdoe_plev_6h'
#LEVEL_BELOW=925; LEVEL=850; LEVEL_ABOVE=700
LEVEL_BELOW=250; LEVEL=200; LEVEL_ABOVE=150

YEAR_BEG=2011; YEAR_END=2016
MONTH1=MONTH2=-999 # Set both MONTH1 and MONTH2 to same (irrelevant) value if outfile_frequency is 'year'
#MONTH1=7; MONTH2=7 # Set month ranges if outfile_frequency is less than 'year'

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['source']=SOURCE
descriptor['level']=LEVEL

# Create instance of CubeDiagnostics object
aa=da.CubeDiagnostics(descriptor,verbose=VERBOSE)

# Lazy read data
aa.f_read_data('uwnd',LEVEL_BELOW)
aa.f_read_data('uwnd',LEVEL)
aa.f_read_data('uwnd',LEVEL_ABOVE)
aa.f_read_data('vwnd',LEVEL_BELOW)
aa.f_read_data('vwnd',LEVEL)
aa.f_read_data('vwnd',LEVEL_ABOVE)
aa.f_read_data('vrt',LEVEL_BELOW)
aa.f_read_data('vrt',LEVEL)
aa.f_read_data('vrt',LEVEL_ABOVE)
aa.f_read_data('omega',LEVEL)
aa.f_read_data('div',LEVEL)

for year in range(YEAR_BEG,YEAR_END+1):
    for month in range(MONTH1,MONTH2+1):
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.f_vrtbudget(LEVEL_BELOW,LEVEL,LEVEL_ABOVE)

if PLOT:
    time1=aa.dvrtdt.coord('time').points[3]
    time_constraint=iris.Constraint(time=time1)
    x1=aa.dvrtdt.extract(time_constraint)

    qplt.contourf(x1)
    plt.gca().coastlines()
    
    plt.show()
