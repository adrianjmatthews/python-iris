"""Calculate mixed layer depth from conservative temperature."""

import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

LEVEL='all'; SOURCE='sg613m031oi01_zlev_h'

YEAR_BEG=2016; YEAR_END=2016
MONTH1=MONTH2=-999 # Set both MONTH1 and MONTH2 to same (irrelevant) value if outfile_frequency is 'year'
#MONTH1=7; MONTH2=7 # Set month ranges if outfile_frequency is less than 'year'

PLOT=True

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['source']=SOURCE
descriptor['level']=LEVEL

# Create instance of CubeDiagnostics object
aa=da.CubeDiagnostics(descriptor,verbose=VERBOSE)

# Lazy read data: tsc
aa.f_read_data('tsc','all')

for year in range(YEAR_BEG,YEAR_END+1):
    for month in range(MONTH1,MONTH2+1):
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.f_mld(deltatsc=0.2)

if PLOT:
    x1=aa.mltt
    qplt.plot(x1)
    
    plt.show()
