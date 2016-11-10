"""Average data to lower time resolution using data_analysis.TimeAverage."""

import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

BASEDIR='/gpfs/afm/matthews/data/'

VAR_NAME='ppt'; LEVEL=1; SOURCE1='trmm3b42v7_sfc_3'; SOURCE2='trmm3b42v7_sfc_d'

YEAR_BEG=2016; YEAR_END=2016
#MONTH1=1; MONTH2=MONTH1 # Set both MONTH1 and MONTH2 to same value if outfile_frequency is 'year'
MONTH1=2; MONTH2=8 # Set month ranges if outfile_frequency is less than 'year'

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source1']=SOURCE1
descriptor['source2']=SOURCE2
descriptor['basedir']='/gpfs/afm/matthews/data/'
#descriptor['file_data_in']=descriptor['basedir']+descriptor['source1']+\
#          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
#          '_????.nc'
#descriptor['file_data_out']=descriptor['basedir']+descriptor['source2']+\
#          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
#          '_????.nc'

# Create instance of TimeAverage object
aa=da.TimeAverage(descriptor,verbose=VERBOSE)

for year in range(YEAR_BEG,YEAR_END+1):
    for month in range(MONTH1,MONTH2+1):
        print('### year={0!s} month={1!s}'.format(year,month))
        aa.year=year
        aa.month=month
        aa.f_time_average()

if PLOT:
    tcoord=aa.cube_out.coord('time')[0]
    time1=tcoord.units.num2date(tcoord.cell(0)[0])
    timecon=iris.Constraint(time=time1)
    with iris.FUTURE.context(cell_datetime_objects=True):
        x1=aa.cube_out.extract(timecon)

    qplt.contourf(x1)
    plt.gca().coastlines()
    
    plt.show()
