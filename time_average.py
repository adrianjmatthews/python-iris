"""Average data to lower time resolution using data_analysis.TimeAverage."""

import datetime

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

#VAR_NAME='dvrtdt';
LEVEL=200; SOURCE1='ncepdoe_plev_6h'; SOURCE2='ncepdoe_plev_d'
#VAR_NAME='ppt'; LEVEL=1; SOURCE1='trmm3b42v7_sfc_3h'; SOURCE2='trmm3b42v7_sfc_d'

#YEAR=1979
MONTH=-999
#MONTH=9

PLOT=False

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source1']=SOURCE1
descriptor['source2']=SOURCE2

# Create instance of TimeAverage object
aa=da.TimeAverage(descriptor,verbose=VERBOSE)

print('# YEAR={0!s} MONTH={1!s}'.format(YEAR,MONTH))
aa.year=YEAR
aa.month=MONTH
aa.f_time_average()

if PLOT:
    #tcoord=aa.cube_out.coord('time')[0]
    #time1=tcoord.units.num2date(tcoord.cell(0)[0])
    #timecon=iris.Constraint(time=time1)
    #with iris.FUTURE.context(cell_datetime_objects=True):
    #    x1=aa.cube_out.extract(timecon)

    #qplt.contourf(x1)
    #plt.gca().coastlines()

    #latcon=iris.Constraint(latitude=0.125)
    #loncon=iris.Constraint(longitude=80.125)
    latcon=iris.Constraint(latitude=50.0)
    loncon=iris.Constraint(longitude=140.0)
    x1=aa.cube_in.extract(latcon & loncon)
    x2=aa.cube_out.extract(latcon & loncon)
    qplt.plot(x1,label='in')
    qplt.plot(x2,label='out')
    plt.legend()
    
    plt.show()
