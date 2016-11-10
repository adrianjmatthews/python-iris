"""Interpolate data to higher time resolution using data_analysis.Interpolate."""

import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

BASEDIR='/gpfs/afm/matthews/data/'

VAR_NAME='sst'; LEVEL=1; SOURCE1='sstrey_sfc_w'; SOURCE2='sstrey_sfc_d'

YEAR_BEG=1981; YEAR_END=2016

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
descriptor['file_data_in']=descriptor['basedir']+descriptor['source1']+\
          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_????.nc'
descriptor['file_data_out']=descriptor['basedir']+descriptor['source2']+\
          '/std/'+descriptor['var_name']+'_'+str(descriptor['level'])+\
          '_????.nc'


# Create instance of Interpolate object
aa=da.Interpolate(descriptor,verbose=VERBOSE)

for year in range(YEAR_BEG,YEAR_END+1):
    print('### year={0!s}'.format(year))
    aa.year=year
    if aa.frequency=='d':
        aa.time1_out=datetime.datetime(aa.year,1,1,0,0)
        aa.time2_out=datetime.datetime(aa.year,12,31,23,59)
    else:
        raise UserWarning('Need code for interpolating to other than daily data.')
    aa.f_interpolate_time()

if PLOT:
    #timecon=iris.Constraint(time=datetime.datetime(1985,3,17))
    timecon=iris.Constraint(time=lambda cell: datetime.datetime(1985,1,1)<=cell<=datetime.datetime(1985,12,31))
    latcon=iris.Constraint(latitude=13.5)
    loncon=iris.Constraint(longitude=89.5)
    with iris.FUTURE.context(cell_datetime_objects=True):
        #x1=aa.cube_out.extract(timecon)
        #
        x1=aa.cube_out.extract(timecon & latcon & loncon)
        x2=aa.cube_in.extract(timecon & latcon & loncon)

    
    #qplt.contourf(x1)
    #plt.gca().coastlines()
    #plt.colorbar()
    #
    qplt.plot(x1)
    qplt.plot(x2)
    
    plt.show()
