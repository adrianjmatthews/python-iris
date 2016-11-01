import data_analysis as da
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

BASEDIR='/gpfs/afm/matthews/data/'

FILE_MASK=False # Default value

#VAR_NAME='uwnd'; LEVEL=200; SOURCE='ncepdoe_plev_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrcdr_toa_d'
#VAR_NAME='olr'; LEVEL=0; SOURCE='olrinterp_toa_d'
#VAR_NAME='sst'; LEVEL=1; SOURCE='sstrey_sfc_w'; FILE_MASK='lsmask.nc'
VAR_NAME='lhfd'; LEVEL=1; SOURCE='tropflux_sfc_d'

YEAR_BEG=2016; YEAR_END=2016

PLOT=True

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
    print('### year={0!s}'.format(year))
    aa.year=year
    aa.read_cube()
    aa.format_cube()
    aa.write_cube()

if PLOT:
    time1=aa.cube.coord('time')[0].points[-1]
    time_constraint=iris.Constraint(time=time1)
    x1=aa.cube.extract(time_constraint)
    
    qplt.contourf(x1)
    plt.gca().coastlines()
    #plt.colorbar()
    
    plt.show()
