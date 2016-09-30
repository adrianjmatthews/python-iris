import data_analysis as da

BASEDIR='/gpfs/afm/matthews/data/'

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

YEAR_BEG=1982; YEAR_END=1984

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['basedir']=BASEDIR
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE

aa=da.DataConverter(descriptor,verbose=VERBOSE)

for year in range(YEAR_BEG,YEAR_END+1):
    print('### year={0!s}'.format(year))
    aa.year=year
    aa.read_cube()
    aa.format_cube()
    aa.write_cube()
