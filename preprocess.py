import data_analysis as da

VAR_NAME='vwnd'; LEVEL=1000; SOURCE='ncepdoe_plev_d'

YEAR_BEG=2015; YEAR_END=2015

VERBOSE=2

#------------------------------------------------------------------

descriptor={}
descriptor['var_name']=VAR_NAME
descriptor['level']=LEVEL
descriptor['source']=SOURCE

aa=da.DataConverter(descriptor,verbose=VERBOSE)
aa.source_info()

for year in range(YEAR_BEG,YEAR_END+1):
    print('year=',year)
    aa.year=year
    aa.read_cube()
    aa.format_cube()
    aa.write_cube()
