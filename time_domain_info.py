"""Get basic information on time domain."""

import data_analysis as da

TDOMAINID='rmm002djf5'

VERBOSE=True

#==========================================================================

# Create instance of TimeDomain object
aa=da.TimeDomain(TDOMAINID,verbose=VERBOSE)

# Get information on time domain
aa.info()
