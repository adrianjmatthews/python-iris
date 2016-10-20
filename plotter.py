"""Plot labels etc. """

months_Jan={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
months_1_Jan={1:'1 Jan', 2:'1 Feb', 3:'1 Mar', 4:'1 Apr', 5:'1 May', 6:'1 Jun', 7:'1 Jul', 8:'1 Aug', 9:'1 Sep', 10:'1 Oct', 11:'1 Nov', 12:'1 Dec'}
months_J={1:'J', 2:'F', 3:'M', 4:'A', 5:'M', 6:'J', 7:'J', 8:'A', 9:'S', 10:'O', 11:'N', 12:'D'}


def lonlat2string(val,lonlat):
    """Return a string of form e.g., '5$^\{circ}$N'.

    Use to create longitude and latitude axes labels for plotting.

    Inputs:

    val is a float or integer, e.g., 5, -112.5

    lonlat is a string, either 'lon' or 'lat'

    Outputs:

    lonlatstring is a string, e.g., '5$^\{circ}$S'.

    """
    # Check lonlat is valid
    if lonlat not in ['lon','lat']:
        raise UserWarning("lonlat not valid. Must be 'lon' or 'lat'.")
    # Remove decimal point if integer value, eg 5.0 becomes 5
    #   and take absolute value
    if int(val)==val:
        xx=abs(int(val))
    else:
        xx=abs(val)
    # Set sign
    if lonlat=='lon':
        if val<0:
            sign='W'
        else:
            sign='E'
    else:
        if val<0:
            sign='S'
        else:
            sign='N'
    # Create string
    lonlatstring=str(xx)+'$^\circ$'+sign
    return lonlatstring

