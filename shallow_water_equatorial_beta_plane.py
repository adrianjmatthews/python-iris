# -*- coding: utf-8 -*-
"""
Shallow water model on equatorial beta plane

Created on Fri Dec 18 09:03:56 2015

@author: Adrian
"""

print('# Import modules')
import numpy as np
import iris
from iris.time import PartialDateTime
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.gridspec as gridspec
import iris.plot as iplt
import iris.quickplot as qplt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os

def x_gdt_centered_diff(ff,deltax):
    """Calculate x-gradient by centered differences.
    
    Periodic in x"""
    # ff is ff(i)
    ff2=np.roll(ff,-2,axis=1) # ff(i+2)
    #print('ff2',ff2)
    x_deriv=(ff2-ff)/(2.*deltax)
    return x_deriv

def y_gdt_centered_diff(ff,deltay):
    """Calculate y-gradient by centered differences.
    
    Forward difference at lower y boundary.
    Backward difference at upper y boundary"""
    # ff is ff(i)
    ff2=np.roll(ff,-2,axis=0) # ff(j+2)
    #print('ff2',ff2)
    y_deriv=(ff2-ff)/(2.*deltay)
    y_deriv[0,:]=(ff[1,:]-ff[0,:])/deltay
    y_deriv[ny-1,:]=(ff[ny-1,:]-ff[ny-2,:])/deltay
    return y_deriv
    
def space_filter(ff,weights,axis):
    """Calculate spatial smoothed field in x or y direction."""
    nfilt=len(weights)
    if nfilt!=3:
        raise('Need to recode')
    filt_sum=sum(weights)
    afilt=np.array(weights)/float(filt_sum)
    #print('afilt',afilt)
    ff_1=np.roll(ff,-1,axis=axis) # ff(i,j+1)
    ff_m1=np.roll(ff,1,axis=axis) # ff(i,j-1)
    ff_filt=afilt[0]*ff_m1+afilt[1]*ff+afilt[2]*ff_1

    #  Need to add code to take account of y dimension not being periodic    
    
    return ff_filt
    
def create_cube(uu,vv,zz,tt):
    tt_coord=iris.coords.DimCoord(tt,standard_name='time',var_name='time',units=tunits)

    uu_cube=iris.cube.Cube(np.reshape(uu,(1,)+uu.shape),var_name='u',units='m s-1')
    uu_cube.add_dim_coord(tt_coord,0)
    uu_cube.add_dim_coord(lat_coord,1)
    uu_cube.add_dim_coord(lon_coord,2)

    vv_cube=iris.cube.Cube(np.reshape(vv,(1,)+vv.shape),var_name='v',units='m s-1')
    vv_cube.add_dim_coord(tt_coord,0)
    vv_cube.add_dim_coord(lat_coord,1)
    vv_cube.add_dim_coord(lon_coord,2)

    zz_cube=iris.cube.Cube(np.reshape(zz,(1,)+zz.shape),var_name='Z',units='m')
    zz_cube.add_dim_coord(tt_coord,0)
    zz_cube.add_dim_coord(lat_coord,1)
    zz_cube.add_dim_coord(lon_coord,2)
    
    return uu_cube,vv_cube,zz_cube

def concatenate_arrays(ff,ff_all):
    ff_all=np.concatenate(ff_all,np.reshape(ff,(1,)+ff.shape))
    return ff_all

def time_filter(ff_nminus1,ff_n):
    ff_nminus1_filt=(1-alpha)*ff_nminus1+alpha*ff_n
    ff_n_filt=(1-alpha)*ff_n+alpha*ff_nminus1
    return ff_nminus1_filt,ff_n_filt

print('# Output file')
fileout='./shallow_water.nc'
if os.path.exists(fileout):
    print('Removing old version',fileout)
    os.remove(fileout)

print('# Set parameters')
aa=6.37e6 # Earth radius (m)
omega=7.292e-5 # Earth sidereal rotation rate (s-1)
gg=9.81 # Earth gravitational acceleration (m s-2)
beta=2*omega/aa # Meridional gradient of planetary vorticity (m-1 s-1)
circum=2*np.pi*aa# Earth circumference (m)
print('aa',aa)
print('omega',omega)
print('beta',beta)
print('circum',circum)
lx=circum # Length of x domain (m)
print('lx',lx)
nx=180 # Number of x grid points
ny=81 # Number of y grid points
nt=1440*10 # Number of time steps
print('nx,ny,nt',nx,ny,nt)
deltax=lx/float(nx) # x grid spacing (m)
deltalambda=360.0/float(nx) # x grid spacing in degrees longitude (plotting only)
print('deltax,deltalambda',deltax,deltalambda)
phimax=60 # y domain runs from phimax degrees south to phimax degrees north
print('phimax',phimax)
deltaphi=float(2*phimax)/float(ny-1) # y grid spacing in degrees latitude (plotting only)
print('deltaphi',deltaphi)
ly=float(2*phimax)*circum/360.0 # Length of y domain (m)
print('ly',ly)
deltay=ly/float(ny-1) # y grid spacing (m)
print('deltay',deltay)
ce=30. # Phase speed of initial equatorial Kelvin (m s-1)
print('ce',ce)
deltat_cfl_criterion=min(deltax,deltay)/ce # CFL criterion for deltat (s)
#deltat=0.2*deltat_cfl_criterion # time step (s)
deltat=60
print('deltat_cfl_criterion,deltat',deltat_cfl_criterion,deltat)
if deltat*10>deltat_cfl_criterion:
    raise('Decrease deltat')
ss=1 # Zonal wavenumber of initial Kelvin wave (integer)
kk=float(ss)/aa # x-wavenumber of initial Kelvin wave (m-1)
print('ss,kk',ss,kk)
u0=1. # Peak amplitude of initial Kelvin wave zonal wind (m s-1)
print('u0',u0)
tunits='seconds since 2000-01-01 00:00'

# Time filtering to stop CT scheme splitting solutions
nfilt_time=50 #  Apply time filtering every nfilt_time time steps
alpha=0.05 # Time filter parameter.

# Spatial filtering (aka explicit numerical diffusion)
nfilt_space=25 #  Apply space filtering every nfilt_space time steps
weights=[1,2,1] # Spatial filter to apply in x and y directions

# Newtonian damping
rtau1=1./86400. # reciprocal of damping time scale (s-1) for whole domain
rtau2=1./3600. # reciprocal of damping time scale (s-1) for buffer region at y boundaries
ny_buffer=5 # Number of y grid points in buffer at south and north boundaries
rtau=rtau1*np.ones((ny,nx))
for jj in range(ny_buffer):
    rtau3=(float(ny_buffer-jj)/float(ny_buffer))*rtau2
    #print jj,rtau3
    rtau[jj,:]=max(rtau3,rtau1)
    rtau[ny-1-jj,:]=max(rtau3,rtau1)
rtau=np.zeros((ny,nx)) # Uncomment to set Newtonian damping to zero

nstep_save=60 # Save output every nstep_save time steps.  Set to 1 normally.
print('nstep_save',nstep_save)

print('# Create initial arrays')
xx=np.linspace(0.,lx,num=nx,endpoint=False) # x grid points (m)
#print('xx',xx)
xx_coord=iris.coords.DimCoord(xx,var_name='x',units='m',circular=True)
yy=np.linspace(-ly/2,ly/2,num=ny,endpoint=True) # y grid points (m)
#print('yy',yy)
yy_coord=iris.coords.DimCoord(yy,var_name='y',units='m')

lon=np.linspace(0.,360.,num=nx,endpoint=False) # longitude axis (degrees)
#print('lon',lon)
lon_coord=iris.coords.DimCoord(lon,var_name='longitude',units='degrees',circular=True)
lat=np.linspace(-phimax,phimax,num=ny,endpoint=True) # latitude axis (degrees)
#print('lat',lat)
lat_coord=iris.coords.DimCoord(lat,var_name='latitude',units='degrees')

ff=beta*yy # Coriolis parameter (s-1)
#print('ff',ff)
ff=np.reshape(ff,ff.shape+(1,))
ff=np.dot(ff,np.ones((1,nx)))
gauss=np.exp(-beta*yy**2/(2*ce)) # Gaussian y shape of initial Kelvin wave
#print('gauss',gauss)
coswave=np.cos(kk*xx)
#print('coswave',coswave)
gauss=np.reshape(gauss,gauss.shape+(1,))
coswave=np.reshape(coswave,(1,)+coswave.shape)
uu_n=u0*np.dot(gauss,coswave)
vv_n=np.zeros(uu_n.shape)
zz_n=ce*uu_n/gg
nn=0
# End of n=0 initialisation
tt=[nn*deltat]
tt_all=tt
uu_all=np.reshape(uu_n,(1,)+uu_n.shape)
vv_all=np.reshape(vv_n,(1,)+vv_n.shape)
zz_all=np.reshape(zz_n,(1,)+zz_n.shape)

print('# First time step: forward difference')
zz_x_deriv=x_gdt_centered_diff(zz_n,deltax)
zz_y_deriv=y_gdt_centered_diff(zz_n,deltay)
uu_x_deriv=x_gdt_centered_diff(uu_n,deltax)
vv_y_deriv=y_gdt_centered_diff(vv_n,deltay)
uu_nplus1=uu_n+deltat*(ff*vv_n-gg*zz_x_deriv-rtau*uu_n)
vv_nplus1=vv_n+deltat*(-ff*uu_n-gg*zz_y_deriv-rtau*vv_n)
zz_nplus1=zz_n+deltat*(-(ce**2/gg)*(uu_x_deriv+vv_y_deriv)-rtau*zz_n)
# Lateral boundary condition v=0
vv_nplus1[0,:]=0
vv_nplus1[ny-1,:]=0
uu_nminus1=uu_n
vv_nminus1=vv_n
zz_nminus1=zz_n
uu_n=uu_nplus1
vv_n=vv_nplus1
zz_n=zz_nplus1
nn=1
# End of n=1 first time step
tt=[nn*deltat]
tt_all+=tt
uu_all=np.concatenate((uu_all,np.reshape(uu_n,(1,)+uu_n.shape)))
vv_all=np.concatenate((vv_all,np.reshape(vv_n,(1,)+vv_n.shape)))
zz_all=np.concatenate((zz_all,np.reshape(zz_n,(1,)+zz_n.shape)))

print('# Loop over remaining time steps')
for nn in range(2,nt):
    print('nn',nn)
    # Calculate horizontal derivatives
    zz_x_deriv=x_gdt_centered_diff(zz_n,deltax)
    zz_y_deriv=y_gdt_centered_diff(zz_n,deltay)
    uu_x_deriv=x_gdt_centered_diff(uu_n,deltax)
    vv_y_deriv=y_gdt_centered_diff(vv_n,deltay)
    # Calculate CTCS time step
    uu_nplus1=uu_nminus1+2*deltat*(ff*vv_n-gg*zz_x_deriv-rtau*uu_n)
    vv_nplus1=vv_nminus1+2*deltat*(-ff*uu_n-gg*zz_y_deriv-rtau*vv_n)
    zz_nplus1=zz_nminus1+2*deltat*(-(ce**2/gg)*(uu_x_deriv+vv_y_deriv)-rtau*zz_n)
    # Calculate FTCS time step
    #uu_nplus1=uu_n+deltat*(ff*vv_n-gg*zz_x_deriv-rtau*uu_n)
    #vv_nplus1=vv_n+deltat*(-ff*uu_n-gg*zz_y_deriv-rtau*vv_n)
    #zz_nplus1=zz_n+deltat*(-(ce**2/gg)*(uu_x_deriv+vv_y_deriv)-rtau*zz_n)
    # Lateral boundary condition v=0
    vv_nplus1[0,:]=0
    vv_nplus1[ny-1,:]=0
    # Update arrays
    uu_nminus1=uu_n
    vv_nminus1=vv_n
    zz_nminus1=zz_n
    uu_n=uu_nplus1
    vv_n=vv_nplus1
    zz_n=zz_nplus1
    # Time filter
    if divmod(nn,nfilt_time)[1]==0:
        print('Applying time filter')
        uu_nminus1,uu_n=time_filter(uu_nminus1,uu_n)
        vv_nminus1,vv_n=time_filter(vv_nminus1,vv_n)
        zz_nminus1,zz_n=time_filter(zz_nminus1,zz_n)
    if divmod(nn,nfilt_space)[1]==0:
        print('Applying space filter')
        uu_n=space_filter(uu_n,weights,axis=1)
        vv_n=space_filter(vv_n,weights,axis=1)
        zz_n=space_filter(zz_n,weights,axis=1)
        uu_nminus1=space_filter(uu_nminus1,weights,axis=1)
        vv_nminus1=space_filter(vv_nminus1,weights,axis=1)
        zz_nminus1=space_filter(zz_nminus1,weights,axis=1)
        #
        uu_n=space_filter(uu_n,weights,axis=0)
        vv_n=space_filter(vv_n,weights,axis=0)
        zz_n=space_filter(zz_n,weights,axis=0)
        uu_nminus1=space_filter(uu_nminus1,weights,axis=0)
        vv_nminus1=space_filter(vv_nminus1,weights,axis=0)
        zz_nminus1=space_filter(zz_nminus1,weights,axis=0)

    # End of time step
    tt=[nn*deltat]
    if divmod(nn,nstep_save)[1]==0:
        print('Saving output')
        tt_all+=tt
        uu_all=np.concatenate((uu_all,np.reshape(uu_n,(1,)+uu_n.shape)))
        vv_all=np.concatenate((vv_all,np.reshape(vv_n,(1,)+vv_n.shape)))
        zz_all=np.concatenate((zz_all,np.reshape(zz_n,(1,)+zz_n.shape)))

print('# Create cubes') 
tt_coord=iris.coords.DimCoord(tt_all,standard_name='time',var_name='time',units=tunits)

uu_cube=iris.cube.Cube(uu_all,var_name='u',units='m s-1')
uu_cube.add_dim_coord(tt_coord,0)
uu_cube.add_dim_coord(lat_coord,1)
uu_cube.add_dim_coord(lon_coord,2)

vv_cube=iris.cube.Cube(vv_all,var_name='v',units='m s-1')
vv_cube.add_dim_coord(tt_coord,0)
vv_cube.add_dim_coord(lat_coord,1)
vv_cube.add_dim_coord(lon_coord,2)

zz_cube=iris.cube.Cube(zz_all,var_name='Z',units='m')
zz_cube.add_dim_coord(tt_coord,0)
zz_cube.add_dim_coord(lat_coord,1)
zz_cube.add_dim_coord(lon_coord,2)

print('# Plot')
qplt.contourf(iris.util.squeeze(uu_cube[-1]))

