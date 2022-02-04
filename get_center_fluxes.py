from scipy.io import netcdf
import sys
import numpy as np
import numpy.matlib

br = 8

num = '800'
if (len(sys.argv) > 1):
    num = str(sys.argv[1])
basedir = '/marconi_work/FUA35_OXGK/stonge0/rad_scan/rho_CBC/'
center_file = basedir + num + '/center.out.nc'

print(center_file)

center_nc = netcdf.netcdf_file(center_file,'r')


def read_stella_float(infile, var):

  import numpy as np

  try:
    #print('a')
    #arr = np.copy(infile.variables[var][:])
    arr = infile.variables[var][:]
    #print('b')
    flag = True
  except KeyError:
    print('INFO: '+var+' not found in netcdf file')
    arr =np.arange(1,dtype=float)
    flag = FLAG

  return arr, flag

print('0')
nakxc = center_nc.dimensions['kx']

t  = np.copy(center_nc.variables['t'][:])
nt = t.size

zed  = np.copy(center_nc.variables['zed'][:])
nzed = zed.size
delzed = zed[1]-zed[0]

radgrid  = np.copy(center_nc.variables['rad_grid'][:])
rho = radgrid[:,2]

rhoc = 0.5
jacob     = np.copy(center_nc.variables['jacob'][:])
djacobdr  = np.copy(center_nc.variables['djacdrho'][:])

q       = center_nc.variables['q'].getValue()
shat    = center_nc.variables['shat'].getValue()
d2qdr2  = center_nc.variables['d2qdr2'].getValue()
drhodpsi  = center_nc.variables['drhodpsi'].getValue()
d2psidr2  = center_nc.variables['d2psidr2'].getValue()
dqdr = shat*q/rhoc

dVolume = jacob*delzed/(dqdr*drhodpsi)
dVolume = dVolume*(1.0 + (rho-rhoc)*(djacobdr/jacob - d2qdr2/dqdr + d2psidr2*drhodpsi))

dVolume[0,:]      = 0.5*dVolume[0,:]
dVolume[nzed-1,:] = 0.5*dVolume[nzed-1,:]
volume = np.sum(dVolume[:,br:(nakxc-br)])

dVpsi = delzed*(sum(jacob[0:(nzed-1),0]) + (rho-rhoc)*sum(djacobdr[0:(nzed-1),0]))
newJac = dVpsi*(1.0/(dqdr*drhodpsi))*(1.0 + (rho-rhoc)*(d2psidr2*drhodpsi - d2qdr2/dqdr))

#print(volume)
#print(sum(newJac[br:nakxc-br]))

print('1')

# t spec x
pfluxc  = np.copy(center_nc.variables['pflux_x'][:,0,:])
vfluxc  = np.copy(center_nc.variables['vflux_x'][:,0,:])
qfluxc  = np.copy(center_nc.variables['qflux_x'][:,0,:])

print('2')

densc  = np.copy(center_nc.variables['dens_x'][:])
uparc  = np.copy(center_nc.variables['upar_x'][:])
tempc  = np.copy(center_nc.variables['temp_x'][:])

print('3')

pfl = pfluxc*newJac/volume
vfl = vfluxc*newJac/volume
qfl = qfluxc*newJac/volume

pvol = np.sum(pfl,axis=1)
vvol = np.sum(vfl,axis=1)
qvol = np.sum(qfl,axis=1)

cout = open(basedir + num + '/center.phys_flux','w')
cout.write('#')
cout.write('[1] t    ')
cout.write('[2] dens ')
cout.write('[3] upar ')
cout.write('[4] temp ')
cout.write('\n')
for i in range (0, nt):
  cout.write('%e ' % t[i])
  cout.write('%e ' % pvol[i]) 
  cout.write('%e ' % vvol[i])
  cout.write('%e ' % qvol[i]) 
  cout.write('\n')
cout.close()
