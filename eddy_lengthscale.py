from scipy.io import netcdf
from array import array
import numpy as np
import scipy.stats as sp
import numpy.matlib

basedir = '/work/e607/e607/dstonge/rho_CBC4/tdprim2/'
infile = basedir + 'master.out.nc'
infile_nc = netcdf.netcdf_file(infile,'r')

tave = 300

zero_zonal_modes = True

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

def phi_vs_t(infile,var,ny,nx):
# t ntube z kx ky ri
  avt, present = read_stella_float(infile,var)
  arr = ny*nx*(avt[:,0,:,:,:,0] + 1j*avt[:,0,:,:,:,1])
  return arr

print('0')
naky = infile_nc.dimensions['ky']
nakx = infile_nc.dimensions['kx']
ny = 2*naky - 1

ky  = np.copy(infile_nc.variables['ky'][:])
kx  = np.copy(infile_nc.variables['kx'][:])
kperp2  = np.squeeze(np.copy(infile_nc.variables['kperp2'][:]))

Lx = 2*np.pi/kx[1]
Ly = 2*np.pi/ky[1]
dx = Lx/nakx
dy = Ly/ny


t  = np.copy(infile_nc.variables['t'][:])
nt = t.size

tind=nt-1
for i in range (0, nt):
  if(t[i]> tave):
    tind = i
    break
    
print(str(tind) + '  ' + str(nt))

zed  = np.copy(infile_nc.variables['zed'][:])
nzed = zed.size
omp = ((nzed+1)//2) - 1

print('2')

phi_kxky = phi_vs_t(infile_nc,'phi_vs_t',naky,nakx)
phi_kxky = phi_kxky[:,omp,:,:]

if (zero_zonal_modes):
    phi_kxky[:,:,0] = 0.0

phi_kxky2 = np.zeros((nt,nakx,ny),dtype=numpy.complex128)
phi_kxky2[:,:,0:naky] = phi_kxky


for j in range (1,naky):
  phi_kxky2[:,:,ny-j] = np.conjugate(phi_kxky[:,:,j])

#2d 
phiabs = phi_kxky2 * np.conj(phi_kxky2)
phi_auto = np.real(np.fft.ifft(np.fft.ifft(phiabs,axis=1),axis=2))
for i in range (0, nt):
    phi_auto[i,:,:] = phi_auto[i,:,:] / phi_auto[i,0,0]
phi_auto_ave_2d = np.squeeze(np.mean(phi_auto[tind:nt,:,:],axis=0))
phi_auto_std_2d = np.squeeze(np.std( phi_auto[tind:nt,:,:],axis=0))
phi_auto_skw_2d = np.squeeze(sp.skew(phi_auto[tind:nt,:,:],axis=0))
phi_auto_kur_2d = np.squeeze(sp.kurtosis(phi_auto[tind:nt,:,:],axis=0))

#1d
#phi_kxy = np.real(np.fft.ifft(phi_kxky2,axis=2))
#phiabs = phi_kxy * np.conj(phi_kxy)
#phi_auto = np.mean(np.real(np.fft.ifft(phiabs,axis=1)), axis=2)
#phi_auto_ave = np.squeeze(np.mean(phi_auto[tind:nt,:],axis=0))
#phi_auto_ave = phi_auto_ave/phi_auto_ave[0]
phi_auto_ave = phi_auto_ave_2d[:,0]
phi_auto_std = phi_auto_std_2d[:,0]
phi_auto_skw = phi_auto_skw_2d[:,0]
phi_auto_kur = phi_auto_kur_2d[:,0]

phi_auto_ave_2d = np.fft.ifftshift(phi_auto_ave_2d)

print(np.shape(phi_auto))
print(np.shape(phi_auto_ave))

cout = open(basedir + 'eddy_lengthscale_t','w')
for n in range (nt):
    for i in range (nakx):
        cout.write('%e ' % t[n])
        cout.write('%e ' % (dx*i))
        cout.write('%e ' % phi_auto[n,i,0]) 
        cout.write('\n')
    cout.write('\n')
cout.close()

cout = open(basedir + 'eddy_lengthscale','w')
for i in range (nakx):
    cout.write('%e ' % (dx*i))
    cout.write('%e ' % phi_auto_ave[i]) 
    cout.write('%e ' % phi_auto_std[i]) 
    cout.write('%e ' % phi_auto_skw[i]) 
    cout.write('%e ' % phi_auto_kur[i]) 
    cout.write('\n')
cout.close()

cout = open(basedir + 'eddy_lengthscale_2d','w')
for i in range (nakx):
    for j in range (ny):
        cout.write('%e ' % (dx*i - 0.5*(Lx+dx)))
        cout.write('%e ' % (dy*j - 0.5*(Ly+dy)))
        cout.write('%e ' % phi_auto_ave_2d[i,j]) 
        cout.write('\n')
    cout.write('\n')
cout.close()
