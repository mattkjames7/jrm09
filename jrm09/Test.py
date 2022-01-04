import numpy as np
from .Model import Model,ModelScalar
import time
import pstats
import cProfile

def Test(R=0.85,MaxDeg=10):
	'''
	This is a simple function to test the model by recreating a plot in 
	Connerney et al 2018 (figure 4, sort of).
	
	Inputs
	======
	R : float
		The radial distance to evaluate the model at.
	MaxDeg : int
		Maximum model degree to calculate.
	
	'''
	try:
		import matplotlib.pyplot as plt
		import matplotlib.colors as colors
		from mpl_toolkits.axes_grid1 import make_axes_locatable
	except:
		raise SystemError('This function requires "matplotlib" to be instaled')

	#get the coordinates to calculate the model at
	lat = np.linspace(-90,90,181)
	lon = np.linspace(0.0,360.0,361)
	latc = 0.5*(lat[1:] + lat[:-1])
	lonc = 0.5*(lon[1:] + lon[:-1])
	long,latg = np.meshgrid(lon,lat)
	longc,latgc = np.meshgrid(lonc,latc)

	longcr = longc*np.pi/180.0
	latgcr = (90.0 - latgc)*np.pi/180.0
	r = np.zeros(longcr.shape) + R
	
	#calculate the model
	Br,Bt,Bp = Model(r,latgcr,longcr,MaxDeg)
	
	#B = np.sqrt(Br**2 + Bt**2 + Bp**2)
	
	#convert to Gauss
	Bg = Br.reshape(longcr.shape)*1e-5
	
	

	#get the scale
	scale = [-60.0,60.0]
	
	#set norm
	norm = colors.Normalize(vmin=scale[0],vmax=scale[1])	
		
	maps = [1,1,0,0]
	fig = plt
	fig.figure()
	ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
	ax.set_aspect(1.0)
	ax.set_xlabel('SIII East Longitude ($^\circ$)')
	ax.set_ylabel('SIII Latitude ($^\circ$)')
		
	sm = ax.pcolormesh(long,latg,Bg,cmap='RdYlBu_r',norm=norm)
	ct = ax.contour(longc,latgc,Bg,colors='grey',levels=np.linspace(-50,50,11))
	ax.clabel(ct, inline=True, fontsize=8,fmt='%2d')

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	cbar = plt.colorbar(sm,cax=cax) 
	cbar.set_label('$B_r$ (Gauss) at $r$ = {:4.2f}'.format(R) + ' R$_{j}$')
	
	return ax


def Timing(R=0.85,MaxDeg=10):

		#get the coordinates to calculate the model at
	lat = np.linspace(-90,90,181)
	lon = np.linspace(0.0,360.0,361)
	latc = 0.5*(lat[1:] + lat[:-1])
	lonc = 0.5*(lon[1:] + lon[:-1])
	long,latg = np.meshgrid(lon,lat)
	longc,latgc = np.meshgrid(lonc,latc)

	longcr = longc*np.pi/180.0
	latgcr = (90.0 - latgc)*np.pi/180.0
	r = np.zeros(longcr.shape) + R
	
	r = r.flatten()
	longcr = longcr.flatten()
	latgcr = latgcr.flatten()

	
	print('Timing 64800 model vectors')
	t0 = time.time()

	#calculate the model
	Br,Bt,Bp = Model(r,latgcr,longcr,MaxDeg)
		
	t1 = time.time()
	print('Completed in {:f}s'.format(t1-t0))


	
	print('Timing 64800 individual vectors (scalar code)')
	t0 = time.time()

	for i in range(0,r.size):
		#calculate the model
		Br,Bt,Bp = ModelScalar(r[i],latgcr[i],longcr[i],MaxDeg)
		
	t1 = time.time()
	print('Completed in {:f}s'.format(t1-t0))	
	
	
	print('Timing 64800 individual vectors')
	t0 = time.time()

	for i in range(0,r.size):
		#calculate the model
		Br,Bt,Bp = Model(r[i],latgcr[i],longcr[i],MaxDeg)
		
	t1 = time.time()
	print('Completed in {:f}s'.format(t1-t0))	
	


def _GetRTP(R=0.85,MaxDeg=10):
	#get the coordinates to calculate the model at
	lat = np.linspace(-90,90,181)
	lon = np.linspace(0.0,360.0,361)
	latc = 0.5*(lat[1:] + lat[:-1])
	lonc = 0.5*(lon[1:] + lon[:-1])
	long,latg = np.meshgrid(lon,lat)
	longc,latgc = np.meshgrid(lonc,latc)

	longcr = longc*np.pi/180.0
	latgcr = (90.0 - latgc)*np.pi/180.0
	r = np.zeros(longcr.shape) + R
	
	r = r.flatten()
	longcr = longcr.flatten()
	latgcr = latgcr.flatten()
	
	return r,longcr,latgcr
	
def _CallVectorized(r,t,p,MaxDeg):

		#calculate the model
		Br,Bt,Bp = Model(r,t,p,MaxDeg)

	
def _CallVectorizedLoop(r,t,p,MaxDeg):

	for i in range(0,r.size):
		#calculate the model
		Br,Bt,Bp = Model(r[i],t[i],p[i],MaxDeg)		
	
	
def _CallScalarLoop(r,t,p,MaxDeg):

	for i in range(0,r.size):
		#calculate the model
		Br,Bt,Bp = ModelScalar(r[i],t[i],p[i],MaxDeg)		
	
def TimeVectorized():

	print('Timing 64800 model vectors (Vectorized Code)')
	#_CallVectorized()
	
	#cProfile.run("_CallVectorized",'restats')
	#p = pstats.Stats('restats')
	#p.strip_dirs().sort_stats(2).print_stats()

	R = 0.85
	MaxDeg = 10
	r,t,p = _GetRTP(R,MaxDeg)
	
	pr = cProfile.Profile()
	pr.enable()
	_CallVectorized(r,t,p,MaxDeg)
	pr.disable()
	stats = pstats.Stats(pr).strip_dirs().sort_stats('cumtime')
	stats.print_stats()
	
	

	
def TimeVectorizedLoop(R=0.85,MaxDeg=10):


	#get the coordinates to calculate the model at

	print('Timing 64800 model vectors (Vectorized Code, Looping through individual vectors)')
	
	R = 0.85
	MaxDeg = 10
	r,t,p = _GetRTP(R,MaxDeg)
	
	pr = cProfile.Profile()
	pr.enable()
	_CallVectorizedLoop(r,t,p,MaxDeg)
	pr.disable()
	stats = pstats.Stats(pr).strip_dirs().sort_stats('cumtime')
	stats.print_stats()
	
	
	
def TimeScalarLoop(R=0.85,MaxDeg=10):


	print('Timing 64800 model vectors (Scalar Code, Looping through individual vectors)')
	
	R = 0.85
	MaxDeg = 10
	r,t,p = _GetRTP(R,MaxDeg)
	
	pr = cProfile.Profile()
	pr.enable()
	_CallScalarLoop(r,t,p,MaxDeg)
	pr.disable()
	stats = pstats.Stats(pr).strip_dirs().sort_stats('cumtime')
	stats.print_stats()
