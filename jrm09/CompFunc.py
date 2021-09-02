import numpy as np
import pandas as pd
from . import Globals
from math import factorial

def CompFunc(r,theta,phi,MaxDeg=10):
	'''
	My wrapper function for the other code
	
	'''
	_r = np.array([r]).flatten()
	_t = np.array([theta]).flatten()*180.0/np.pi
	_p = np.array([phi]).flatten()*180.0/np.pi

	gh = ReadCoeffs()
	
	LenPol, dLenPol, SP, dSP = legendre_poly(_t,MaxDeg)

	return int_field(_r, _t, _p, gh, LenPol, dLenPol, SP, dSP)

def CompFuncCart(x,y,z,MaxDeg=10):
	#convert to spherical polar coords
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(z/r)
	phi = (np.arctan2(y,x) + (2*np.pi)) % (2*np.pi)
	
	#call the model
	Br,Bt,Bp = CompFunc(r,theta,phi,MaxDeg)

	#convert to Cartesian (hopefully correctly...)
	cost = np.cos(theta)
	sint = np.sin(theta)
	cosp = np.cos(phi)
	sinp = np.sin(phi)
	Bx = Br*sint*cosp + Bt*cost*cosp - Bp*sinp
	By = Br*sint*sinp + Bt*cost*sinp + Bp*cosp
	Bz = Br*cost - Bt*sint
	
	return Bx,By,Bz
	
def ReadCoeffs():
	'''
	This code is adapted to read the coeffs.dat into the correct format
	for the rest of the code in this file.
	
	'''
	df = pd.read_table(Globals.ModulePath + '__data/coeffs.dat', header=None, names=['g,h','n','m','values'])
	df.rename(columns=df.iloc[0])
	df = df.dropna(axis='columns')
	df.reset_index(inplace=True)
	ghvals = df['values'].tolist()
	gh = np.asarray(ghvals)
	return gh
	
	

def legendre_poly(theta, max_degree):
	'''
	not my function to get the Legendre polynomials
	
	'''
	
	ang_theta = np.deg2rad(theta)
	num_of_data = len(theta)
	num_of_LenPol = 0.5 * (max_degree**2 + 3*max_degree) + 1
    
	LenPol = np.zeros((num_of_data, max_degree+1, max_degree+1))
	dLenPol = np.zeros((num_of_data, max_degree+1, max_degree+1))
	SP_pf = np.zeros((num_of_data, max_degree+1, max_degree+1))
	SP = np.zeros((num_of_data, max_degree+1, max_degree+1))
	dSP = np.zeros((num_of_data, max_degree+1, max_degree+1))
    
	#have to manually define the first 3 polynomials
	P00 = np.ones((len(ang_theta)))
	P10 = np.cos(ang_theta)
	P11 = -np.sin(ang_theta)
    
	#have to manually define the first 3 derivatives of the polynomials
	dP00 = np.zeros((len(ang_theta)))
	dP10 = -np.sin(ang_theta)
	dP11 = -np.cos(ang_theta)
    
	#initialise the matrix - put polynomials and their derivatives into their place in the arrays
	LenPol[:,0,0] = P00
	LenPol[:,1,0] = P10
	LenPol[:,1,1] = P11
    
	dLenPol[:,0,0] = dP00
	dLenPol[:,1,0] = dP10
	dLenPol[:,1,1] = dP11
    
	#Now need to calulate the normalisation components
	SP_pf[:,0,0] = 1.
	SP_pf[:,1,0] = 1.
	SP_pf[:,1,1] = ((-1)**1) * np.sqrt(2.0*factorial(0)/factorial(2))
    
	for i in range(0, max_degree+1):
		for j in range(0,i+1):
			if j == 0:
				SP_pf[:,i,j] = 1.
			else: 
				SP_pf[:,i,j] = ((-1)**j) * np.sqrt(2.0*factorial(i-j)/factorial(i+j))

    
	#calculate the polynomials and their first derivatives - supposedly like a Pascal's triangle method
	if max_degree >= 2:
		for i in range(2, max_degree+1):
			for j in range(0, i+1):
				if j <= i-2:
					LenPol[:, i, j] = (1.0/(i-j)) * (np.cos(ang_theta) * ((2*i)-1) * LenPol[:,(i-1),j] - (i+j-1) * LenPol[:,(i-2),j])
					dLenPol[:, i, j] = (1.0/(i-j)) * (((-np.sin(ang_theta) * ((2*i)-1) * LenPol[:, (i-1),j] + np.cos(ang_theta) * ((2*i)-1) * dLenPol[:,(i-1),j]) - ((i+j-1) * dLenPol[:,(i-2),j])))
				if j == i-1:
					LenPol[:,i,j] = np.cos(ang_theta) * ((2*i)-1) * LenPol[:,(i-1),(i-1)]
					dLenPol[:,i,j] = (-np.sin(ang_theta)) * ((2*i)-1) * LenPol[:, (i-1),(i-1)] + np.cos(ang_theta) * (2*i-1) * dLenPol[:,(i-1),(i-1)]
				if j == i:
					LenPol[:,i,j] = (-1.0) * ((2*i)-1) * np.sin(ang_theta) * LenPol[:,(i-1),(i-1)]
					dLenPol[:,i,j] = (-1.0) * ((2*i)-1) *(np.cos(ang_theta) * LenPol[:,(i-1),(i-1)] + np.sin(ang_theta) * dLenPol[:,(i-1),(i-1)])
    
	#end by combining the polynomials and normalisation
	SP = SP_pf * LenPol
	dSP = SP_pf * dLenPol
    
	return(LenPol, dLenPol, SP, dSP)


def int_field(r, theta, phi, gh, LenPol, dLenPol, SP, dSP):
	'''
	Not my function to get the internal field
	
	'''

	#defining array lengths and setting up angle values from degrees to radians
	len_gh = len(gh)
	l_max = np.sqrt(len_gh)-1
	max_degree = int(l_max)
	ang_theta = np.deg2rad(theta)
	ang_phi = np.deg2rad(phi)
	num_of_data = len(r)
	Br_Forward = np.zeros((num_of_data, len_gh))
	Bt_Forward = np.zeros((num_of_data, len_gh))
	Bp_Forward = np.zeros((num_of_data, len_gh))
    
	#create empty arrays for mag field values for each degree of model, r for rad, t for theta, p for phi
	#final arrays bring all values together
    
	gnmr = np.zeros((num_of_data, max_degree+1, max_degree+1))
	hnmr = np.zeros((num_of_data, max_degree+1, max_degree+1))
    
	gnmt = np.zeros((num_of_data, max_degree+1, max_degree+1))
	hnmt = np.zeros((num_of_data, max_degree+1, max_degree+1))
    
	gnmp = np.zeros((num_of_data, max_degree+1, max_degree+1))
	hnmp = np.zeros((num_of_data, max_degree+1, max_degree+1))
    
	#gnm = np.zeros((3*num_of_data, max_degree+1, max_degree+1))
	#hnm = np.zeros((3*num_of_data, max_degree+1, max_degree+1))

	#4.calculate the field for each coord dimension and degree of model
	#each of these dimensions are then put into the same array and separated later
    
	for i in range(0, max_degree+1):
    
		r_term = (1.0/r)**(i+2)

		for j in range(0, i+1):

			gnmr[:,i,j] = (i+1) * r_term * SP[:,i,j] * np.cos(j*ang_phi)
			hnmr[:,i,j] = (i+1) * r_term * SP[:,i,j] * np.sin(j*ang_phi)
	    
			gnmt[:,i,j] = (-1) * r_term * dSP[:,i,j] * np.cos(j*ang_phi)
			hnmt[:,i,j] = (-1) * r_term * dSP[:,i,j] * np.sin(j*ang_phi)
	     
			gnmp[:,i,j] = ((-1) / np.sin(ang_theta)) * j * r_term * SP[:,i,j] * (-np.sin(j*ang_phi))
			hnmp[:,i,j] = ((-1) / np.sin(ang_theta)) * j * r_term * SP[:,i,j] * np.cos(j*ang_phi)
	     
    #5.sort coeffs for forward matrix
	for i in range(0, max_degree+1):
		for j in range(0, i+1):
			if j == 0:
				idx_coeff = (i-1)**2 +2*(i-1) + 1
				if i == 0:
					idx_coeff = 0
				Br_Forward[:,idx_coeff] = gnmr[:,i,j]
				Bt_Forward[:,idx_coeff] = gnmt[:,i,j]
				Bp_Forward[:,idx_coeff] = gnmp[:,i,j]
             
			else:
				idx_coeff_g = (i-1)**2 + 2*(i-1) + (2*j)
				idx_coeff_h = (i-1)**2 + 2*(i-1) + (2*j + 1)
		
				Br_Forward[:, idx_coeff_g] = gnmr[:,i,j]
				Br_Forward[:, idx_coeff_h] = hnmr[:,i,j]
				Bt_Forward[:, idx_coeff_g] = gnmt[:,i,j]
				Bt_Forward[:, idx_coeff_h] = hnmt[:,i,j]
				Bp_Forward[:, idx_coeff_g] = gnmp[:,i,j]
				Bp_Forward[:, idx_coeff_h] = hnmp[:,i,j]

	#6.Finish by putting all separate degree terms into model field
    
	brs = np.zeros((num_of_data, len_gh))
	bts = np.zeros((num_of_data, len_gh))
	bps = np.zeros((num_of_data, len_gh))

	for i in range(0, len_gh):
		brs[:,i] = gh[i] * Br_Forward[:, i]
		bts[:,i] = gh[i] * Bt_Forward[:, i]
		bps[:,i] = gh[i] * Bp_Forward[:, i]
    
	brtot = np.zeros((num_of_data))
	bttot = np.zeros((num_of_data))
	bptot = np.zeros((num_of_data))
    
    
	brtot = brs.sum(axis=1)
	bttot = bts.sum(axis=1)
	bptot = bps.sum(axis=1)
    
	bmag = (brtot**2 + bttot**2 + bptot**2)**0.5
	intbmag = bmag

	return brtot, bttot, bptot
