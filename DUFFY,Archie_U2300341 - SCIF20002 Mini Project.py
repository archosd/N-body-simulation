import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def AccCalc(pos, mass, epsilon, sigma, softening):

	"""
	Calculate acceleration for each particle.

	Args:
	- pos (ndarray): N x 3 matrix of positions.
	- mass (ndarray): N x 1 vector of masses.
	- epsilon (float): Lennard-Jones potential energy well depth.
	- sigma (float): Lennard-Jones distance parameter.
	- softening (float): Softening parameter.

	Returns:
	- a (ndarray): N x 3 matrix containing acceleration components.
	"""

	# positions r = [x,y,z] for all particles
	x = pos[:, 0:1]
	y = pos[:, 1:2]
	z = pos[:, 2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores r for all particle pairwise particle separations 
	r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)

	# Calculate force magnitude using Lennard-Jones force
	force_magnitude = (24 * epsilon / sigma) * ((-2) * (sigma / r) ** 13 + (sigma / r) ** 7)

	# Compute acceleration components
	ax = (force_magnitude * dx / r) @ (1 / mass)
	ay = (force_magnitude * dy / r) @ (1 / mass)
	az = (force_magnitude * dz / r) @ (1 / mass)

	# pack together the acceleration components
	a = np.hstack((ax, ay, az))

	return a

def EnergyCalc(pos, vel, mass, epsilon, sigma,softening):
	"""
	Calculate kinetic and potential energy of the system.

	Args:
	- pos (ndarray): N x 3 matrix of positions.
	- vel (ndarray): N x 3 matrix of velocities.
	- mass (ndarray): N x 1 vector of masses.
	- epsilon (float): Lennard-Jones potential energy well depth.
	- sigma (float): Lennard-Jones distance parameter.
	- softening (float): Softening parameter.

	Returns:
	- KE (float): Kinetic energy of the system.
	- PE (float): Potential energy of the system.
	"""
	# Kinetic Energy:
	KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

	# positions r = [x,y,z] for all particles
	x = pos[:, 0:1]
	y = pos[:, 1:2]
	z = pos[:, 2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores r for all particle pairwise particle separations
	r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)

	# LJ Potential Energy:
	PE = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

	# Exclude self-interaction and double counting
	np.fill_diagonal(PE, 0)
	PE = np.triu(PE).sum()

	return KE, PE

def TempCalc(KE,N):

	"""
	Calculate temperature of the system.

	Args:
	- KE (float): Kinetic energy of the system.
	- N (int): Number of particles.

	Returns:
	- T (float): Temperature of the system.
	""" 
	T = (2*KE)/(6*N)
	return T

def PressureCalcX(pos, vel, mass, A, delta_t):
	"""
	Calculate pressure in the x-direction.

	Args:
	- pos (ndarray): N x 3 matrix of positions.
	- vel (ndarray): N x 3 matrix of velocities.
	- mass (ndarray): N x 1 vector of masses.
	- A (float): Area lying in the y-z plane.
	- delta_t (float): Time interval.

	Returns:
	- Px (float): Pressure in the x-direction.
	- Py (float): Pressure in the y-direction.
	- Pz (float): Pressure in the z-direction.
	- P_tot (float): Total pressure.
	"""
	# Calculate the x-component of momentum for particles crossing the plane
	px = mass * np.sqrt(np.power(vel[:, 0],2))

	# Filter particles crossing the plane in the time interval delta_t
	crossing_particles = (pos[:, 0] <= 0) & ((pos[:, 0] + vel[:, 0] * delta_t) >= 0)

	# Calculate the sum of x-component of momentum for crossing particles
	sum_px = np.sum(px[crossing_particles])

	# Calculate pressure in the x-direction
	Px = sum_px / (A * delta_t)

	# Assuming isotropic pressure
	P = Px

	# Calculate pressure in y and z direction
	Py = P
	Pz = P

	P_tot = P*3

	return Px, Py, Pz,P_tot

def PressureCalcY(pos, vel, mass, A, delta_t):
	"""
	Calculate pressure in the y-direction.

	Args:
	- pos (ndarray): N x 3 matrix of positions.
	- vel (ndarray): N x 3 matrix of velocities.
	- mass (ndarray): N x 1 vector of masses.
	- A (float): Area lying in the x-z plane.
	- delta_t (float): Time interval.

	Returns:
	- Px (float): Pressure in the x-direction.
	- Py (float): Pressure in the y-direction.
	- Pz (float): Pressure in the z-direction.
	- P_tot (float): Total pressure.
	"""
	# Calculate the y-component of momentum for particles crossing the plane
	py = mass * np.sqrt(np.power(vel[:, 1],2))

	# Filter particles crossing the plane in the time interval delta_t
	crossing_particles = (pos[:, 1] <= 0) & ((pos[:, 1] + vel[:, 1] * delta_t) >= 0)

	# Calculate the sum of y-component of momentum for crossing particles
	sum_py = np.sum(py[crossing_particles])

	# Calculate pressure in the y-direction
	Py = sum_py / (A * delta_t)

	# Assuming isotropic pressure
	P = Py

	# Calculate pressure in x and z direction
	Px = P
	Pz = P

	P_tot = P*3

	return Px, Py, Pz, P_tot

def PressureCalcZ(pos, vel, mass, A, delta_t):
	"""
	Calculate pressure in the z-direction.

	Args:
	- pos (ndarray): N x 3 matrix of positions.
	- vel (ndarray): N x 3 matrix of velocities.
	- mass (ndarray): N x 1 vector of masses.
	- A (float): Area lying in the x-y plane.
	- delta_t (float): Time interval.

	Returns:
	- Px (float): Pressure in the x-direction.
	- Py (float): Pressure in the y-direction.
	- Pz (float): Pressure in the z-direction.
	- P_tot (float): Total pressure.
	"""
	# Calculate the z-component of momentum for particles crossing the plane
	pz = mass * np.sqrt(np.power(vel[:, 2],2))

	# Filter particles crossing the plane in the time interval delta_t
	crossing_particles = (pos[:, 2] <= 0) & ((pos[:, 2] + vel[:, 2] * delta_t) >= 0)

	# Calculate the sum of z-component of momentum for crossing particles
	sum_pz = np.sum(pz[crossing_particles])

	# Calculate pressure in the z-direction
	Pz = sum_pz / (A * delta_t)

	# Assuming isotropic pressure
	P = Pz

	# Calculate pressure in x and y direction
	Px = P
	Py = P

	P_tot = P*3

	return Px, Py, Pz, P_tot

def GridIntialisation(N, box_size):
	"""
	Initialize particle positions on a grid inside the simulation box.

	Args:
	- N (int): Number of particles.
	- box_size (float): Size of the simulation box.

	Returns:
	- pos (ndarray): N x 3 array containing particle positions.
	"""
	# Calculate grid size for cube arrangement of particles
	grid_size = int(np.ceil(np.power(N, 1/3)))  
	# Calculate spacing between particles in each dimension
	spacing = box_size / grid_size  

	# Create grid of coordinates
	grid_points = np.linspace(-box_size/2 + spacing/2, box_size/2 - spacing/2, grid_size)
	grid_x, grid_y, grid_z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')

	# Flatten the grid to get positions
	pos_grid = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))

	# Ensure number of particles does not exceed grid points
	pos = np.zeros((N, 3))
	pos[:N] = pos_grid[:N]

	return pos

def NonPeriodicBoundaryConditions(pos, prev_pos, vel, length):
	"""
	Handle particles leaving the simulation box.

	Args:
	- pos (ndarray): Array of shape (N, 3) containing current positions of particles.
	- prev_pos (ndarray): Array of shape (N, 3) containing previous positions of particles.
	- vel (ndarray): Array of shape (N, 3) containing velocities of particles.
	- length (float): Size of the simulation box.

	Returns:
	- pos (ndarray): Updated positions of particles.
	- vel (ndarray): Updated velocities of particles.
	"""
	exceed_box = (pos[:, 0] > length) | (pos[:, 0] < 0) | (pos[:, 1] > length) | (pos[:, 1] < -length) | (pos[:, 2] > length) | (pos[:, 2] < -length)

	left_box = (pos[:, 0] < -length)
	right_box = (pos[:, 0] > length)
	top_box = (pos[:, 1] < -length)
	bottom_box = (pos[:, 1] > length)
	front_box = (pos[:, 2] < -length)
	back_box = (pos[:, 2] > length)

	if np.any(exceed_box):
		pos[exceed_box] = prev_pos[exceed_box]
		vel[left_box | right_box, 0] *= -1
		vel[top_box | bottom_box, 1] *= -1
		vel[front_box | back_box, 2] *= -1

	return pos, vel

def main(n,velocity,size):
	"""
	Main n-body simulation.

	Args:
	- n (int) : Number of particles.
	- velocity (float) : Initial velocity.

	Returns:
	- P_ave (float) : Time averaged pressure value.
	- temp (floar) : Temperature value calculated from kinetic energy.
	
	"""
	# Simulation parameters
	N         = n 					# Number of particles
	t         = 0     				# current time of the simulation
	tEnd      = 800   				# time at which simulation ends
	dt        = 0.01   				# timestep
	softening = 0.2    				# softening length
	sigma     = 1  					# sigma value - set to one for dimensionless 
	epsilon   = 1					# epsilon value - set to one for dimensionless
	box_size  = size					# total length of one side of the box 
	length    = box_size/2			# length from 0 to edge of the box (leave as half of box_size)
	A         = length*2			# area for pressure calculation
	V         = velocity			# initial Velocity
	buffer    = 0					# buffer for 2D plot
	initialiseRandomPos = False 	# random or grid initial conditions
	initialiseRandomVel = False	# if set to True MUST define V = some value above
	ThreeDimensionalPlot = False 	# switch between 3D and 2D plots
	LivePlot = False 				# switch on for plotting as the simulation goes along
	Plot = False 					# Show plots
	ShowAllHistory	= False  	# Show entire trace history 
	AssumeIsotropicPressure = True 	# toggle between isotropic calculation of pressure and calculation in each direction

	# Generate Initial Conditions
 
	np.random.seed(42)# set the random number generator seed

	mass = np.ones((N,1))

	pos = np.zeros((N,3))

	if initialiseRandomPos == True:
		pos  = np.random.randn(N,3)*length/2  # randomly selected positions and velocities
	
	if initialiseRandomPos == False:
		pos = GridIntialisation(N, box_size)
	
	if initialiseRandomVel:
		vel  = np.random.randn(N,3)*length/6	

	if initialiseRandomVel == False:
		vel = np.ones((N,3))*V

	# # # # # #Generate positions with two particles on the x-axis, set N = 2 and uncomment.	
	# pos = np.zeros((N, 3))
	# vel = np.zeros ((N,3))
	# vel[0,0] = 0
	# pos[1, 0] = 1.0  # Second particle is located 1 unit along the x-axis

	# calculate initial accelerations
	acc = AccCalc(pos, mass, sigma, epsilon, softening)

	# calculate initial energy of system
	KE, PE = EnergyCalc( pos, vel,mass,sigma,epsilon,softening)
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# save energies, particle positions for plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	acc_save = np.zeros((N,3,Nt+1))
	acc_save[:,:,0] = acc
	vel_save = np.zeros((N,3,Nt+1))
	vel_save[:,:,0] = vel
	KE_save = np.zeros(Nt+1)
	KE_save[0] = KE
	PE_save = np.zeros(Nt+1)
	PE_save[0] = PE
	t_all = np.arange(Nt+1)*dt
	
	# save pressure
	if AssumeIsotropicPressure:
		P_save = np.zeros(Nt+1)
		P_save[0] = 0

	if AssumeIsotropicPressure ==False: 
		P_save_x = np.zeros(Nt+1)
		P_save_x[0] = 0
		P_save_y = np.zeros(Nt+1)
		P_save_y[0] = 0
		P_save_z = np.zeros(Nt+1)
		P_save_z[0] = 0
	
	# prep figures
	if Plot == True:
	
		if ThreeDimensionalPlot == True:

			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111, projection='3d')

		if ThreeDimensionalPlot == False: 

			grid = plt.GridSpec(4, 1, wspace=0.0, hspace=0.3)
			ax2 = plt.subplot(grid[0:2,0])
			ax3 = plt.subplot(grid[2,0])
			ax4 = plt.subplot(grid[3, 0])
	
	# Simulation Main Loop
	for i in range(Nt):

		# (1/2) velocity update
		vel += acc * dt/2.0
		
		# position drift
		pos += vel * dt

		# variable to store previous positions
		prev_pos = pos.copy()  

		# handle particles leaving the box
		pos, vel = NonPeriodicBoundaryConditions(pos, prev_pos, vel, length)

		# update accelerations with lennard-jones force
		acc = AccCalc(pos, mass, sigma,epsilon, softening)
		
		# (1/2) velocity update with updated acceleration
		vel += acc * dt/2.0
		
		# update time
		t += dt
		
		# get energy of system
		KE, PE  = EnergyCalc( pos, vel, mass, epsilon,sigma,softening )

		# get pressure of system

		if AssumeIsotropicPressure: 
			Px, Py, Pz, P_tot = PressureCalcX(pos, vel, mass, A, dt)
			P_save[i+1] = P_tot
		
		if AssumeIsotropicPressure == False:
			Px, Py, Pz, P_tot_x = PressureCalcX(pos, vel, mass, A, dt)

			P_save_x[i+1] = Px

			Px, Py, Pz, P_tot_y = PressureCalcY(pos, vel, mass, A, dt)

			P_save_y[i+1] = Py

			Px, Py, Pz, P_tot_z = PressureCalcZ(pos, vel, mass, A, dt)

			P_save_z[i+1] = Pz

		# save energies and positions for plotting traces
		pos_save[:,:,i+1] = pos
		acc_save[:,:,i+1] = acc
		vel_save[:,:,i+1] = vel
		KE_save[i+1] = KE
		PE_save[i+1] = PE
		
		# real time plot
		if Plot and LivePlot or Plot and (i == Nt-1):
			
			if ThreeDimensionalPlot == True:

				if ShowAllHistory:
					xx = pos_save[:,0]   
					yy = pos_save[:,1]
					zz = pos_save[:,2]

				if ShowAllHistory == False:
					xx = pos_save[:,0,max(i-400,0):i+1]
					yy = pos_save[:,1,max(i-400,0):i+1]
					zz = pos_save[:,2,max(i-400,0):i+1]

				plt.sca(ax1)
				plt.cla()
				ax1.scatter(xx,yy,zz,alpha = 0.25,color='blue',s = .01)
				ax1.scatter(pos[:,0],pos[:,1],pos[:,2],color='red', s=20)
				ax1.set(xlim=(-length-buffer,length+buffer), ylim=(-length-buffer,length+buffer),zlim = (-length - buffer, length + buffer))
				ax1.set_box_aspect([1, 1, 1])

			if ThreeDimensionalPlot == False:

				# 2D trace
			

				if ShowAllHistory:
					xx = pos_save[:,0]   
					yy = pos_save[:,1]

				if ShowAllHistory == False:
					xx = pos_save[:,0,max(i-400,0):i+1]
					yy = pos_save[:,1,max(i-400,0):i+1]

				plt.sca(ax2)
				plt.cla()
				plt.scatter(xx,yy,s=1,color=[.7,.7,1])
				plt.scatter(pos[:,0],pos[:,1],s=10,color='crimson')
				ax2.set(xlim=(-length-buffer,length+buffer), ylim=(-length-buffer,length+buffer))
				ax2.set_aspect('equal', 'box')

				# KE and PE plots

				plt.sca(ax3)
				plt.cla()
				plt.scatter(t_all,KE_save,color='red',s=1,label='KE')
				plt.scatter(t_all,PE_save,color='blue',s=1,label='PE')
				plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot')
				ax3.set(xlim=(0, tEnd))
				
				# pos vel and acc plots

				plt.sca(ax4)
				plt.cla()
				plt.plot(t_all[:i + 1], acc_save[0, 0, :i + 1], label='Acceleration X (σ/τ^2)', color='green')
				plt.plot(t_all[:i + 1], vel_save[0, 0, :i + 1], label='Velocity X (σ/τ)', color='purple')
				plt.plot(t_all[:i + 1], pos_save[0, 0, :i + 1], label='Position X (σ)', color='orange')
				ax4.set(xlim=(0, tEnd))
				ax4.legend()
			
			plt.pause(0.001)

	# pressure calculation
	if AssumeIsotropicPressure == False:
		P_ave_x = np.sum(P_save_x[1000:])/(N*(t-10))
		P_ave_x_err = np.std(P_save_x[1000:])/(N*(t-10))
		P_ave_y = np.sum(P_save_y[1000:])/(N*(t-10))
		P_ave_y_err = np.std(P_save_y[1000:])/(N*(t-10))
		P_ave_z = np.sum(P_save_z[1000:])/(N*(t-10))
		P_ave_z_err = np.std(P_save_z[1000:]/(N*(t-10)))
	
	if AssumeIsotropicPressure:
		P_ave = np.sum(P_save[1000:])/(N*(t-10))

	# final plots

	if Plot == True:     

		if ThreeDimensionalPlot == True:

			plt.sca(ax1)
			ax1.set_xlabel('X Position (σ)')
			ax1.set_ylabel('Y Position (σ)')
			ax1.set_zlabel('Z Position (σ)')

		if ThreeDimensionalPlot == False:
		
			plt.sca(ax2)
			plt.xlabel('X Position (σ)')
			plt.ylabel('Y Position (σ)')

			plt.sca(ax3)
			plt.xlabel('Time (τ)')
			plt.ylabel('Energy (ε)')
			ax3.legend(loc='upper right')

			plt.sca(ax4)
			plt.xlabel('Time (τ)')
			plt.ylabel('Kinematic Variables')
			ax4.legend(loc='upper right')
		
		# Save figure
		plt.savefig('nbody.png',dpi=240)
		plt.show()

		#Plot pressure 

		if AssumeIsotropicPressure ==False:
			
			print(P_save_x,P_ave_x, P_ave_y,P_ave_z)	
			plt.scatter(t_all[1000:], P_save_x[1000:], marker='o', label="Pressure in x direction")
			plt.scatter(t_all[1000:], P_save_y[1000:], marker='o', label="Pressure in y direction", color="red")
			plt.scatter(t_all[1000:], P_save_z[1000:], marker='o', label="Pressure in z direction", color="green")
			plt.text(0.5, -0.1, f"Avg. Pressure (x): {P_ave_x:.3f} $\pm$ {P_ave_x_err:.3f}, Avg. Pressure (y): {P_ave_y:.3f}$\pm$ {P_ave_y_err:.3f}, Avg. Pressure (z): {P_ave_z:.3f}$\pm$ {P_ave_z_err:.3f}",
				horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

			plt.ylabel('Pressure ($m_a σ^{-1} τ^{-2}$)')
			plt.xlabel('Time (τ)')
			plt.legend()
			plt.show()
		
		if AssumeIsotropicPressure:
			plt.scatter(t_all[1000:], P_save[1000:], marker='o', label="Pressure in x direction")
			plt.text(0.5, -0.1, f"Avg. Pressure: {P_ave:.3f}.")
			plt.ylabel('Pressure ($m_a σ^{-1} τ^{-2}$)')
			plt.xlabel('Time (τ)')
			plt.legend()
			plt.show()
		

	#diagnostic print statements /temp and pressure 
	temp = np.round(TempCalc(KE,N),5)
	return P_ave , temp


# if __name__== "__main__":
#   main(1,2.5,30)


# TEST - PRESSURE AND VOLUME

# box_sizes = np.linspace(30,50,19)
# volume = np.array(np.power((box_sizes),3))
# print(box_sizes, volume)

# average_pressures = []


# for box_size in tqdm(box_sizes):
# 	average_pressure, temp = main(50, 2.5, box_size)
# 	average_pressures.append(average_pressure)

# # fit a line of best fit
# coefficients = np.polyfit(1/volume, average_pressures, 1)
# line_of_best_fit = np.poly1d(coefficients)

# # plot average pressures against box size
# plt.scatter(1/volume, average_pressures, marker='o', label='Data')
# plt.plot(1/volume, line_of_best_fit(1/volume), color='red', label='Line of Best Fit')
# plt.xlabel('1/Volume($\sigma^3$)')
# plt.ylabel('Average Pressure($m_a σ^{-1} τ^{-2}$)')
# plt.title('Average Pressure vs. 1/Volume')
# plt.grid(True)
# plt.legend()
# plt.show()

# plt.savefig('volume.png', dpi=240)



# # TEST - PRESSURE AND N	

# N_values = np.arange(5,105,5)
# print(N_values)

# # Lists to store average pressures for each N value
# average_pressures = []

# # Loop over each N value
# for n in tqdm(N_values):
# 	average_pressure, temp = main(n,2.5,30)
# 	average_pressures.append(average_pressure)

# # fit a line of best fit
# coefficients = np.polyfit(N_values, average_pressures, 1)
# line_of_best_fit = np.poly1d(coefficients)

# # Plot average pressures against N values
# plt.scatter(N_values, average_pressures, marker='o')
# plt.plot(N_values, line_of_best_fit(N_values), color='red', label='Line of Best Fit')
# plt.xlabel('Number of Particles (N)')
# plt.ylabel('Average Pressure($m_a σ^{-1} τ^{-2}$)')
# plt.title('Average Pressure vs. Number of Particles')
# plt.grid(True)
# plt.show()


# plt.savefig('pressure.png',dpi=240)


# TEST - PRESSURE AND TEMP

V_values = np.linspace(0.1,2.5,25)  # List of velocity values to loop over

print(V_values)

# Lists to store average pressures for each velocity value
average_pressures = []
temps = []

# Loop over each velocity value
for v in tqdm(V_values):
	# The main function of the simulation (main()) should be modified to accept velocity as a parameter
	average_pressure, temp = main(50, v, 30)
	average_pressures.append(average_pressure)
	temps.append(temp)
	print(v,temp)

# fit a line of best fit
coefficients = np.polyfit(V_values, average_pressures, 1)
line_of_best_fit = np.poly1d(coefficients)

plt.scatter(temps, average_pressures, marker='o')
plt.plot(V_values, line_of_best_fit(V_values), color='red', label='Line of Best Fit')
plt.xlabel('Temperature(k)')
plt.ylabel('Average Pressure($m_a σ^{-1} τ^{-2}$)')
plt.title('Average Pressure vs. Temperature')
plt.grid(True)
plt.show()
plt.savefig('temp.png',dpi=240)