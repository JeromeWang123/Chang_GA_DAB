# -------------------------------------------------------------------------------
# Name:        Chang_GA_DAB.py
#
# Purpose:     Code servers as the studying material for paper entitled: 
#              "Genetic Algorithm Assisted Parametric Design of Splitting Inductance in High Frequency GaN-based Dual Active Bridge Converter"
#			   Which is published in IEEE Transactions on Industrial Electronics. 
#
# Author:      Chang Wang (chawa@elektro.dtu.dk) (changwangjerome@gmail.com)
#
# Created:     22-Jul-2021
# Licence:     MIT
# -------------------------------------------------------------------------------

import os
import math
import timeit
import ltspice
import multiprocessing
import numpy as np 
import matplotlib.pyplot as plt
from shutil import copyfile, copy
from PyLTSpice.LTSpiceBatch import LTCommander
from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead


DNA_SIZE = 10            # DNA length
POP_SIZE = 2          # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.005    # mutation probability
N_GENERATIONS = 1
X_BOUND = [0.01, 24]         # x upper and lower bounds


def F(ld,n): # to find the minimum value of the current amplitude
	#def f(ld): # define the batch of LTSpice simulations
	# get script absolute path
	meAbsPath = os.path.dirname(os.path.realpath(__file__))
	# copy .asc file in the same folder for parallel processing purpose
	copyfile(meAbsPath + '/Chang_GA_DAB.asc', meAbsPath + '/Paralleled_Simulations' + '/{}_{}_{}.asc'.format(ld, 'Simulation',n))
	# select spice model
	LTC = LTCommander(meAbsPath + '/Paralleled_Simulations' + '/{}_{}_{}.asc'.format(ld, 'Simulation',n)) # the directive changes to the copied .asc file
	LTC.reset_netlist() # This resets all the changes done to the checklist
	# Define the parameters
	f=1e6
	vin=200
	vout=50
	t=1/f
	halft=t/2
	duty=0.46
	d=duty*t
	phase=0.20*t
	delay=phase+halft
	npp=646e-6
	ns=npp/16
	ldp=ld*1e-6
	lds=(24.01e-6-ldp)/16
	cp1=28e-12 # PCB derived
	cp2=430e-12 # PCB derived
	lk1=1300e-9 # PCB derived
	lk2=81e-9 # PCB derived

	print('Ldp is: ' + str(ldp))
	# Redefining parameters in the netlist
	LTC.set_parameters(T=t)
	LTC.set_parameters(halfT=halft)
	LTC.set_parameters(d=d)
	LTC.set_parameters(Phase=phase)
	LTC.set_parameters(Delay=delay)
	LTC.set_parameters(Npri=npp)
	LTC.set_parameters(Nsec=ns)
	LTC.set_parameters(Ldp=ldp)
	LTC.set_parameters(Lds=lds)
	LTC.set_parameters(Cp1=cp1)
	LTC.set_parameters(Cp2=cp2)
	LTC.set_parameters(Lk1=lk1)
	LTC.set_parameters(Lk2=lk2)

	# define simulation
	LTC.add_instructions( # Changing the simulation file
	"; Simulation settings",
	" .tran 0 50u 40u 1n startup", # the time to start storing data should be wait until steady state
	)

	rawfile, logfile = LTC.run() # Runs the simulation with the updated netlist

	LTR = LTSpiceRawRead(meAbsPath + '\\Paralleled_Simulations' + '\\{}_{}_{}_{}.raw'.format(ld, 'Simulation',n, 'run')) # the directive changes to the copied .asc fil

	ILdp = LTR.get_trace("I(Ldp)")
	ILds = LTR.get_trace("I(Lds)")
	Vo = LTR.get_trace("V(n002)")
	Vd=LTR.get_trace("V(n001)")
	Vs=LTR.get_trace("V(n006)")
	x = LTR.get_trace('time') # Gets the time axis
	steps = LTR.get_steps()

	for step in range(len(steps)):
		plt.plot(x.get_time_axis(step), ILdp.get_wave(step), label=steps[step])
		ILdpamp=max(ILdp.get_wave(step))
		print('The current amplitude is: ' + str(ILdpamp))
		arr_mean=np.mean(np.abs(ILdp.get_wave(step)))
		print('The current RMS is: ' + str(arr_mean))
		ILdsamp=max(ILds.get_wave(step))
		print('The current amplitude of sec is: ' + str(ILdsamp))
		arr_mean_sec=np.mean(np.abs(ILds.get_wave(step)))
		print('The current RMS of sec is: ' + str(arr_mean_sec))
		kkk=2.82e8*(math.pow(arr_mean,2)*math.pow(ILdpamp,2)*math.pow(ldp,2)*2+math.pow(arr_mean_sec,2)*math.pow(ILdsamp,2)*math.pow(lds,2))
	return kkk	

# find non-zero fitness for selection, using peak(Minimum) value at this moment
def get_fitness(pred): 
	return np.power((np.max(pred)-pred),4)

# convert binary DNA to decimal and normalize it to a ran
def translateDNA(pop): 
	return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) *(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]


def select(pop, fitness):    # nature selection wrt pop's fitness
	idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
	 p=(fitness/fitness.sum()))
	return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
	if np.random.rand() < CROSS_RATE:
		i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop

		cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
		parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
	return parent

def mutate(child):
	for point in range(DNA_SIZE):
		if np.random.rand() < MUTATION_RATE:
			child[point] = 1 if child[point] == 0 else 0
		return child


if __name__ == '__main__':
	cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=cores)
	print('Cores are '+ str(cores))
	start=timeit.default_timer() # calculate the computating time
	pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA
	print('first pop is:')
	print(pop)
	plt.xlabel('The primary external inductance value (uH)', fontsize=14)
	plt.ylabel('The total copper loss of both inductors (W)', fontsize=14)
	plt.axis([-1, 26, 0, 3]) # set the axis range
	plt.ion()       # something about plotting
	x = np.linspace(*X_BOUND, 2)
	print('x is: ')
	print(x)
	n=0 # used for naming the files in case same filename causing simualtion batching error
	for _ in range(N_GENERATIONS):
		nth=range(n+1,n+POP_SIZE+1)
		n=n+POP_SIZE # number of generations, used for naming the files in case same filename causing simualtion batching error
		print(' translated DNA is:')
		print(translateDNA(pop))
		print(type(translateDNA(pop)))
		zip_arg=list(zip(translateDNA(pop),nth)) # using zip() to input two lists as variables into function
		F_values = list(pool.starmap(F,zip_arg)) # compute function value by extracting D
		print('F_value is : (the total inductors copper loss)')
		print(F_values) # print the inductors copper loss
		# something about plotting
		if 'sca' in globals(): sca.remove()
		sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)
		fitness = get_fitness(F_values)
		print("Most fitted DNA: ", pop[np.argmax(fitness), :])

		
		pop = select(pop, fitness)
		pop_copy = pop.copy()
		for parent in pop:
			child = crossover(parent, pop_copy)
			child = mutate(child)
			parent[:] = child       # parent is replaced by its child
		print('Next generation is:')
		print(pop)
	plt.ioff()
	plt.show()

	stop=timeit.default_timer() # calculate the computating time
	print('Time: ', stop-start)
