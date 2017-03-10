import numpy as np 
import matplotlib.pyplot as plt


def cost_Function(A, b, x):
	return 1/float(len(b)) *np.linalg.norm(np.dot(A,x)-b)**2

def step_Gradient(A, b, x, learning_Rate):
	N = float(len(b))
	x_Gradient= 2 / N * np.dot(np.transpose(A) , (np.dot(A , x) - b))
	new_x = x - learning_Rate*x_Gradient
	return new_x

def gradient_Descent_Runner(A, b, starting_x, learning_Rate, num_Iterations):
	x = starting_x
	j = 0
	for i in range(num_Iterations):
		x = step_Gradient(A, b, x, learning_Rate)
		
	return x

def linear_f(x,t):
	return x[0]*t+x[1]*t

def run():
	#set parameters (hyperparameters?)
	num_Iterations = 100
	learning_Rate= .0001
	N = 2 # number of dimentions on the inputs

	#creation of input data
	A = np.linspace(0,10,100) + np.random.normal(0,1,100)
	for i in range(N-1):
		A =  np.column_stack((A,np.linspace(0,10,100) + np.random.normal(0,1,100)))

	#creation of output data
	true_x = np.linspace(0,100,N)
	b = np.dot(A,true_x) + np.random.normal(0,1,100)	
	x = np.zeros(N)
	
	
	
	##for testing with one/two dimentional cases
	t = np.linspace(-2,12,100000)	
	plt.figure(1)
	plt.subplot(211)
	plt.plot(A[:,0],b,'ro', t,linear_f(x,t),'k')
	plt.title('Initial Guess')

	print('Starting gradient descent at error = {0}'.format(cost_Function(A,b,x)))
	print('Running...')

	x = gradient_Descent_Runner(A,b,x, learning_Rate, num_Iterations)

	plt.subplot(212)
	plt.plot(A[:,0],b,'ro', t,linear_f(x,t),'k')
	plt.title('After Gradient Descent')
	

	print("After {0} iterations, error = {1}".format(num_Iterations, cost_Function(A,b,x)))

	plt.show()

run()
