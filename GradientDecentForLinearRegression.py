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
	for i in range(num_Iterations):
		x = step_Gradient(A, b, x, learning_Rate)
	return x

def linear_f(x,t):
	return x[0]*t+x[1]

def run():
	learning_Rate= .0001
	A = np.linspace(0,10,100) + np.random.normal(0,1,100)
	ones = np.ones(100)	
	A =  np.column_stack((A,ones))	
	b = 2*A[:,0] + 1 + np.random.normal(0,1,100)	
	x = np.zeros(2)
	print(x)
	num_Iterations = 100000
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
	

	print("After {0} iterations, x_1 = {1}, x_2 ={2}, error = {3}".format(num_Iterations,x[0],x[1], cost_Function(A,b,x)))

	plt.show()

run()
