# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import scipy.linalg as sp
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
with open('data2.pickle', 'rb') as f:
    data = pickle.load(f)


b = data[1]
#--------------Q6---------------
#------a------
def t_funciton(data,x):
    row = np.zeros(5)
    row[0] =  x[0]*1
    row[1] = x[1]*np.sin(data)
    row[2] = x[2]*np.sin(2*data)
    row[3] = x[3]*np.sin(3*data)
    row[4] = x[4]*np.sin(4*data)
    return row

ones = np.array([1,1,1,1,1])

A = t_funciton(data[0][0],ones)
for i in range(1,50):
    row = t_funciton(data[0][i],ones)
    A = np.append(A,row)
A = A.reshape(50,5)

b = data[1]
#------b------
x= np.linalg.lstsq(A, b,rcond=None)[0]

#------c------




#------d------
inputs = np.linspace(0.,4.,num = 1000)
def sums(data,x):
    return np.sum(t_funciton(data,x))


outputs1 = np.zeros(1000)
for i in range(1000):
    outputs1[i] = sums(inputs[i],x);



plt.title('"Question 6. Data and fitted polynomial"')
plt.plot(inputs,outputs1, 'g')
plt.plot(data[0],data[1], linestyle = 'dotted')
plt.show()

Ax = np.matmul(A,x)

error = np.linalg.norm(Ax-b)


#--------------Q7---------------
#------a------

I = np.identity(A.shape[0])

topHalf = np.hstack((I,A))
O = np.zeros((A.shape[1],A.shape[1]))

bottomHalf = np.hstack((A.T,O))



AugM = np.vstack((topHalf,bottomHalf))

#------b------
P,L,U = sp.lu(AugM, permute_l=False)

o = np.zeros(5)

bo = np.hstack((b,o))

Pb = np.matmul(P,bo)



s = sp.solve_triangular(L,Pb, lower = True)

rx = sp.solve_triangular(U,s)

print("\n\n\n")

#------c------
x = rx[-5:55]
print("coefficients of x is{0}".format(x))

outputs1 = np.zeros(1000)
for i in range(1000):
    outputs1[i] = sums(inputs[i],x);



plt.title('"Question 7. Data and fitted polynomial"')
plt.plot(inputs,outputs1, 'b')
plt.plot(data[0],data[1], linestyle = 'dotted')
plt.show()
errors = np.linalg.norm(rx[0:-5])

print("r =|Ax - b| = {0} ".format(errors))

#--------------Q8---------------
#------a------
A = np.random.randn(784,700)
data = np.load('/Users/oluwaseuncardoso/Downloads/npMnist.npy')
x = data[1]

z= np.linalg.lstsq(A, x,rcond=None)[0]

x_ = np.matmul(A,z)


p = np.reshape(x,(28,28))


plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 8.a uncompressed image')
plt.show()

p2 = np.reshape(x_,(28,28))
plt.imshow(p2, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 8.a compressed image')
plt.show()

plt.subplot(211)
plt.imshow(p, interpolation='nearest', cmap='Greys')

plt.subplot(212)
plt.suptitle('Question 8.a compressed vs uncompressed image', fontsize=12)
plt.imshow(p2, interpolation='nearest', cmap='Greys')

error = np.linalg.norm(x-x_)
#------b------
q,r = np.linalg.qr(A)

O = np.zeros((84,700))



RO = np.vstack((r,O))


b = data[1]


b = np.matmul(q.T,b)



x_ = sp.solve_triangular(r,b)

#------c------
# IDK
#------d------
# IDK
#------e------
# IDK
#------f------
# IDK
