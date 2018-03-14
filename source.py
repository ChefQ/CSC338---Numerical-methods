import math
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as sp


#--------------Q4---------------

#------a------
def ap_function(x):
    return x - (x ** 3) / math.factorial(3)

let = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]


for x in let:
    f_error = np.sin(x) - ap_function(x)
    x_ = np.arcsin(ap_function(x))
    b_error = x - x_
    print("f(x)={0} f'(x)={1} \n\tforward error = {2}".format(np.sin(x),ap_function(x),f_error))
    print("x ={0} x'={1} \n\tbackward error = {2}\n".format(x,x_,b_error))

print("for values x=3.0 and x=4.0 does not produse a backward error because: \n "
      "The domanin values for arcsin(x) are real numbers from -1 t0 1.\n"
      " values f'(4.0)=-6.666666666666666 and f'(3.0)=-1.5 are both beyond the domain values for arcsin(x) ")



#------c------


inputs = np.linspace(-3.,3.,num = 1000)
outputs1 = np.apply_along_axis(ap_function,0,inputs)
outputs2 = np.apply_along_axis(np.sin,0,inputs)
plt.title('“Question 4(c): f(x) and sin(x)”')
plt.plot(inputs,outputs1, 'r')
plt.plot( inputs,outputs2, 'b')
plt.show()

#--------------Q5---------------

#------b------

def algorithm1(x,N):
    t = x
    y = x
    negXsq = -x**2
    for n in range(1, N+1):
        t = t*negXsq/(2*n*(2*n + 1))
        y = y + t
    return y



#------ c ------
inputs = np.linspace(-7.,7.,num = 1000)
outputs1 = np.apply_along_axis(algorithm1, 0, arr = inputs, N=6 )
plt.title('Question 5(c): f(x), N=6')
plt.plot(inputs,outputs1, 'r')
plt.show()
inputs = np.linspace(-11.,11.,num = 1000)
outputs2 = np.apply_along_axis(algorithm1, 0, arr = inputs, N=11 )
plt.title('Question 5(c): f(x), N=11')
plt.plot(inputs,outputs2, 'B')
plt.show()

#------ e ------
inputs = np.linspace(0.,30.,num = 1000)
outputs1 = np.apply_along_axis(algorithm1, 0, arr = inputs, N=100 )
plt.title('Question 5(e): f(x), N=100')
plt.plot(inputs,outputs1, 'r')
plt.show()
inputs = np.linspace(0.,30.,num = 1000)
outputs2 = np.apply_along_axis(np.sin, 0, arr = inputs)
plt.title('Question 5(e): sin(x)')
plt.plot(inputs,outputs2, 'B')
plt.show()

#------ f ------
inputs = np.linspace(30.,40.,num = 1000)
outputs1 = np.apply_along_axis(algorithm1, 0, arr = inputs, N=100 )
plt.title('Question 5(f): f(x), N=100')
plt.plot(inputs,outputs1, 'r')
plt.show()

#------ g ------
inputs = np.linspace(40.,50.,num = 10000)
outputs1 = np.apply_along_axis(algorithm1, 0, arr = inputs, N=100 )
plt.title('Question 5(g): f(x), N=100')
plt.plot(inputs,outputs1, 'r')
plt.show()

#--------------Q7---------------

def expression(n):
    return ((2.0**n)+10.0)-((2.0**n+5.0)+5.0)


a = []

for num in range(1000):
    if (expression(num) != 0 ):
        a.append(num)

print("positive integers n for which the value of this expression is non-zero\n{0}\n".format(a))

#--------------Q9---------------
def kindofsquare(A):
    row,col = A.shape
    count = 0
    mult = np.zeros((row, row))
    for k in range(row):
        for j in range(row):
            sum = 0
            for i in range(col):
                x = A[k][i]
                y = A[j][i]
                sum+=(x*y)
            mult[k,j]= sum
            count +=1
    print("in this case we have {0} floating-point multiplications".format(count))
    return mult
A = np.array([[1,2],[4,5]])




def compare(I,K):
    A = np.random.randn(I,K)
    start = time.time()
    B1 = kindofsquare(A)
    end = time.time()
    print("Execution time of KindofSquare: {0}".format(end-start))
    #print("\nkindofsquare: {0}".format(B1))

    finish = (end-start)

    B = np.transpose(A)
    start1 = time.time()
    B2 = np.matmul(A,B)
    end1 = time.time()

    finish2 = (end1-start1)

    print("\n\Execution time to compute AA^T= {0}".format(end1-start1))
    #print("\nAA^T= {0}".format(B2))

    #print("\nObServation= {0}".format(finish2 - finish))

   # B1_magnitude = np.sum(np.absolute(B1))
   # B2_magnitude = np.sum(np.absolute(B2))
    magnitude = np.sum(np.abs(B1-B2))
    print("|B1ij −B2ij|:")
    #print(B1_magnitude-B2_magnitude)
    print(magnitude)
compare(1000,100)
compare(1000,100)
compare(1000,1000)



data = np.load('/Users/oluwaseuncardoso/Downloads/npMnist.npy')
x = data[1]


#--------------Q10---------------

#------ a ------
I = np.identity(4)
perm = [3,2,0,1]
P = I[perm,:]
print("P ={0}\n".format(P))
print("PP^T ={0}".format(np.matmul(P,np.transpose(P))))

#------ b ------
I = np.identity(784)
perm = np.arange(784)
np.random.shuffle(perm)
P = I[perm,:]


p = np.reshape(x,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 10(b): an MNIST digit')
plt.show()

y = np.matmul(P,x)
p = np.reshape(y,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 10(b): permuted digit')
plt.show()


z = np.matmul(np.transpose(P),y)
p = np.reshape(z,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 10(b): restored digit')
plt.show()

#--------------Q10---------------

#------ a ------
A = np.random.rand(784,784)


y = np.matmul(A,x)

p = np.reshape(y,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 11(a)')
plt.show()

#------ b ------
A_inv = np.linalg.inv(A)
z = np.matmul(A_inv,y)
p = np.reshape(z,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 11(b)')
plt.show()

#------ c ------

y = np.matmul(A,x)

P,L,U = sp.lu(A, permute_l=False)

Py = np.matmul(y,P)

s = sp.solve_triangular(L,Py, lower = True)

z = sp.solve_triangular(U,s)

p = np.reshape(z,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 11(c)')
plt.show()

#------ d ------

B = np.matmul(A,np.transpose(A))

print(np.linalg.cond(B))

#------ e ------

C = np.matmul(B,np.transpose(B))

print(np.linalg.cond(C))

#------ f ------
y = np.matmul(C,x)
p = np.reshape(y,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 11(f)')
plt.show()

#------ g ------

C_inv = np.linalg.inv(C)
z = np.matmul(C_inv,y)
p = np.reshape(z,(28,28))
plt.imshow(p, interpolation='nearest', cmap='Greys')
plt.axis('off')
plt.title('Question 11(g)')
plt.show()


