# import numpy as np
#
# class Linear_eq():
#
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.sum_x = np.sum(x)
#         self.sum_y = np.sum(y)
#         self.sum_square_x = np.sum(x**2)
#         self.sum_square_y = np.sum(y**2)
#         self.sum_xy = np.sum(x*y)
#         self.a1 = np.array([self.sum_xy, self.sum_y])
#         self.a2 = np.array([[self.sum_square_x, self.sum_x],[self.sum_x, 100]])
#         self.A1 = np.mat(self.a1)
#         self.A2 = np.mat(self.a2)
#
#     def loss(self,a,b):
#         loss = self.sum_square_y+((a**2)*self.sum_square_x)-(2*a*self.sum_xy)+(2*a*b*self.sum_x)+(2*b*self.sum_y)+100*(b**2)
#         return loss
#
#     def result(self):
#         A2_I = self.A2.I
#         print(self.A1 * self.A2.I)
#
#
#
# x,y = np.loadtxt("/home/skylu/Downloads/data.txt")
# a = np.random.rand()
# b = np.random.rand()
# linear = Linear_eq(x,y)
# lr_rate = 1e-6
# deri_a = linear.sum_square_x*a+linear.sum_x-linear.sum_xy
# deri_b = linear.sum_x*a + 100*b - linear.sum_y
# iterate = 1000
# print("deri_A:"+str(deri_a))
# loss = linear.loss(a,b)
# for i in range(iterate):
#     a = a-lr_rate*deri_a
#     b = b-lr_rate*deri_b
#     loss = linear.loss(a,b)
#     print(loss)
# print("final loss is:"+str(loss))
# print("a:"+str(a))
# print("b:"+str(b))
import numpy as np

class Linear_eq():

    def __init__(self, x, y):
        self.x = x
        self.A = np.array([np.ones(100),self.x]).T
        self.B = y

    def loss(self,X):
        mtra = np.matmul(self.A, X) - self.B
        loss = 0.5*np.matmul(mtra.T, mtra)
        return loss

    def Gradient_de(self,X):
        gradient = np.matmul(self.A.T, (np.matmul(self.A, X)-self.B))
        n = np.linalg.norm(gradient)
        return gradient,n

    def interating(self,X,lr_rate):
        gradient,n = self.Gradient_de(X)
        X = X - lr_rate * gradient
        return X




x,y = np.loadtxt("/home/skylu/Downloads/data.txt")
a = np.random.rand()
b = np.random.rand()
X = np.array([a,b]).T
lr_rate = 1e-6
iteration = 0
linear = Linear_eq(x, y)

while(True):
    gradient, n = linear.Gradient_de(X)
    print(n)
    if (n<1e-4):
        break
    iteration = iteration+1
    X = linear.interating(X, lr_rate)
print(X)
print(linear.loss(X))
print(iteration)
