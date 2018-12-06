from ufl import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import timeit

start = timeit.default_timer()


error_u = []
error_p = []
cost = []
num = [8, 16, 32, 64, 128]

for i in range(len(num)):

    mesh = UnitSquareMesh(num[i], num[i])
    # pressure
    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    # velocity
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # pressure
    #p_exact = Expression('pi*sin(2*pi*x[0])*sin(pi*x[1]*x[1])*sin(pi*x[1]*x[1])', degree = 2)
    p_exact= Expression('1 + x[0]*x[0]*x[0] + 2*x[1]*x[1]', degree=3)
    # velocity
    #u_exact = Expression('2*pi*pi*cos(2*pi*x[0])*sin(pi*x[1]*x[1])*sin(pi*x[1]*x[1])', '2*pi*pi*x[1]*sin(pi*x[1]*x[1])', degree = 1)
    
    u_exact = Expression(('-3*x[0]*x[0]', '-4*x[1]'), degree = 2)
    f = Expression('-6*x[0] - 4', degree = 1)

    n = FacetNormal(mesh)
    a = (dot(u, v) - div(v)*p - div(u)*q)*dx
    L = -f*q*dx - (p_exact*dot(v,n))*ds

    u0 = interpolate(u_exact, W.sub(0).collapse())
    p0 = interpolate(p_exact, W.sub(1).collapse())

    w = Function(W)
    assign(w, [u0, p0])

    solve(a == L, w)
    (u, p) = w.split()

    error_L2_p = errornorm(p_exact, p, 'L2', degree_rise = 0)
    error_L2_u = errornorm(u_exact, u, 'L2')

    stop = timeit.default_timer()

    print('grid size', i)
    print('error_L2_velocity =', error_L2_u)
    print('error_L2_pressure =', error_L2_p)
    print('Time: ', stop - start)

    error_u.append(error_L2_u)
    error_p.append(error_L2_p)
    cost.append(stop - start)    

h = [ 1/x for x in num]
h2 = [ x ** 2 for x in h]


plt.figure(1)
plt.loglog(h, error_u, label = 'velocity')
plt.loglog(h, error_p, label = 'pressure')
plt.loglog(h, h2, label = 'O(h^2)')
plt.xlabel('h')
plt.ylabel('L2 error')
plt.legend(loc = 'upper right')
plt.show()


plt.figure(2)
plt.loglog(error_p, cost, label = 'e_p')
plt.loglog(error_u, cost, label = 'e_u')
plt.xlabel('L2 error')
plt.ylabel('Cost')
plt.legend(loc = 'lower left')
plt.show()

'''
plt.figure()
plot(u)

plt.figure()
plot(p)

plt.show()
'''
