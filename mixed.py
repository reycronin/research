from ufl import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import timeit

start = timeit.default_timer()


error_u = []
error_p = []
count = 0;
num = [8, 16, 32, 64, 128]

for i in range(len(num)):

    mesh = UnitSquareMesh(num[i], num[i])

    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # pressure
    p_exact= Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    # velocity
    u_exact = Expression(('-2*x[0]', '-4*x[1]'), degree = 1)
    f = Expression('-6', degree = 0)

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
    
    count = count + 1



plt.plot(num, error_u)
plt.xlabel('Mesh Courseness')
plt.show()


'''
plt.figure()
plot(u)

plt.figure()
plot(p)

plt.show()
'''
