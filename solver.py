from ufl import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import timeit

#T = 5.0            # final time
#num_steps = 500    # number of time steps
#dt = T / num_steps # time step size
abs_tol = 1E-7
rel_tol = 1E-4
max_iter = 1000



start = timeit.default_timer()

mesh = UnitSquareMesh(64, 64)

BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
'''
# pressure
p_exact= Expression('1 + x[0]*x[0]*x[0] + 2*x[1]*x[1]', degree=3)
# velocity
u_exact = Expression(('-3*x[0]*x[0]', '-4*x[1]'), degree = 2)
f = Expression('-6*x[0] - 4', degree = 1)
'''
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

A = assemble(a)
b = assemble(L)



# solver
#PETScOptions.set('ksp_view')
#PETScOptions.set('ksp_monitor_true_residual')
#PETScOptions.set('pc_type', 'fieldsplit')
#PETScOptions.set('pc_fieldsplit_type', 'additive')
#PETScOptions.set('pc_fieldsplit_detect_saddle_point')
#PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_0_pc_type', 'lu')
#PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
#PETScOptions.set('fieldsplit_1_pc_type', 'jacobi')

solver = PETScKrylovSolver('gmres')
solver.set_operator(A)
solver.set_from_options()

solver.parameters['absolute_tolerance'] = abs_tol
solver.parameters['relative_tolerance'] = rel_tol
solver.parameters['maximum_iterations'] = max_iter

# Solve
solver.solve(w.vector(), b)



(u, p) = w.split()

error_L2_p = errornorm(p_exact, p, 'L2', degree_rise = 0)
error_L2_u = errornorm(u_exact, u, 'L2')




stop = timeit.default_timer()

print('grid size', i)
print('error_L2_velocity =', error_L2_u)
print('error_L2_pressure =', error_L2_p)
print('Time: ', stop - start)


'''
plt.figure()
plot(u)

plt.figure()
plot(p)

plt.show()
'''
