from ufl import *
from dolfin import *
import matplotlib.pyplot as plt


#from ufl import *

t = 0
mesh = UnitSquareMesh(64, 64)
order = 0

BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)


(u, p) = TrialFunctions(W)
(tau, v) = TestFunctions(W)


f = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t, degree = 2)

# exact solution
# Pressure
p_exact = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t, degree = 3)
# Velocity
u_exact = Expression(('-cos(x[0])*cos(t)','sin(x[1])*cos(t)'), t=t, degree = 2)




n = FacetNormal(mesh)

a = (dot(u, tau) + div(tau)*p + div(u)*v)*dx
L = -f*v*dx + (p_exact*dot(tau,n))*ds

u0 = interpolate(u_exact, W.sub(0).collapse())
p0 = interpolate(p_exact, W.sub(1).collapse())

w = Function(W)
assign(w, [u0, p0])

solve(a == L, w)
#solve(a == L, w)
(u, p) = w.split()

error_L2_p = errornorm(p_exact, p, 'L2')


error_L2_u = errornorm(u_exact, u, 'L2')


print('error_L2_velocity  =', error_L2_u)

print('error_L2_p  =', error_L2_p)


#plt.figure()
#plot(u)

#plt.figure()
#plot(p)

#plt.show()
