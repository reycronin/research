from ufl import *
from dolfin import *



def gravity(u):

    Ra = 1.0
    val = as_vector([0.0, Ra*u])
    return val

nx = 32
order = 0
t = 0
dt = 0.1
T = 10

mesh = UnitSquareMesh(nx,nx)

# Pressure
DG0 = FiniteElement("DG", mesh.ufl_cell(), order)
#DGO = FunctionSpace(mesh,"DG",order)

# Velocity
BDM = FiniteElement("BDM", mesh.ufl_cell(),order+1)
#BDM = FunctionSpace(mesh, "BDM", order+1)

# Temperature
#DG1 = FiniteElement("DG", mesh.ufl_cell(), order+1)
#DG1 = FunctionSpace(mesh, "DG", order+1)

W = FunctionSpace(mesh, DG0*BDM)

# source terms

f_1 = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t, degree = 2)
f_2 = Expression(('0.0', '(-sin(x[0]) - sin(x[1]))*cos(t)'), t=t, degree = 2)
f_3 = Expression('-(sin(x[0]) + sin(x[1]))*sin(t) + (-cos(x[0])*cos(x[0]) +sin(x[1])*cos(x[1]))*cos(t)*cos(t) + (sin(x[0]) + sin(x[1]))*cos(t)', t=t, degree = 2)

# exact solution

# Pressure
p_exact = Expression('(sin(x[0]) + cos(x[1]))*cos(t)', t=t, degree = 3)
# Velocity
u_exact = Expression(('-cos(x[0])*cos(t)','sin(x[1])*cos(t)'), t=t, degree = 2)

w = Function(W)
#w0 = Function(W)
w0, w1 = split(w)



#assigner = FunctionAssigner(W.sub(0), FunctionSpace(mesh,"DG",order))
#assigner.assign(p_exact, w)

p_int = interpolate(p_exact, FunctionSpace(mesh,"DG",order))
u_int = interpolate(u_exact, FunctionSpace(mesh,"BDM",order+1))


assign(w.sub(0), p_int)
assign(w.sub(1), u_int)


(p, u) = split(w)
#(p0,u0) = split(w)
(tau, v) = TestFunctions(W)
n = FacetNormal(mesh)

a = (dot(u, tau) + div(tau)*p + div(u)*v)*dx
L = inner(nabla_div(u),tau)*dx - f_1*tau*dx

solve(a == L, w)

error_L2 = errornorm(u_exact, u, 'L2')

print('error_L2  =', error_L2)

print('flag')
