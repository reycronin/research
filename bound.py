from ufl import *
from dolfin import *
import matplotlib.pyplot as plt


#from ufl import *


mesh = UnitSquareMesh(32, 32)
order = 0

BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)


(u, p) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

# pressure
p_exact= Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
# velocity
u_exact = Expression(('2*x[0]', '4*x[1]'), degree = 2)
#f = Expression('4',degree = 2)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree = 2)


n = FacetNormal(mesh)

a = (dot(u, tau) + div(tau)*p + div(u)*v)*dx
g = Expression('4*x[1]', degree = 2)
L = -f*v*dx + (p_exact*dot(tau,n))*ds


#p_int = interpolate(p_exact, FunctionSpace(mesh,"DG",order))
#u_int = interpolate(u_exact, FunctionSpace(mesh,"BDM",order+1))

#assign(w.sub(0), p_int)
#assign(w.sub(1), u_int)

'''

class BoundarySource(Expression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        values[0] = 2*x[0]*n[0]
        values[1] = 4*x[1]*n[1]
        values[2] = 2*x[0]*n[0]
        values[3] = 4*x[1]*n[1]
    def value_shape(self):
        return (2, 2)

G = BoundarySource(mesh)

# Define essential boundary   
def boundary(x):
   return x[0] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS 
    
bc = DirichletBC(W.sub(0), G, boundary)

'''


class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = 0.0
        values[0] = g*n[0]
        values[1] = g*n[1]
        values[2] = g*n[0]
        values[3] = g*n[1]
    def value_shape(self):
        return (2,)
G = BoundarySource(mesh)
# Define essential boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
bc = DirichletBC(W.sub(0), G, boundary)

w = Function(W)



solve(a == L, w, bc)
#solve(a == L, w)
(u, p) = w.split()

error_L2 = errornorm(p_exact, p, 'L2')

print('error_L2  =', error_L2)


#plt.figure()
#plot(u)

#plt.figure()
#plot(p)

#plt.show()
