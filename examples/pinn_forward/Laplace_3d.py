##
## 3D Laplace equation \Delta u = -1 with zero Dirichlet BCs on [0,1]^3
##      use "hard constraint option" to get boudary condtions decently enforced
##

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
# Import tf if using backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
# Import torch if using backend pytorch
import torch
# Import paddle if using backend paddle
# import paddle

hard_constraint = True # require exact BCs
weights = 100  # if hard_constraint == False


def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_zz = dde.grad.hessian(u, x, i=2, j=2)
    return du_xx + du_yy + du_zz + 1


geom = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[1, 1, 1])

def boundary(_, on_boundary):
    return on_boundary

bc_zero = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(
    geom, pde, bc_zero, num_domain=2000, num_boundary=200
) # solution=None

# 20 pts per edge + 2500 pts per face = 15240 pts on boundary (!!!)
#  5 pts per edge +  400 pts per face = 2460 pts on boundary


net = dde.nn.FNN([3] + [20] * 3 + [1], "tanh", "Glorot normal")

def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2]) * x[:, 2:3] * (1 - x[:, 2:3])
    return res * y

if hard_constraint == True:
    net.apply_output_transform(transform)


model = dde.Model(data, net)

model.compile("adam", lr=1e-3)


if hard_constraint == True:
    print("==> compiling with hard boundary constraint")
    model.compile("adam", lr=1e-3)
else:
    print("==> compiling with loss-based boundary constraint")
    loss_weights = [1, weights]
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=loss_weights
    )


losshistory, train_state = model.train(iterations=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
