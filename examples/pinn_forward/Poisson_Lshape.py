"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde


def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0, component=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1, component=0)
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# model.train(iterations=50000)
# losshistory, train_state = model.train(iterations=100, model_save_path='/Users/gillette7/Desktop/projects/nfp4va/deepxde_nfp/tempnew')
# print("\nlosshistory=\n", losshistory)
# print("\ntrain_state=\n", train_state)
# model.compile("L-BFGS")
# losshistory, train_state = model.train()
# losshistory, train_state = model.train(iterations=100)
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
