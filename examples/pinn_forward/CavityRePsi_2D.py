##
## Original 2D lid-driven cavity script by 
##  Christopher McDevitt <cmcdevitt@ufl.edu>
##  received March 2023
## 

import deepxde as dde
import numpy as np
from deepxde.backend import torch

dde.config.set_default_float("float64")

print("\n *** Original script sets random seed here (and uses tensor flow) ***\n")
import time
time.sleep(1)

# tf.random.set_random_seed(1234)

# epochsADAM = 10000
# epochsLBFGS = 50000
epochsADAM = 100
epochsLBFGS = 500

lr = 5.e-4
interiorpts = 50000
ReMin = 100
ReMax = 1000

dx = np.sqrt(1.e-3)
dy = np.sqrt(1.e-2)

def save_solution(geom, model, filename):
    x = geom.uniform_points(80**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred)))

    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_coarse.dat", np.hstack((x, y_pred)))



def pde(inputs, outputs): # u, v, p
    u, v = outputs[:, 0:1], outputs[:, 1:2]
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    du_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dv_x = dde.grad.jacobian(outputs, inputs, i=1, j=0)
    dv_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    du_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    dp_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    dp_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)

    ReNorm = inputs[:,2:3]

    Re = ReMin + (ReMax-ReMin)*ReNorm

    loss1 = (u*du_x + v*du_y - (1/Re)*(du_xx + du_yy) + dp_x )
    loss2 = (u*dv_x + v*dv_y - (1/Re)*(dv_xx + dv_yy) + dp_y )
    #loss3 = (du_x + dv_y)

    return loss1, loss2


def output_transform_cavity_flow(inputs, outputs):
    x, y = inputs[:, 0:1], inputs[:, 1:2]

    bcv = 16 * x * (1 - x) * y * (1 - y)

    ExpB = torch.exp(-(1-y)**2/dy**2)

    psilid = (y-1)*y**2 * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * (1-torch.exp(-x*x/dx**2))*ExpB

    psilid_x = (y-1)*y**2 * ( 2*(x-1)/dx**2*torch.exp(-(x-1)*(x-1)/dx**2) ) * (1-torch.exp(-x*x/dx**2))*ExpB + (y-1)*y**2 * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * ( 2*x/dx**2*torch.exp(-x*x/dx**2) )*ExpB
    psilid_y = ( y**2 + 2*y*(y-1) ) * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * (1-torch.exp(-x*x/dx**2))*ExpB + psilid*2*(1-y)/dy**2

    dbcv_x = 16 * ( 1-2*x ) * y * (1 - y)
    dbcv_y = 16 * x * (1 - x) * ( 1 - 2*y)

    psiprime_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    psiprime_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    

    # u
    u = psilid_y + 2*bcv*dbcv_y * outputs[:, :1] + bcv**2*psiprime_y
    # v
    v = -(psilid_x + 2*bcv*dbcv_x * outputs[:, :1] + bcv**2*psiprime_x)
    # p
    p = outputs[:, 2:3]


    return torch.cat((u, v, p), axis=1)


def main():
    geom = dde.geometry.Rectangle([0, 0, 0], [1, 1, 1])
    uniform_points = geom.random_points(interiorpts)
    points = uniform_points

    net = dde.maps.FNN([3] + [64] * 6 + [3], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform_cavity_flow)

    losses = []


    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=0,
        num_boundary=0,
        num_test=2**15,
        anchors = points
    )

    model = dde.Model(data, net)


    loss_weights = [1] * 2
    loss = ["MSE"] * 2
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=epochsADAM)

    model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)


    model.train_step.optimizer_kwargs = {'options': {'maxcor': 150,
                                                     'ftol': 1.0 * np.finfo(float).eps,
                                                     'gtol': 1.0 * np.finfo(float).eps,
                                                     'maxiter': epochsLBFGS,
                                                     'maxfun':  epochsLBFGS,
                                                     'maxls': 200}}


    losshistory, train_state = model.train(model_save_path="./model.ckpt")
    save_solution(geom, model, "./solution0")

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
