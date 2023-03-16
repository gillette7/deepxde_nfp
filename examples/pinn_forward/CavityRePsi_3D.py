##
## 3D lid-driven cavity PINN
##  Andrew Gillette, Matthew Berger, Josh Levine
##  based on 2D script by
##  Christopher McDevitt <cmcdevitt@ufl.edu> and colleagues
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

eps = np.sqrt(1.e-3)

def save_solution(geom, model, filename):
    x = geom.uniform_points(80**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred)))

    x = geom.uniform_points(20**3)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_coarse.dat", np.hstack((x, y_pred)))



def pde(inputs, outputs): # ((x,y,z,ReNorm), (u,v,w,p))
    u, v, w = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    du_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    du_z = dde.grad.jacobian(outputs, inputs, i=0, j=2)
    dv_x = dde.grad.jacobian(outputs, inputs, i=1, j=0)
    dv_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    dv_z = dde.grad.jacobian(outputs, inputs, i=1, j=2)
    dw_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    dw_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)
    dw_z = dde.grad.jacobian(outputs, inputs, i=2, j=2)
    du_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    du_zz = dde.grad.hessian(outputs, inputs, component=0, i=2, j=2)
    dv_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    dv_zz = dde.grad.hessian(outputs, inputs, component=1, i=2, j=2)
    dw_xx = dde.grad.hessian(outputs, inputs, component=2, i=0, j=0)
    dw_yy = dde.grad.hessian(outputs, inputs, component=2, i=1, j=1)
    dw_zz = dde.grad.hessian(outputs, inputs, component=2, i=2, j=2)
    dp_x = dde.grad.jacobian(outputs, inputs, i=3, j=0)
    dp_y = dde.grad.jacobian(outputs, inputs, i=3, j=1)
    dp_z = dde.grad.jacobian(outputs, inputs, i=3, j=2)

    ReNorm = inputs[:,3:4]

    Re = ReMin + (ReMax-ReMin)*ReNorm

    loss1 = ( u*du_x + v*du_y + w*du_z - (1/Re)*(du_xx + du_yy + du_zz) + dp_x )
    loss2 = ( u*dv_x + v*dv_y + w*dv_z - (1/Re)*(dv_xx + dv_yy + dv_zz) + dp_y )
    loss3 = ( u*dw_x + v*dw_y + w*dw_z - (1/Re)*(dw_xx + dw_yy + dw_zz) + dp_z )
    #loss4 = (du_x + dv_y + dw_z)

    return loss1, loss2, loss3


def output_transform_cavity_flow(inputs, outputs): # inputs = (x,y,z,p); outputs = net(x) (I think)
                                                   # inputs shape = (interiorpts, 4); outputs shape = (interiorpts, 4)
    x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    
    Phix = (1-torch.exp(-(x-1)*(x-1)/eps)) * (1-torch.exp(-x*x/eps))
    Phiy = (1-torch.exp(-(y-1)*(y-1)/eps)) * (1-torch.exp(-y*y/eps))
    blid = torch.sin((3*np.pi/2)*(z - (2/3))) * Phix * Phiy
    
    b0 = 64 * x * (1 - x) * y * (1 - y) * z * (1-z)

    u = blid + b0 * outputs[:, :1] # indexing for outputs???
    v = b0 * outputs[:, :1] # indexing for outputs???
    w = b0 * outputs[:, :1] # indexing for outputs???
    p = outputs[:, 3:4]

    return torch.cat((u, v, w, p), axis=1)

    # bcv = 16 * x * (1 - x) * y * (1 - y) # boundary condition for v, sort of;
    #                                      # taking some algebraic shortcuts to simplify the code

    # ExpB = torch.exp(-(1-y)**2/dy**2)

    # psilid = (y-1)*y**2 * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * (1-torch.exp(-x*x/dx**2))*ExpB

    # psilid_x = (y-1)*y**2 * ( 2*(x-1)/dx**2*torch.exp(-(x-1)*(x-1)/dx**2) ) * (1-torch.exp(-x*x/dx**2))*ExpB + (y-1)*y**2 * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * ( 2*x/dx**2*torch.exp(-x*x/dx**2) )*ExpB
    # psilid_y = ( y**2 + 2*y*(y-1) ) * (1-torch.exp(-(x-1)*(x-1)/dx**2)) * (1-torch.exp(-x*x/dx**2))*ExpB + psilid*2*(1-y)/dy**2

    # dbcv_x = 16 * ( 1-2*x ) * y * (1 - y)  # partial deriv of BC for v w/r/t x
    # dbcv_y = 16 * x * (1 - x) * ( 1 - 2*y)

    # psiprime_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    # psiprime_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    

    # # u
    # u = psilid_y + 2*bcv*dbcv_y * outputs[:, :1] + bcv**2*psiprime_y
    # # v
    # v = -(psilid_x + 2*bcv*dbcv_x * outputs[:, :1] + bcv**2*psiprime_x)
    # # p
    # p = outputs[:, 2:3]
    #
    # return torch.cat((u, v, p), axis=1)


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
