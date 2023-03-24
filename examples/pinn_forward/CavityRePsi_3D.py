##
## 3D lid-driven cavity PINN
##  Andrew Gillette, Matthew Berger, Josh Levine
##  based on 2D script by
##  Christopher McDevitt <cmcdevitt@ufl.edu> and colleagues
## 

# This file needs to make a json that does this:

	# {"model": "deepxde", "filename": "net_dde.pt", "pinn_driver_file": "examples.pinn_forward.CavityRePsi_3D", "net_arch": "fnn_4_64_6_4_tanh_glorot"}


import deepxde as dde
import numpy as np
from deepxde.backend import torch

dde.config.set_default_float("float64")

# print("\n *** Original script uses tensor flow ***\n")
torch.manual_seed(1234)

# epochsADAM = 10000
# epochsLBFGS = 50000
epochsADAM = 10
epochsLBFGS = 50

lr = 5.e-4
interiorpts = 2000 # 50000
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


def output_transform_cavity_flow_3DVP(inputs, outputs): # inputs  = (x,y,z,p)
                                                   # outputs = (u,v,w,p) = net(x,y,z,p)
                                                   # inputs.shape = outputs.shape = (interiorpts, 4)
    

    # x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    x, y, z = inputs[..., 0:1], inputs[..., 1:2], inputs[..., 2:3]

    # print("x shape", x.shape)
    # print("y shape", y.shape)
    # print("z shape", z.shape)
    
    Phix = (1-torch.exp(-(x-1)*(x-1)/eps)) * (1-torch.exp(-x*x/eps))
    Phiy = (1-torch.exp(-(y-1)*(y-1)/eps)) * (1-torch.exp(-y*y/eps))
    blid = torch.sin((3*np.pi/2)*(z - (2/3))) * Phix * Phiy
    
    b0 = 64 * x * (1 - x) * y * (1 - y) * z * (1-z)

    u = blid + b0 * outputs[..., :1] 
    v = b0 * outputs[..., 1:2] 
    w = b0 * outputs[..., 2:3] 
    p = outputs[..., 3:4]
    # if p.shape[1] == 0: # if no pressure data was provided, which happens when doing viz
    #     p = torch.zeros_like(u)
    # print("\nu shape", u.shape)
    # print("v shape", v.shape)
    # print("w shape", w.shape)
    # print("p shape", p.shape)
    # print("return shape:",torch.cat((u, v, w, p), axis=-1).shape,"\n" )

    return torch.cat((u, v, w, p), axis=-1)


def main():
    geom = dde.geometry.Rectangle([0, 0, 0, 0], [1, 1, 1, 1])
    points = geom.random_points(interiorpts)

    net = dde.maps.FNN([4] + [64] * 6 + [4], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform_cavity_flow_3DVP)

    losses = []

    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=0,
        num_boundary=0,
        num_test=200,   #2**15,
        anchors = points
    )

    model = dde.Model(data, net)


    loss_weights = [1] * 3
    loss = ["MSE"] * 3

    dde.optimizers.set_LBFGS_options(
            maxcor=150,
            ftol=1.0 * np.finfo(float).eps,
            gtol=1.0 * np.finfo(float).eps,
            maxiter=epochsLBFGS,
            maxfun= epochsLBFGS,
            maxls=200
            )
    print("\n==> Set these LBFGS options: ", dde.optimizers.LBFGS_options, "\n")
    print("\n==> Compling with adam...")
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)
    losshistory, train_state = model.train(epochs=epochsADAM)
    print("\n==> Compling with L-BFGS-B...")
    model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
    print("\n==> Training...")
    losshistory, train_state = model.train(model_save_path="./model_ldc_3d.ckpt")
    save_solution(geom, model, "./solution0")

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()


# need to save: net, output_transform, 

## syntax from 2D version, using tensorflow
#
# Note: _train_step(...) in model.py doesn't have options for aux vbls
# model.train_step.optimizer_kwargs = {'options': {'maxcor': 150,
                                                #  'ftol': 1.0 * np.finfo(float).eps,
                                                #  'gtol': 1.0 * np.finfo(float).eps,
                                                #  'maxiter': epochsLBFGS,
                                                #  'maxfun':  epochsLBFGS,
                                                #  'maxls': 200}}