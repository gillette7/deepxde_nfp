##
## 3D lid-driven cavity PINN
##  Matthew Berger, Andrew Gillette, Josh Levine
##  based on 2D script by
##  Christopher McDevitt <cmcdevitt@ufl.edu> and colleagues
## 

import os
import json
import deepxde as dde
import numpy as np
from deepxde.backend import torch

dde.config.set_default_float("float64")

# print("\n *** Original script uses tensor flow ***\n")
torchseed = 1234
torch.manual_seed(torchseed)

epochsADAM = 10000
epochsLBFGS = 50000
# # epochsADAM = 20000
# # epochsLBFGS = 100000

chkpt_save_prefix = 'net_ldc_3d'
lr = 5.e-4
num_anchor_pts = 50000 # 100
num_test_pts = 2**15  # 2**10
ReMin = 900
ReMax = 1100 # 1000
eps = np.sqrt(1.e-3)
ReViz = 0.5

ldc_config = {
    'model': 'deepxde', 
    'chkpt_prefix' : chkpt_save_prefix,
    'pinn_driver_file': 'examples.pinn_forward.CavityRePsi_3D',
    'net_arch' : 'fnn_4_64_6_4_tanh_glorot',
    'epochsADAM' : epochsADAM,
    'epochsLBFGS' : epochsLBFGS,
    'lr' : lr,
    'num_anchor_pts' : num_anchor_pts,
    'num_test_pts' : num_test_pts,
    'ReMin' : ReMin,
    'ReMax' : ReMax,
    'eps' : eps,
    'seed' : torchseed,
    'ReViz' : ReViz
}

json_save_dir = os.getcwd()
with open(json_save_dir+"/ldc_config.json", 'w') as f: 
    json.dump(ldc_config,f)

# def save_solution(geom, model, filename):
#     x = geom.uniform_points(27**4)
#     y_pred = model.predict(x)
#     print("Saving u and p ...\n")
#     np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred)))

#     x = geom.uniform_points(10**4)
#     y_pred = model.predict(x)
#     print("Saving u and p ...\n")
#     np.savetxt(filename + "_coarse.dat", np.hstack((x, y_pred)))



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
    loss4 = (du_x + dv_y + dw_z)

    return loss1, loss2, loss3, loss4


def output_transform_cavity_flow_3DVP(inputs, outputs): # inputs  = (x,y,z,p)
                                                   # outputs = net(x,y,z,p) = (u,v,w,p) before transform
                                                   # inputs.shape = outputs.shape = (num_anchor_pts, 4)
    

    # x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    x, y, z = inputs[..., 0:1], inputs[..., 1:2], inputs[..., 2:3]
    
    Phix = (1-torch.exp(-(x-1)*(x-1)/eps)) * (1-torch.exp(-x*x/eps))
    Phiy = (1-torch.exp(-(y-1)*(y-1)/eps)) * (1-torch.exp(-y*y/eps))
    blid = torch.sin((3*np.pi/2)*(z - (2/3))) * Phix * Phiy
    
    b0 = 64 * x * (1 - x) * y * (1 - y) * z * (1-z)

    u = blid + b0 * outputs[..., :1] 
    v = b0 * outputs[..., 1:2] 
    w = b0 * outputs[..., 2:3] 
    p = outputs[..., 3:4]

    return torch.cat((u, v, w, p), axis=-1)


def main():
    geom = dde.geometry.Rectangle([0, 0, 0, 0], [1, 1, 1, 1])
    anchor_pts = geom.random_points(num_anchor_pts)

    net = dde.maps.FNN([4] + [64] * 6 + [4], "tanh", "Glorot normal")
    net.cuda()
    net.apply_output_transform(output_transform_cavity_flow_3DVP)

    losses = []

    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=0,
        num_boundary=0,
        num_test=num_test_pts, 
        anchors = anchor_pts
    )

    model = dde.Model(data, net)


    loss_weights = [1] * 4
    loss = ["MSE"] * 4

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

    losshistory, train_state = model.train(model_save_path=chkpt_save_prefix)
 
    # save_solution(geom, model, "./solution0")
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
