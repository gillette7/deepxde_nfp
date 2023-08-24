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

############################### Settings ###############################

dde.config.set_default_float("float64")

# Load in settings
settings_path = "LDC_3D_Settings.json" # name of settings file
settings_file = open(settings_path)
settings      = json.load(settings_file)
settings_file.close()

torchseed = settings["Seed"]
torch.manual_seed(torchseed)

architecture        = settings["Architecture"]          # List containing number of nodes per layer, including input and output layers
domain              = settings["Domain"]                # dictionary of domains for "x", "y", and "z", each of which are stored as a list containing left and right interval bounds
epochsADAM          = settings["Num_Adam_Epochs"]       # num epochs to train with adam optimizer
epochsLBFGS         = settings["Num_LBFGS_Epochs"]      # num epochs to train with LBFGS optimizer, after using adam
chkpt_save_prefix   = settings["Checkpoint_name"]       # prefix to use for saving checkpoint file
lr                  = float(settings["Learning_rate"])  # learning rate for optimizer
num_domain_pts      = settings['num_domain_pts']
num_anchor_pts      = settings['num_anchor_pts'] 
num_test_pts        = settings['num_test_pts']  

# Reynold's number bounds
ReMin               = settings["Reynolds_min"]          # minimum Reynold's number to train on
ReMax               = settings["Reynolds_max"]          # maximum Reynold's number to train on
eps                 = settings["eps"]
ReViz               = settings["Reynolds_viz"] 


## Define PDE (Navier Stokes)
def pde(inputs, outputs): 
    '''
    inputs and outputs are both in R^4
    Inputs consist of (x,y,z) and ReNorm
    Outputs consist of velocity in 3 dimensions (u,v,w) and pressure (p)

    This function uses the first and second derivatives
    of each velocity component and pressure with respect to x,y,z
    to compute a loss which quantifies how well inputs/outputs satisfy the PDE.
    '''

    # Store velocity components
    u, v, w = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    # compute first derivatives
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    du_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    du_z = dde.grad.jacobian(outputs, inputs, i=0, j=2)

    dv_x = dde.grad.jacobian(outputs, inputs, i=1, j=0)
    dv_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    dv_z = dde.grad.jacobian(outputs, inputs, i=1, j=2)

    dw_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    dw_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)
    dw_z = dde.grad.jacobian(outputs, inputs, i=2, j=2)

    dp_x = dde.grad.jacobian(outputs, inputs, i=3, j=0)
    dp_y = dde.grad.jacobian(outputs, inputs, i=3, j=1)
    dp_z = dde.grad.jacobian(outputs, inputs, i=3, j=2)

    # compute second derivatives
    du_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    du_zz = dde.grad.hessian(outputs, inputs, component=0, i=2, j=2)

    dv_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    dv_zz = dde.grad.hessian(outputs, inputs, component=1, i=2, j=2)

    dw_xx = dde.grad.hessian(outputs, inputs, component=2, i=0, j=0)
    dw_yy = dde.grad.hessian(outputs, inputs, component=2, i=1, j=1)
    dw_zz = dde.grad.hessian(outputs, inputs, component=2, i=2, j=2)
   

    # determine Reynold's number
    # (note that ReNorm determines which num between ReMin and ReMax to use)
    ReNorm = inputs[:,3:4]
    Re = ReMin + (ReMax - ReMin) * ReNorm

    # Compute loss
    loss1 = ( u * du_x + v * du_y + w * du_z - (1/Re) * (du_xx + du_yy + du_zz) + dp_x )
    loss2 = ( u * dv_x + v * dv_y + w * dv_z - (1/Re) * (dv_xx + dv_yy + dv_zz) + dp_y )
    loss3 = ( u * dw_x + v * dw_y + w * dw_z - (1/Re) * (dw_xx + dw_yy + dw_zz) + dp_z )
    loss4 = (du_x + dv_y + dw_z)

    return loss1, loss2, loss3, loss4


def output_transform_cavity_flow_3DVP(inputs, outputs): 
    """
    Apply transform to solution to enforce oundary conditions
    velocity = 1 on lid and 0 on other boundaries

    inputs  = (x,y,z,p)
    outputs = NN(x,y,z,p) = (u,v,w,p) before transform
    inputs.shape = outputs.shape = (num_anchor_pts, 4)
    """

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
    # Define problem domain for (x,y,z, ReNorm)
    # note that we are hardcoding ReNorm to be betwen 0 and 1 because the Reynold's number range is actually determined by ReMin and ReMax
    geom = dde.geometry.Rectangle([domain["x"][0], domain["y"][0], domain["z"][0], 0], [domain["x"][1], domain["y"][1], domain["z"][1], 1])

    # Choose random anchor points
    anchor_pts = geom.random_points(num_anchor_pts)

    # we assume that the network always uses tanh activation function and Glorot normal initialization, so this is not included in the settings file
    NN = dde.maps.FNN(architecture, "tanh", "Glorot normal")
    if torch.cuda.is_available():
        NN.cuda() 
    NN.apply_output_transform(output_transform_cavity_flow_3DVP)

    losses = []

    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain   = num_domain_pts,
        num_boundary = 0,
        num_test     = num_test_pts, 
        anchors      = anchor_pts
    )

    model = dde.Model(data, NN)


    loss_weights = [1] * 4
    loss = ["MSE"] * 4

    dde.optimizers.set_LBFGS_options(
            maxcor  = 150,
            ftol    = 1.0 * np.finfo(float).eps,
            gtol    = 1.0 * np.finfo(float).eps,
            maxiter = epochsLBFGS,
            maxfun  = epochsLBFGS,
            maxls   = 200
            )

    print("\n==> Set these LBFGS options: ", dde.optimizers.LBFGS_options, "\n")
    print("\n==> Compiling with adam...")
    model.compile("adam", lr = lr, loss = loss, loss_weights = loss_weights)

    # train model using adam
    loss_history, train_state = model.train(epochs = epochsADAM)

    print("\n==> Compiling with L-BFGS-B...")
    model.compile("L-BFGS-B", loss=loss, loss_weights = loss_weights)
    print("\n==> Training...")

    # train model using LBFGS
    loss_history, train_state = model.train(model_save_path = chkpt_save_prefix)
 
    # save_solution(geom, model, "./solution0")
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
