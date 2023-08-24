'''
Load and test lid-driven cavity PINN in 3D.
'''

import json
import deepxde as dde
import numpy as np
from deepxde.backend import torch
from LDC_3D import output_transform_cavity_flow_3DVP, pde

dde.config.set_default_float("float64")

# file paths
settings_path   = "LDC_3D_Settings.json" # name of settings file
checkpoint_path = "model_ldc_3d.ckpt-150000.pt"

def eval_PINN(model, data):
	''' 
	Deterimine loss for current model state
	'''

	# find test data
	test_input = data.test_x
	print("test_input info: {}".format(test_input.shape))

	# use predict on model with test data
	test_output = model.predict(test_input)

	# compute loss from output
	losses = pde(test_input, test_output)

	print("Losses:")
	print(losses)
	
	return losses

# Load in settings
settings_file = open(settings_path)
settings      = json.load(settings_file)
settings_file.close()

architecture        = settings["Architecture"]          # List containing number of nodes per layer, including input and output layers
domain              = settings["Domain"]                # dictionary of domains for "x", "y", and "z", each of which are stored as a list containing left and right interval bounds
chkpt_save_prefix   = settings["Checkpoint_name"]       # prefix to use for saving checkpoint file

num_anchor_pts      = settings['num_anchor_pts'] 
num_domain_pts      = settings['num_domain_pts']
num_test_pts        = settings['num_test_pts']  

# Reynold's number bounds
ReMin               = settings["Reynolds_min"]          # minimum Reynold's number to train on
ReMax               = settings["Reynolds_max"]          # maximum Reynold's number to train on
eps                 = settings["eps"]
ReViz               = settings["Reynolds_viz"] 

def main():
	# Set up NN and model
	NN = dde.maps.FNN(architecture, "tanh", "Glorot normal")
	if torch.cuda.is_available():
		NN.cuda() 
	NN.apply_output_transform(output_transform_cavity_flow_3DVP)

	# Define problem domain for (x,y,z, ReNorm)
	# note that we are hardcoding ReNorm to be betwen 0 and 1 because the Reynold's number range is actually determined by ReMin and ReMax
	geom = dde.geometry.Rectangle([domain["x"][0], domain["y"][0], domain["z"][0], 0], [domain["x"][1], domain["y"][1], domain["z"][1], 1])
	losses = []

	# Choose random anchor points
	anchor_pts = geom.random_points(num_anchor_pts)


	data = dde.data.PDE(
	        geom,
	        pde,
	        losses,
	        num_domain   = 0,
	        num_boundary = 0,
	        num_test     = num_test_pts, 
	        anchors      = anchor_pts
	    )


	# Load weights to NN
	if torch.cuda.is_available():
		checkpoint 	= torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

	NN.load_state_dict(checkpoint['model_state_dict'])
	model = dde.Model(data, NN)
	print(type(model))
	#model.restore(checkpoint_path)
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	#epoch 		= checkpoint['epoch']
	#loss 		= checkpoint['loss']

	print("evalutating network")

	eval_PINN(model, data)

if __name__ == "__main__":
    main()
