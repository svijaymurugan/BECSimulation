SplitStep notebook: initial code

IndiumGPE: torch incorporated full GPE sim

PlotsIndiumGPE: separated into different classes, RTE vs ITE, visualization class as well

Parameters dict:

{"Np":1e5, 
"m":1.9e-25, 
"omega":2*pi*40, 
"a0": 2e-9, 
"a02": 2e-27, 
"gamma": 1, 
"sigma": 11e-6, 
"up": 5.5e-5,
"N": 100, 
"ev_time": 1e-2,
"time_steps": 100,
"cutoff": "Hard Cutoff",
"cutoff_coeff": 1/3, 
"waist": False,
"high_precision": False, 
"z_scale": 1, 
"y_scale": 1,
"time_plots": True, 
"final_plots": True, 
"plot_steps": 0.1, 
"Nz": None,
"imag_dtau": 5e-3, 
"imag_max_steps": 10000, 
"tolerance": 1e-6,
"init_type": "Gaussian",
"device": "cuda" if torch.cuda.is_available() else "cpu"}
