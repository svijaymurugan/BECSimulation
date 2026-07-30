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
"modulation freq": 2.0, #unitless freq
"modulation amplitude": 0.1, #10% of potential
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

Np: Number of particles
m: mass
a0: scattering length
a02: scattering volume
gamma: haarmoinc trap potential frequency scaling in z-direction (V(x,y,z) = x^2 + y^2 + gamma^2 * z^2)
sigma: standard deviation of Gaussian initial state
up: half box length
modulation freq: frequency at which potential trap frequency is modulated: w(t) = w_0(1+Asin(w_dt)) (w_d is modulation freq)
modulation amp: amplitude at which potential trap frequency is modulated: (A is modulation amp)
N: Number of gridpoints along one dimension (so N^3 total gridpoitns)
ev_time: Evolution Time
time_steps: Number of simulation time steps
cutoff: Type of cutoff: Hard cutoff, Soft cutoff (sigmoid), Cylindrical (2 hard cutoffs: one in xy-plane mangitude and one in z-direction), Manual (insert your own code)
waist: Boolean to track waist value
high_precision: False -> complex64, float32; True -> complex128, float64
z_scale: scale standard deviation of Gaussian in z-direction
y_scale: scale standard deviation of Gaussian in y-direction
time_plots: Boolean to save intermdiate plots through simulation (time-dependent plots: XY/XZ densities, X/Y/Z slices)
final_plots: Boolean to save summary plots: Energy vs time, initial densities
plot_steps: frequency to save time plots
Nz: gridpoints in z-direction (taken to be N if this None)
imag_dtau: unitless time step for imaginary time evolution
imag_max_steps: maximum number of time steps, at which point ITE stops even if not convergent
tolerance: threshold that must be met for ITE to be considered as convergent
init_type: specify type of initial state: Gaussian or Ground State 
device: GPU vs CPU