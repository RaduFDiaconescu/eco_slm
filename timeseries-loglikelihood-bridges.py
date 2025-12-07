import numpy as np
from numba import njit, prange
from scipy import optimize
from scipy.special import iv, ive
import time


# everything below a @njit must be compatible with numba

@njit
def random_numbers_for_bridges(NB, cols, seed_value):
    np.random.seed(seed_value)
    return np.random.random(size=(NB, cols))


@njit
def BDM_Lamperti(ts_data):
    arr = ts_data
    # Check for negative values
    if np.any(arr < 0):
        raise ValueError("Input array contains negative values. Square root is not defined for negative numbers.")
    result = 2 * np.sqrt(arr)
    return result

@njit
def SLM_Lamperti(ts_data):
    arr = ts_data
    # Check for negative values
    if np.any(arr < 0):
        raise ValueError("Input array contains negative values. Log is not defined for negative numbers.")
    result = np.log(arr)
    return result



@njit
def bridge(NB, bridge_steps, x_i, x_f, Delta_t_B, D, seed_value):

    random_numbers = random_numbers_for_bridges(NB=NB, cols=bridge_steps-2, seed_value=seed_value)

    B = np.empty((NB, bridge_steps), dtype=np.float32) #to store simulated X values for bridges
    B[:, 0] = x_i
    B[:, -1] = x_f
    for step in range(1, bridge_steps - 1):
        #ATTENTION: Manal's script had (bridge_steps - step) * B[:, step - 1] + x_f with a PLUS!! Is this a grave typo in Javier's?

        # Eq. G4 (Wiener)
        # (t - s) = 1 *\Delta t_{B}. Cancels out
        mean_b = ((bridge_steps - step) * B[:, step - 1] + x_f ) / (bridge_steps - step + 1)
        # Eq. G5 (Wiener)
        var_b = (D**2 * (bridge_steps - step) * Delta_t_B ) / (bridge_steps - step + 1)
        B[:, step] = random_numbers[:, step - 1] * np.sqrt(var_b) + mean_b
    return B

@njit
def I_hat(k, mu, B, Delta_t_B, model='BDM'):

    valid_models = {"BDM", "SLM"}

    # Eq. H5
    # \hat{I} = \Delta t^{(B)} \sum_{j} \tilde{A}^{2}(x^{(B)}) \Delta t_{B}

    # Eq. E3 defined in \tilde{A}(x)

    #Eq. C22: Replace f(x_{t}) term with \tilde{A}(x)
    #Discrete approximation of second intergral using Eq. H5

    # B = np.array(B)

    if B.ndim != 1:
        raise ValueError(f"Expected 1D array, but got {B.ndim}D array with shape {B.shape}")

    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Must be one of: {', '.join(valid_models)}")
    elif model == 'BDM':
        Ihat = np.sum(((2 * k * mu - 0.5 * k * B**2 - 0.5) / B)**2) * Delta_t_B
    elif model == 'SLM':
        Ihat = np.sum((k * (mu - np.exp(B)) - 0.5)**2) * Delta_t_B

    return Ihat

@njit
def Gaussian_PDF(x, mean, var):
    std_dev = np.sqrt(var)
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)


@njit
def log_RN_Derivative_BDM(B, k, mu, D, Delta_t_B, eff_dt):

    # Eq. C22
    # \int_{t_{i}}^{t_{f}} f(x_{t})/B^{2}(x_{t}) dx_{t} - 0.5 \int_{t_{i}}^{t_{f}} f^{2}(x_{t})/B^{2}(x_{t}) dt


    x_i = B[0]
    x_f = B[-1]

    Ihat = I_hat(k, mu, B, Delta_t_B, model='BDM')

    return (( 2 * k * mu * np.log(x_f/x_i) -
              np.log(x_f/x_i) * 0.5 -
              k * (x_f**2 - x_i**2) * 0.25 )/
            (D**2)) - (Ihat / ( 2 * (D**2) ) )




@njit
def log_RN_Derivative_SLM(B, k, mu, D, Delta_t_B, eff_dt):
    x_i = B[0]
    x_f = B[-1]

    Ihat = I_hat(k, mu, B, Delta_t_B, model='SLM')

    return (((k * mu - 0.5)*(x_f - x_i) -
             k*(np.exp(x_f) - np.exp(x_i))
            ) /
            (D**2)) - (Ihat / (2 * (D**2)))



@njit
def log_RN_Derivative(B, k, mu, D, Delta_t_B, eff_dt, model='BDM'):
    valid_models = {"BDM", "SLM"}

    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Must be one of: {', '.join(valid_models)}")
    elif model == 'BDM':
        return log_RN_Derivative_BDM(B, k, mu, D, Delta_t_B, eff_dt)
    elif model == 'SLM':
        return log_RN_Derivative_SLM(B, k, mu, D, Delta_t_B, eff_dt)


@njit(parallel=True)
def logRND_each_bridge(bridges, k, mu, D, Delta_t_B, eff_dt, model='BDM'):

    # Pre-allocate the result array with the correct size
    n_bridges = len(bridges)
    logRND_array = np.empty(n_bridges)  # Pre-allocate

    # #Array of L_t for each bridge
    # logRND_array = np.array([])

    #For each bridge, calculate L and append to array
    for i in prange(n_bridges):
        bridge = bridges[i]
        logRND_array[i] = log_RN_Derivative(bridge, k, mu, D, Delta_t_B, eff_dt, model)

    return logRND_array



@njit
def est_log_propagator(bridges, k, mu, D, Delta_t_B, eff_dt, NB, x_i, x_f, model='BDM'):
    logRND_array = logRND_each_bridge(bridges, k, mu, D, Delta_t_B, eff_dt, model)
    # RND = dP/dV = derivative of path measure of the transformed measure with respect to the wiener process
    # G = dQ/dR^{(B)} = transition probability probability of Wiener process

    # Likelihood = dP/dV * dV/dQ * dQ/dR^{(B)}
    # = P/dV * dQ/dR^{(B)}
    # = RND * G

    # lL_star = Highest log-likelihood in logRND_array
    # Trick to calculate LL:
    # log((1/N) * \sum_{i} exp(l_{i}))
    # = log((1/N) * \sum_{i} exp(l_{i}^{*}) * exp(l_{i} - l_{i}^{*}) )

    # Above derivation based on Eq. H3

    lL_star = np.max(logRND_array)
    G = Gaussian_PDF(x_f, x_i, eff_dt)
    return lL_star + np.log(G) - np.log(NB) + np.log(np.sum(np.exp(logRND_array - lL_star)))




@njit(parallel=True)
def log_propagators_transformed_data(data, NB, bridge_steps, Delta_t_B, D, k, mu, seed_array, model='BDM'):

    # Pre-allocate result array
    n_pairs = len(data) - 1
    log_rho_array = np.empty(n_pairs)  # Pre-allocate with correct size

    # prange signals to numba to compile this process into parallel threads
    for i in prange(n_pairs):
        x_i = data[i]
        x_f = data[i + 1]
        seed_value = seed_array[i]

        bridges = bridge(NB, bridge_steps, x_i, x_f, Delta_t_B, D, seed_value)
        log_propagator = est_log_propagator(bridges, k, mu, D, Delta_t_B, eff_dt, NB, x_i, x_f, model)
        # log_rho_array = np.append(log_rho_array, log_propagator)
        log_rho_array[i] = log_propagator

    return log_rho_array



@njit
def log_propagators_target_process(log_rho_array, data, model='BDM'):
    valid_models = {"BDM", "SLM"}

    data = data[1:]

    # h(x) defined for each model in Table 1 pg. 8

    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Must be one of: {', '.join(valid_models)}")
    elif model == 'BDM':
        h_prime = 1 / np.sqrt(data)
        log_h_prime = np.log(h_prime)
        return log_h_prime + log_rho_array
    elif model == 'SLM':
        h_prime = 1 / data
        log_h_prime = np.log(h_prime)
        return log_h_prime + log_rho_array


@njit
def ts_logL(log_rho_array, p0):

    # p0 = prior of the first datapoint
    # set p0 = 1 when we do not want it to contribute

    return np.sum(log_rho_array) + np.log(p0)




def full_logL_pipeline(sim_data, NB, bridge_steps, Delta_t_B, D, k, mu, seed_array, model_code=0):

    models = ['BDM', 'SLM']
    model = models[model_code]

    # select transformation for each model
    if model == 'BDM':
        lamperti_tf_data = BDM_Lamperti(sim_data)
    elif model == 'SLM':
        lamperti_tf_data = SLM_Lamperti(sim_data)

    # Uses bridge formalism to output array of estimated transition probabilities on the transformed process
    # In theory notation, returns an array of \tilde{\rho}
    log_rho_tilde_array = log_propagators_transformed_data(
        lamperti_tf_data, NB, bridge_steps, Delta_t_B, D, k, mu, seed_array, model
    )

    # back-transforms log-probabilites of transformed process to log-probabilities of original process
    # Eq. H14
    # then sums the array of log-likelihoods
    timeseries_logL = ts_logL(log_propagators_target_process(log_rho_tilde_array, sim_data, model), 1.0)

    return timeseries_logL




@njit
def est_log_propagator_array_target_data(bridges, k, mu, D, Delta_t_B, eff_dt, NB, x_i, x_f, model='BDM'):
    logRND_array = logRND_each_bridge(bridges, k, mu, D, Delta_t_B, eff_dt, model) #RN Derivative for each bridge
    G = Gaussian_PDF(x_f, x_i, eff_dt)
    G_arr = np.repeat(G, len(logRND_array)) #G factor for each bridge
    log_h_prime_array = np.repeat(np.log(1/np.sqrt(x_f)), len(logRND_array)) #transform each propagator back to target data

    # np.log(G_arr) = -0.91
    # log_h_prime_array = 0


    return logRND_array + np.log(G_arr) + log_h_prime_array



def calculate_mean_var_ou(k, mu, D, eff_dt, x_i, x_f):

    mean = x_i*np.exp(-1*k*eff_dt) + mu*(1 - np.exp(-k*eff_dt))
    var = (D/k) * (1 - np.exp(-2*k*eff_dt))

    return mean, var


if __name__ == "__main__":

    print("Running...")

    MODEL = 'BDM'
    model_code = 0

    X0 = 1
    Xf = 1
    k = 1
    mu = 1
    T = 1 #timeseries timespan, starts at 0 and ends at T
    eff_dt = 1 #timestep of the datapoints
    dt = eff_dt * 1e-3 #timestep of the simulation
    real_sigma = 0.54 #real sigma such that D = sqrt(sigma / tau) = sqrt (k * sigma)
    D = np.sqrt(real_sigma * k)
    N = round(T+1 / eff_dt) #number of datapoints
    timestepsperpoint = round(eff_dt / dt)

    NB = 1000  # Number of bridges per transition
    timestep_ratio = 1e3 # dt / Dt
    bridge_steps = int(timestep_ratio) # Number of time steps inside bridge
    Delta_t_B = eff_dt / bridge_steps
    datapoint_times = np.linspace(0, T, N)

    seed=333
    seed_array=np.random.default_rng(seed=seed).integers(1, 5000, size=N-1)


    data = np.array([1.0, 1.0])
    lamperti_tf_data = BDM_Lamperti(data)
    bridges = bridge(NB, bridge_steps, lamperti_tf_data[0], lamperti_tf_data[1], Delta_t_B, D, seed_value=4242)
    log_propagator = est_log_propagator_array_target_data(bridges, k, mu, D, Delta_t_B, eff_dt, NB, X0, Xf, MODEL)

    print(np.mean(log_propagator))
