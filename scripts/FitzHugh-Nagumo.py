import math
import numpy as np
from timeit import default_timer as timer

"""
Numerical simulation of the FitzHugh-Nagumo model on [0, T]:

dV_t = (1/epsilon) * (V_t - V_t^3 - U_t) dt + sigma_1 dW_t^1
dU_t = (gamma * V_t - U_t  + beta) dt + sigma_1 dW_t^2

using either (a) High order splitting method from:
    
    J. Foster, G. Dos Reis and C. Strange, High order splitting methods for SDEs
    satisfying a commutativity condition, arxiv.org/abs/2210.17543, 2022.

             (b) Strang splitting method from:
                 
    E. Buckwar, A. Samson, M. Tamborrino, and I. Tubikanec, A splitting method
    for SDEs with locally Lipschitz drift: Illustration on the FitzHugh-Nagumo
    model, Applied Numerical Mathematics, 2022.
                 
             (c) Tamed Euler-Maruyama method from:

    M. Hutzenthaler, A. Jentzen, and P. E. Kloeden, Strong convergence of an
    explicit numerical method for SDEs with nonglobally Lipschitz continuous
    coefficients, Annals of Applied Probability, 2012.

To estimate Strong (or L2) errors, this is done by Monte Carlo simulation using
different step sizes but with the same Brownian paths.
"""

# Model parameters
epsilon = 1.0
gamma = 1.0
beta = 1.0
sigma1 = 1.0
sigma2 = 1.0

# Useful precomputed constants
one_over_epsilon = 1.0/epsilon
two_over_epsilon = 2.0/epsilon
two_gamma = 2.0*gamma
kappa = (4.0*gamma / epsilon) - 1.0
root_kappa = math.sqrt(abs(kappa))
half_root_kappa = 0.5*root_kappa
one_over_root_kappa = 1.0/root_kappa
epsilon_squared = pow(epsilon, 2)
gamma_over_epsilon = gamma/epsilon
gamma_over_epsilon_squared = gamma/epsilon_squared
two_kappa = 2.0*kappa
two_gamma_kappa = gamma*two_kappa
sigma1_squared = pow(sigma1, 2)
sigma2_squared = pow(sigma2, 2)
sigma2_over_epsilon = sigma2/epsilon
sigma2_squared_over_epsilon = pow(sigma2, 2)/epsilon
sigma2_squared_over_epsilon_squared = pow(sigma2_over_epsilon, 2)

if (kappa != 0.0):
    one_over_root_kappa = 1.0/math.sqrt(abs(kappa))

skew_const = 0.25/math.sqrt(6.0*math.pi)
twelve = 1.0/12.0
fifteenth = 1.0/15.0


# Computes one step of the Euler-Maruyama method
def euler_maruyama(vu, h, w):
    new_vu = np.array([0.0, 0.0]).T
    new_vu[0] = vu[0] + one_over_epsilon*(vu[0] - pow(vu[0], 3) - vu[1])*h + sigma1*w[0]
    new_vu[1] = vu[1] + (gamma*vu[0] - vu[1] + beta)*h + sigma2*w[1]
    
    return new_vu


# Computes one step of the Tamed Euler-Maruyama method from Hutzenthaler et al. (2012)
def tamed_euler_maruyama(vu, h, w):
    drift = np.array([0.0, 0.0]).T
    drift[0] = one_over_epsilon*(vu[0] - pow(vu[0], 3) - vu[1])
    drift[1] = (gamma*vu[0] - vu[1] + beta)
    
    drift_norm = math.sqrt(drift[0] ** 2 + drift[1] ** 2)
    
    new_vu = np.array([0.0, 0.0]).T
    new_vu[0] = vu[0] + (drift[0]*h)/(1.0 + drift_norm*h) + sigma1*w[0]
    new_vu[1] = vu[1] + (drift[1]*h)/(1.0 + drift_norm*h) + sigma2*w[1]
    
    return new_vu


# Exact solution for the linear part of the "drift ODE"
# (the following is a simple implementation of the formulae from Buckwar et al. 2022)
def solve_linear_ODE(vu, h):
    if (kappa == 0.0):  
        c1 = 1.0
        c2 = 0.5*h        
    else:
        if kappa > 0.0:
            cos_term = math.cos(half_root_kappa*h)
            sin_term = math.sin(half_root_kappa*h)
        else:
            cos_term = math.cosh(half_root_kappa*h)
            sin_term = math.sinh(half_root_kappa*h)        
    
        c1 = cos_term
        c2 = one_over_root_kappa*sin_term
    
    exp_matrix = np.array([[c1 + c2, - two_over_epsilon*c2],
                           [two_gamma*c2, c1 - c2]])
    
    return math.exp(-0.5*h)*np.matmul(exp_matrix, vu)


# Exact solution for the linear part of the SDE
# (the following is a simple implementation of the formulae from Buckwar et al. 2022)
def solve_linear_SDE(vu, h):
    exp_h = math.exp(h)
    exp_minus_h = math.exp(-h)
    
    # Compute the covariance matrix
    if (kappa == 0.0):
        c11 = (exp_minus_h / (4.0 * epsilon_squared)) \
                 * (4.0*sigma2_squared*(2.0*exp_h - 2.0 - h*(2.0 + h))) \
                     + epsilon_squared * sigma1_squared * (10.0*exp_h - 10.0 - h * (6.0 + h))
                     
        c12 = (exp_minus_h / (8.0*epsilon)) * (- 4.0*sigma2_squared * pow(h, 2) + epsilon_squared * sigma1_squared * (4.0*exp_h - pow((2.0 + h), 2)))

        c22 = 0.0625*exp_minus_h * (4.0*sigma2_squared*(2.0*exp_h - 2.0 - (h - 2.0) * h) \
                                    + epsilon_squared*sigma1_squared*(2.0*exp_h - 2.0 - h*(2.0+h)))
    else:
        if kappa > 0.0:
            cos_term = math.cos(root_kappa*h)
            sin_term = math.sin(root_kappa*h)
        else:
            cos_term = math.cosh(root_kappa*h)
            sin_term = math.sinh(root_kappa*h)
        
        c11 = ((epsilon*exp_minus_h)/two_gamma_kappa) \
                 *((-4.0*gamma_over_epsilon_squared)*(sigma1_squared*gamma + sigma2_squared_over_epsilon) \
                    + kappa*exp_h*(sigma1_squared*(1.0 + gamma_over_epsilon) + sigma2_squared_over_epsilon_squared) \
                       + (sigma1_squared*(1.0 - 3.0*gamma_over_epsilon) + sigma2_squared_over_epsilon_squared) * cos_term \
                          - root_kappa*(sigma1_squared*(1.0 - gamma_over_epsilon) + sigma2_squared_over_epsilon_squared) * sin_term)
                
        c12 = ((epsilon*exp_minus_h)/two_kappa)*(sigma1_squared*kappa*exp_h - two_over_epsilon*(sigma1_squared*gamma + sigma2_squared_over_epsilon) \
                 + (sigma1_squared*(1.0 - 2.0*gamma_over_epsilon) + 2.0*sigma2_squared_over_epsilon_squared) * cos_term \
                    - sigma1_squared*root_kappa * sin_term)
    
        c22 = ((epsilon*exp_minus_h)/two_kappa)*((sigma2_squared_over_epsilon + sigma1_squared*gamma) \
                 * (cos_term - 4.0*gamma_over_epsilon + kappa*exp_h) \
                    + (sigma2_squared_over_epsilon - sigma1_squared * gamma) * root_kappa * sin_term)
    
    covariance_matrix = np.array([[c11, c12],
                                  [c12, c22]])
        
    return np.random.multivariate_normal(solve_linear_ODE(vu, h), covariance_matrix)


# Exact solution for the nonlinear part of the "drift" ODE
def solve_nonlinear_ODE(vu, h):
    
    exp_two_over_epsilon_h = math.exp(-two_over_epsilon*h)
    
    v_new = vu[0] / math.sqrt(exp_two_over_epsilon_h \
                              + ((vu[0])**2)*(1.0-exp_two_over_epsilon_h))
    
    return np.array([v_new, vu[1] + beta*h]).T


# Computes one step of the SDE Strang splitting from Buckwar et al. (2022)
def SDE_strang_splitting(vu, h):
    
    return solve_nonlinear_ODE(solve_linear_SDE(solve_nonlinear_ODE(vu, 0.5*h), h), 0.5*h)


# Solve the "drift" ODE using a Strang splitting (note there are two choices here!)
def solve_ODE_strang(vu, h):
    # return solve_linear_ODE(solve_nonlinear_ODE(solve_linear_ODE(vu, 0.5*h), h), 0.5*h)
    return solve_nonlinear_ODE(solve_linear_ODE(solve_nonlinear_ODE(vu, 0.5*h), h), 0.5*h)


# Solve the "diffusion" ODE. This is reduces to a simply addition
def solve_diffusion(vu, w):
    
    return np.array([vu[0] + sigma1*w[0], vu[1] + sigma2*w[1]]).T


# Computes one step of the high order splitting from Foster et al. (2022)
def high_order_splitting(vu, h, sqrt_h, increment, area, skew):
    eps = np.array([1, 1]).T
    vertical_shift = np.array([0.0, 0.0]).T
    skew_const_sqrt_h = skew_const*sqrt_h*skew

    for i in range(2):
        if (increment[i] > 6.0*skew_const_sqrt_h[i]):
            eps[i] = -1
    
        vertical_shift[i] = eps[i]*math.sqrt(twelve*(increment[i]**2) + 0.2*(area[i]**2) \
                                             + fifteenth*h - skew_const_sqrt_h[i]*increment[i])
    
    b = 0.5*increment + area + vertical_shift
    c = 0.5*increment + area - vertical_shift
    
    return solve_diffusion(solve_ODE_strang(solve_diffusion(solve_ODE_strang(solve_diffusion(
               vu, b), 0.5*h), c - b), 0.5*h), increment - c)
        

# Initial condition
vu0 = np.array([0.0, 0.0]).T
vu = vu0
fine_vu = vu0

# Time Horizon
T = 5.0

# Number of sample paths
no_of_paths = 1000

# Number of steps
no_of_steps = 320

# Useful constants related to step size
step_size = T/no_of_steps
one_over_half_step_size = 2.0/step_size
sqrt_step_size = math.sqrt(step_size)
sqrt_twelve_step_size = math.sqrt(step_size/12.0)

# The number of "fine" steps per "course" step used to estimate Strong error
half_of_fine_steps = 5

# Useful constants related to the "fine" step size
fine_step_size = step_size/(2.0*half_of_fine_steps)
sqrt_fine_step_size = math.sqrt(fine_step_size)
sqrt_twelve_fine_step_size = math.sqrt(fine_step_size/12.0)

# Information related to the Brownian motion
# (in Foster et al. (2022), this is denoted by W_k, H_k and n_k)
brownian_increment = np.array([0.0, 0.0]).T
fine_brownian_increment = np.array([0.0, 0.0]).T

brownian_area = np.array([0.0, 0.0]).T
fine_brownian_area = np.array([0.0, 0.0]).T

brownian_swing = np.array([0, 0]).T
fine_brownian_swing = np.array([0, 0]).T

L2_error = 0.0

for i in range(no_of_paths):
    # Reset the numerical solutions
    vu = vu0
    fine_vu = vu0

    # Compute SDE numerical solutions using the "fine" and "course" step sizes,
    # but using the same Brownian sample paths so that we can estimate L2 error
    for j in range(no_of_steps):
        half_brownian_increment = [np.array([0.0,0.0]).T] * 2
        half_brownian_area = [np.array([0.0,0.0]).T] * 2
        
        for l in range(2):
            for k in range(half_of_fine_steps):
                # Generate information about the Brownian path over the "fine" increment
                fine_brownian_increment = np.random.normal(0.0, sqrt_fine_step_size, 2).T
                fine_brownian_area = np.random.normal(0.0, sqrt_twelve_fine_step_size, 2).T
                fine_brownian_swing = (2*np.random.randint(2, size=2) - np.ones(2, dtype=int)).T
            
                # Propagate the numerical solution over the fine increment
                fine_vu = high_order_splitting(fine_vu, fine_step_size, sqrt_fine_step_size,
                                               fine_brownian_increment, fine_brownian_area,
                                               fine_brownian_swing)
            
                # Update the information about the Brownian path over the
                # course increment using the recently generated variables.
                # The below procedure can be derived using some elementary
                # properties of integration (additivity and linearity)
                
                # Since we are using space-time Lévy swings, we shall
                # first do this over the two half-intervals seperately
                half_brownian_area[l] = half_brownian_area[l] + fine_step_size \
                                            * (half_brownian_increment[l]  \
                                                   + 0.5*fine_brownian_increment \
                                                   + fine_brownian_area)
                                                
                # Compute the Brownian increment over half the course interval
                half_brownian_increment[l] = half_brownian_increment[l] + fine_brownian_increment

            # Compute space-time area of the path over half the course interval
            half_brownian_area[l] = one_over_half_step_size*half_brownian_area[l] \
                                        - 0.5*half_brownian_increment[l]
        
        # Compute the increment of the Brownian motion over the course interval
        brownian_increment = half_brownian_increment[0] + half_brownian_increment[1]
        
        # Compute the space-time area of the path over the course interval
        brownian_area = 0.5*(half_brownian_area[0] + half_brownian_area[1]) \
                            + 0.25*(half_brownian_increment[0] - half_brownian_increment[1]) 
        
        # As we did these computations for each half-interval, we can compute
        # the space-time Lévy swing (see Foster et al. (2022) for details)
        brownian_swing = np.array([1, 1]).T
         
        for l in range(2):
            if (half_brownian_area[0][l] < half_brownian_area[1][l]):
                brownian_swing[l] = -1
                
        # Propagate the numerical solutions over the course increment
        #vu = tamed_euler_maruyama(vu, step_size, brownian_increment)
        vu = high_order_splitting(vu, step_size, sqrt_step_size,
                                  brownian_increment, brownian_area,
                                  brownian_swing)
        
    # Compute the L2 error between the methods on the fine and course scales 
    L2_error += (vu[0] - fine_vu[0])**2 + (vu[1] - fine_vu[1])**2

L2_error = math.sqrt(L2_error/no_of_paths)

# Time the numerical method
start = timer()

for i in range(no_of_paths):
    vu = vu0

    for j in range(no_of_steps):
        brownian_increment = np.random.normal(0.0, sqrt_step_size, 2).T
        brownian_area = np.random.normal(0.0, sqrt_twelve_step_size, 2).T
        brownian_swing = (2*np.random.randint(2, size=2) - np.ones(2, dtype=int)).T         
       
        #vu = tamed_euler_maruyama(vu, step_size, brownian_increment)
        vu = high_order_splitting(vu, step_size, sqrt_step_size,
                                  brownian_increment, brownian_area,
                                  brownian_swing)
        #vu = SDE_strang_splitting(vu, step_size)
        
end = timer()

time_taken = end - start

# Display results
print("Nubmer of steps: ", no_of_steps)
print("Nubmer of sample paths: ", no_of_paths)
print("L2 Error at T =", T, ": ", L2_error)
print("Time taken: ", time_taken)