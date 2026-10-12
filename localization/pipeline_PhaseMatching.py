"""
PRT -> ROI -> HRT localization pipeline.

Modes:
1. localize(rx, time_axis, antenna_x)
   -> blind/global PRT: global p, global tau, global q.

2. localize(rx, time_axis, antenna_x, theta_hint, tau_hint)
   -> truth-assisted PRT:
      4 p bins around p_hint, 4 tau bins around tau_hint, global q.

After PRT, ROI and HRT do not use hints.
"""

from dataclasses import dataclass
import numpy as np

from config import C, SYSTEM, ARRAY
from signal_model.gaussian_source import gaussian_pulse
from signal_model.propagation import propagation_delays, source_position, propagation_distances
from pipeline import LocalizationResult


# ============================================================
# PRT settings
# ============================================================

PM_THETA_MIN, PM_THETA_MAX = -60.0, 60.0
PM_R_MIN, PM_R_MAX = 50.0, 400.0
PM_R_GRID          = np.linspace(PM_R_MIN, PM_R_MAX, 100)

# ============================================================
# Space-time Beam settings
# ============================================================
K     = 701


# ============================================================
# Helpers
# ============================================================

def get_GaussianBeam(delay_axis, K, bandwidth, dt, return_phase = False):

    delay_axis_centered = delay_axis - np.mean(delay_axis)

    t                   = np.arange(-np.int32((K-1)/2), np.int32((K-1)/2) + 1) * dt

    # Standard Envelope Function
    envelope            = gaussian_pulse(t[None, :] - delay_axis_centered[:, None])
    envelope            = envelope / np.abs(envelope).max()
    
    # Carrier Phase Compensation & Windoing
    carrier_phase       = np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delay_axis_centered)/np.sqrt(len(delay_axis))

    if return_phase:
        return envelope, carrier_phase * np.sqrt(len(delay_axis))
    else: 
        Gaussian_beam       = envelope * carrier_phase[:, None] 
        return Gaussian_beam

def position_grad(range_m: float, theta_deg: float) -> np.ndarray:
    
    """Source coordinates for a ULA along x; theta=0 is array broadside."""
    theta = theta_deg/180*np.pi
    
    xz_to_rt_grad = np.asarray([[np.sin(theta), np.cos(theta)],
                                [range_m*np.cos(theta)*180/np.pi, 
                                -range_m*np.sin(theta)*180/np.pi]])

    return xz_to_rt_grad


def distance_grad(range_m: float, theta_deg: float,
                antenna_x: np.ndarray | None = None) -> np.ndarray:
    
    """Exact spherical-wave distance from the source to every antenna."""
    if antenna_x is None:
        antenna_x = ARRAY.positions(SYSTEM)

    x_src, z_src = source_position(range_m, theta_deg)
    distances    = np.sqrt((x_src - antenna_x) ** 2 + z_src ** 2) 
    dist_to_xz_grad = np.concatenate([
                        (x_src - antenna_x)/distances, z_src/distances
                        ], axis =1)

    return dist_to_xz_grad

def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return float(R * np.sin(t)), float(R * np.cos(t))


def get_InitTheta(rx, antenna_x):

    pass

def get_InitRange(rx_tensor, antenna_x, theta_init):

    # Construct Range Codebook Based on Initial Theta
    r_power                 = np.zeros(PM_R_GRID)

    for i in range(len(PM_R_GRID)):
        FocusedSample       = get_BeamFocus(rx_tensor, PM_R_GRID[i], theta_init, antenna_x)
        r_power[i]          = np.sum(np.mean(FocusedSample)**2)

    return PM_R_GRID[r_power.argmax()]


def get_Newton_Step(rx_tensor, antenna_x, R, theta):

    # This function computes the Newton's Update for Range and Angle
    # The steps include Gradient & Hession Computation 
    
    # As an approximation, the gradient from the envelope is set to 0 
    # Only the gradient from the phase information is considered

    # Gradient Computation (Objective is to maximize the energy of the focused samples)

    # Compute Delay Information 

    delays  =  propagation_delays(R, theta, antenna_x=antenna_x)

    # Compute Gaussian Beam & PhaseShifts
    ST_envelope, PS_carrier = get_GaussianBeam(delays, K, 
                                            SYSTEM.bandwidth, SYSTEM.dt,
                                            return_phase=True)

    rx_adjoint = np.einsum('klm, km-> kl', rx_tensor, ST_envelope)
    
    # The objective to take the Newton's Step is  is: np.sum(np.abs(rx_adjoint.T.conj() * PS_carrier)**2) 
    # This is a classic problem:  
    #       d/dr, d/dtheta ||A * x(r, theta)||^2
    #       d^2/dtheta^2, d^2/drdtheta, d^2/dr^2 ||A * x(r, theta)||^2,  
    # Preparing Materials for these calculation: 
    
    scal_const          = -1j * 2.0 * np.pi * SYSTEM.fc / C
    dist_grad           = distance_grad(R, theta, antenna_x=antenna_x)
    dist_hess           = position_grad(R, theta, antenna_x=antenna_x)
    ps_grad             = scal_const * dist_grad
    ps_hess             = scal_const * dist_hess

    A_x                 = rx_adjoint.conj().T @ PS_carrier
    A_x_grad            = rx_adjoint.conj().T @ (PS_carrier[:, None] * ps_grad)

    ps_grad2            = PS_carrier[:, None, None] * ps_grad[:, :, None] * ps_grad[:, None, :]
    ps_2grad            = PS_carrier[:, None, None] * ps_hess


    A_x_2grad           = np.einsum('kl, kmn -> lmn', rx_adjoint.conj(), ps_grad2 + ps_2grad)

    PS_grad             = np.real(A_x_grad.conj().T @ A_x)
    PS_hess             = np.real(A_x_grad.conj().T @ A_x_grad + \
                                  np.einsum('ijk, i -> jk', A_x_2grad, A_x)) 

    # Solve for Newton's Method
    delta_r_theta      = np.linalg.solve(PS_hess, PS_grad)

    return delta_r_theta[0], delta_r_theta[1], np.linalg.norm(PS_grad)

def get_RxTensor(rx, K):

    zero_padding    = np.zeros((rx.shape[0], np.int32((K-1)/2)),dtype = rx.dtype)
    rx_padded       = np.concatenate([zero_padding,rx,zero_padding], axis = 1)
    rx_tensor       = np.lib.stride_tricks.sliding_window_view(rx_padded, 
                                                               window_shape=K, 
                                                               axis=1)
    return rx_tensor

def get_BeamFocus(rx_tensor, R_est, theta_est, antenna_x):

    delays          = propagation_delays(R_est, theta_est, antenna_x=antenna_x)
    StGaussBeam     = get_GaussianBeam(delays, K, SYSTEM.bandwidth, SYSTEM.dt)

    FocusedSample   = np.einsum('klm, km-> l', rx_tensor, StGaussBeam.conj())
    
    return FocusedSample

# ============================================================
# Complete pipeline
# ============================================================

def localize_PhaseMatching(rx, time_axis, antenna_x, theta_hint=None, tau_hint=None):

    if (theta_hint is None) != (tau_hint is None):
        raise ValueError("theta_hint and tau_hint must be provided together.")

    if theta_hint is None: 
        theta_init = get_InitTheta(rx, antenna_x)
    else:   
        theta_init = theta_hint 

    rx_tensor   = get_RxTensor(rx, K)       # Define Tensor Rx
    R_init      = get_InitRange(rx_tensor, antenna_x, theta_init)

    # Define Iteration Parameters 
    # =======================================================
    max_iter    = 50                        # maximum iterations 
    grad_tol    = 1e-7                      # tolerance of gradient 
    R_est       = R_init                    # Define initial R
    theta_est   = theta_init                # Define initial Theta

    for it in range(max_iter): 
        
        R_delta, theta_delta, grad_norm = get_Newton_Step(rx, antenna_x, R_est, theta_est) 

        if grad_norm <= grad_tol:

            print(f"Iteration Stop with Grad: {grad_norm:.3e} !!")
            break
        else: 
            R_est = R_est + 0.5 * R_delta 
            theta_est = theta_est + 0.5 * theta_delta 

    theta_est       = theta_est
    rx_focus        = get_BeamFocus(rx_tensor, R_est, theta_est)
    tau_hat         = time_axis[np.abs(rx_focus).argmax()]
    x_hat, y_hat    = polar_to_xy(R_est, theta_est)   
    
    return LocalizationResult(R_est, theta_est, tau_hat, x_hat, y_hat)