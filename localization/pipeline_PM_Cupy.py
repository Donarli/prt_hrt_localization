"""
Method: 
1. Produce Initial Guess By Maximizing Rx Combined Energy (If Hint is not Given)
2. Perform Newton's Method from Initial Guess to Local Stationary Point that Maximizes the energy 

The initial Guess is produced with "FracLPF" beams. The variable 
"BW" controls the trade-off between combined SNR and the range resolution. 
On the other hand, the "Gauss" beam is used in the refinement for match filtering. 
With "clean_rx", the post-refinement RMSE is on the level of 3e-2. 
"""

from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cupy_cache"))
import cupy as cp

from config import C, SYSTEM, ARRAY
from signal_model.propagation import propagation_delays
from localization.pipeline import LocalizationResult, localize

# ============================================================
# PRT settings
# ============================================================

PM_THETA_MIN, PM_THETA_MAX = -60.0, 60.0
PM_R_MIN, PM_R_MAX = 50.0, 400.0
PM_R_GRID          = np.linspace(PM_R_MIN, PM_R_MAX, 15)

# ============================================================
# Space-time Beam settings
# ============================================================
K     = 701
BW    = SYSTEM.bandwidth
PRNT  = False

# ============================================================
# Helpers
# ============================================================

def gaussian_pulse_gpu(t, sigma=SYSTEM.gaussian_sigma):
    return cp.exp(-0.5 * (t / sigma) ** 2).astype(cp.complex128)


def get_GaussianBeam(delay_axis, K, bandwidth, dt, return_phase = False):

    delay_axis = cp.asarray(delay_axis, dtype=cp.float64)
    delay_axis_centered = delay_axis - cp.mean(delay_axis)

    t                   = cp.arange(-np.int32((K-1)/2), np.int32((K-1)/2) + 1, dtype=cp.float64) * dt

    # Standard Envelope Function
    envelope            = gaussian_pulse_gpu(t[None, :] - delay_axis_centered[:, None])
    envelope            = envelope / cp.abs(envelope).max()
    
    # Carrier Phase Compensation & Windoing
    carrier_phase       = cp.exp(-1j * 2.0 * cp.pi * SYSTEM.fc * delay_axis_centered)/cp.sqrt(len(delay_axis))

    if return_phase:
        return envelope, carrier_phase * np.sqrt(len(delay_axis))
    else: 
        Gaussian_beam       = envelope * carrier_phase[:, None] 
        return Gaussian_beam

def get_FracLPFBeam(delay_axis, K, bandwidth, dt, batch_delay = True, return_phase = False):

    delay_axis          = cp.asarray(delay_axis, dtype=cp.float64)
    t                   = cp.arange(-np.int32((K-1)/2), np.int32((K-1)/2) + 1, dtype=cp.float64) * dt
    # Standard Envelope Function
    envelope            = gaussian_pulse_gpu(t)
    envelope            = envelope / cp.abs(envelope).max()

    # Find Length of the fractional-delay LPF filter  
    k                   = int(cp.asnumpy(cp.sum(cp.abs(envelope) >= 1e-4)) * 2)

    # Construct Fractional Delay & Hamming Window
    window              = cp.blackman(2 * k + 1)

    # Get Fractional Delay with LPF
    Cutoff_Freq         = bandwidth * dt

    if delay_axis.ndim == 1:
        delay_axis_centered = delay_axis - cp.mean(delay_axis)

        FracLPFBeam         = cp.zeros((len(delay_axis), K), dtype = cp.complex128)

        # Compute Interger Delay 
        st_delay    = cp.arange(-k, k+1, dtype=cp.float64)
        frac_delay  = (delay_axis_centered / dt) % 1
        int_delay   = (delay_axis_centered / dt - frac_delay).astype(cp.int32)

        # Create Entry Mask in Slepian Interpolation 
        int_delay_idx = int_delay + np.int32((K-1)/2)
        entry_idx_y   = (int_delay_idx[:, None] + st_delay[None, :]).astype(cp.int32)
        entry_idx_x   = cp.arange(0, len(delay_axis), dtype=cp.int32)[:, None] * cp.ones((1, 2*k + 1), dtype=cp.int32)

        FracLPF             = 2 * Cutoff_Freq * cp.sinc(2 * Cutoff_Freq * \
                                (frac_delay[:, None] + st_delay[None, :]))
        
        # Carrier Phase Compensation & Windoing
        carrier_phase       = cp.exp(-1j * 2.0 * cp.pi * SYSTEM.fc * delay_axis_centered)
        envelope            = FracLPF * window[None, :]

        if return_phase:
            # Assign FracLPF to Corresponding Integer Delay 
            FracLPFBeam[entry_idx_x.reshape(-1),
                        entry_idx_y.reshape(-1)] = envelope.reshape(-1)
            
            return FracLPFBeam, carrier_phase
        
        else: 
            # Assign FracLPF to Corresponding Integer Delay 
            FracLPFBeam[entry_idx_x.reshape(-1),
                        entry_idx_y.reshape(-1)] = (envelope * carrier_phase[:, None]).reshape(-1)
            
            return FracLPFBeam
        
    elif delay_axis.ndim == 2:
        # Enable Bach Processing For Acceleration
        delay_axis_centered = delay_axis - cp.mean(delay_axis, axis = 0, keepdims = True)
        n_elem              = delay_axis_centered.shape[0]
        n_beams             = delay_axis_centered.shape[1]
        FracLPFBeam         = cp.zeros((n_elem, n_beams, K) ,dtype = cp.complex128)

        # Compute Interger Delay 
        st_delay            = cp.arange(-k, k+1, dtype=cp.float64)
        frac_delay          = (delay_axis_centered / dt) % 1
        int_delay           = (delay_axis_centered / dt - frac_delay).astype(cp.int32)
        
        # Create Entry Mask in Slepian Interpolation 
        int_delay_idx       = int_delay + np.int32((K-1)/2)
        entry_idx_y         = (int_delay_idx[:, :, None] + st_delay[None, None, :]).astype(cp.int32)
        entry_idx_x         = cp.arange(0, n_elem, dtype=cp.int32)[:, None, None] * cp.ones((1, n_beams, 2*k + 1), dtype=cp.int32)
        entry_idx_z         = cp.arange(0, n_beams, dtype=cp.int32)[None, :, None] * cp.ones((n_elem, 1, 2*k + 1), dtype=cp.int32)

        # Broadcast for Envelope Generation
        FracLPF             = 2 * Cutoff_Freq * cp.sinc(2 * Cutoff_Freq * \
                                        (frac_delay[:, :, None] + st_delay[None, None, :]))

        # Carrier Phase Compensation & Windoing
        carrier_phase       = cp.exp(-1j * 2.0 * cp.pi * SYSTEM.fc * delay_axis_centered)
        FracLPF             = FracLPF * window[None, None, :] * carrier_phase[:, :, None]

        # Assign FracLPF to Corresponding Integer Delay 
        FracLPFBeam[entry_idx_x.reshape(-1),
                    entry_idx_z.reshape(-1),
                    entry_idx_y.reshape(-1)] = FracLPF.reshape(-1)

        return FracLPFBeam
    else: 
        FracLPFBeam = None
        
        return FracLPFBeam

def f(A, r, theta, xy_pos, f_c):
    """
    Computes the gradient (2x1) and Hessian (2x2) of ||A.H @ a(r, theta)||^2 
    with respect to r and theta.

    Parameters:
    -----------
    A : ndarray of shape (N, M)
        Complex matrix.
    r : float
        Scalar distance from source to center.
    theta : float
        Scalar angle in degrees.
    xy_pos : ndarray of shape (N, 2)
        x, y coordinates of the array positions.
    f_c : float
        Carrier frequency in Hz.

    Returns:
    --------
    hessian : ndarray of shape (2, 2)
        2x2 real symmetric matrix.
    gradient : ndarray of shape (2, 1)
        2x1 real column vector [dJ/dr, dJ/dtheta]^T.
    """
    c = C  # Speed of light
    
    A = cp.asarray(A)
    xy_pos = cp.asarray(xy_pos, dtype=cp.float64)

    k = 2.0 * cp.pi * f_c / c
    deg2rad = cp.pi / 180.0

    # Angle conversion
    th_rad = cp.deg2rad(theta)
    sin_th = cp.sin(th_rad)
    cos_th = cp.cos(th_rad)

    # Coordinates & distances
    x_i = xy_pos[:, 0]
    y_i = xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = cp.sqrt(dx**2 + dy**2)

    # Array response vector a
    a = cp.exp(-1j * k * d)  # Shape (N,)

    # Matrix-vector multiplications
    w = A.T.conj() @ a       # Shape (M,)
    v = A @ w                # Shape (N,)

    # First derivatives of distance d
    dd_dr = (dx * sin_th + dy * cos_th) / d
    dd_dth = deg2rad * r * (dx * cos_th - dy * sin_th) / d

    # Second derivatives of distance d
    d2d_dr2 = (1.0 - dd_dr**2) / d
    d2d_drdth = - (dd_dr * dd_dth) / d + dd_dth / r
    d2d_dth2 = - (dd_dth**2) / d + (deg2rad**2 * r**2) / d - deg2rad**2 * r * dd_dr

    # First derivatives of steering vector a
    da_dr = -1j * k * dd_dr * a
    da_dth = -1j * k * dd_dth * a

    # Intermediate term: a_conj * v
    a_conj_v = cp.conj(a) * v

    # Gradient calculations
    dJ_dr = -2.0 * k * cp.imag(cp.sum(dd_dr * a_conj_v))
    dJ_dth = -2.0 * k * cp.imag(cp.sum(dd_dth * a_conj_v))
    gradient = cp.array([[dJ_dr], [dJ_dth]], dtype=cp.float64)

    # Intermediate terms for Hessian
    w_r = A.T.conj() @ da_dr
    w_th = A.T.conj() @ da_dth

    term1_rr = 2.0 * cp.real(cp.vdot(w_r, w_r))
    term1_rth = 2.0 * cp.real(cp.vdot(w_r, w_th))
    term1_thth = 2.0 * cp.real(cp.vdot(w_th, w_th))

    # Hessian components
    H_rr = term1_rr - 2.0 * k * cp.imag(cp.sum(d2d_dr2 * a_conj_v)) \
           - 2.0 * (k**2) * cp.real(cp.sum((dd_dr**2) * a_conj_v))

    H_rth = term1_rth - 2.0 * k * cp.imag(cp.sum(d2d_drdth * a_conj_v)) \
            - 2.0 * (k**2) * cp.real(cp.sum((dd_dr * dd_dth) * a_conj_v))

    H_thth = term1_thth - 2.0 * k * cp.imag(cp.sum(d2d_dth2 * a_conj_v)) \
             - 2.0 * (k**2) * cp.real(cp.sum((dd_dth**2) * a_conj_v))

    hessian = cp.array([[H_rr, H_rth],
                        [H_rth, H_thth]], dtype=cp.float64)

    return hessian, gradient


def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return float(R * np.sin(t)), float(R * np.cos(t))


def get_InitTheta(rx, antenna_x):

    pass

def get_InitRange(rx_tensor, antenna_x, theta_init, batch_process = True):

    if batch_process:
        # Construct Delay Matrix 
        delay_mat               = np.zeros((len(antenna_x), len(PM_R_GRID)), dtype=np.float64)

        for i in range(len(PM_R_GRID)):
            delay_mat[:, i]     = propagation_delays(PM_R_GRID[i], theta_init, antenna_x=antenna_x)

    
        ST_codebook             = get_FracLPFBeam(delay_mat, K, BW, SYSTEM.dt)
        r_power                 = cp.linalg.norm(cp.einsum('klm, knm -> ln',
                                                           rx_tensor, 
                                                           ST_codebook.conj()), 2,
                                                           axis = 0)  / \
                                  cp.sqrt(cp.sum(cp.abs(ST_codebook)**2, axis = [0, 2]))
    else:
        # Construct Range Codebook Based on Initial Theta
        r_power                 = cp.zeros(len(PM_R_GRID))

        for i in range(len(PM_R_GRID)):
            FocusedSample       = get_BeamFocus(rx_tensor, PM_R_GRID[i], 
                                                theta_init, 
                                                antenna_x, 
                                                beam_type='Gauss')
            
            r_power[i]          = cp.abs(FocusedSample).max()

    return float(PM_R_GRID[int(cp.asnumpy(r_power.argmax()))])

def get_NewtonGem_Step(rx_tensor, antenna_x, R, theta, confinement = None):

    # This function computes the Newton's Update for Range and Angle
    # The steps include Gradient & Hession Computation 
    
    # As an approximation, the gradient from the envelope is set to 0 
    # Only the gradient from the phase information is considered

    # Gradient Computation (Objective is to maximize the energy of the focused samples)

    # Compute Delay Information 

    delays  =  propagation_delays(R, theta, antenna_x=antenna_x)

    # Compute Gaussian Beam & PhaseShifts
    ST_envelope, PS_carrier = get_GaussianBeam(delays, K, 
                                            BW, SYSTEM.dt,
                                            return_phase=True)

    # Compute Adjoint Rx
    rx_adjoint = cp.einsum('klm, km-> kl', rx_tensor, ST_envelope.conj())
    A_x        = rx_adjoint.conj().T @ PS_carrier / cp.sqrt(len(delays))


    if confinement is not None:
        peak_indx  = np.abs(A_x).argmax()
        A_x        = A_x[peak_indx - confinement : peak_indx + confinement+1]
        rx_adjoint = rx_adjoint[:, peak_indx - confinement : peak_indx + confinement+1]

    antenna_xy = cp.stack([cp.asarray(antenna_x, dtype=cp.float64),
                           cp.zeros(len(antenna_x), dtype=cp.float64)], axis = 1)
    PS_hess, PS_grad = f(rx_adjoint, R, theta, antenna_xy, SYSTEM.fc)

    # Solve for Newton's Method
    delta_r_theta      = cp.linalg.solve(PS_hess, PS_grad)

    return (float(delta_r_theta[0].item()),
            float(delta_r_theta[1].item()),
            float(cp.linalg.norm(PS_grad).item()),
            A_x)


def get_RxTensor(rx, K):

    rx_gpu          = cp.asarray(rx)
    zero_padding    = cp.zeros((rx_gpu.shape[0], np.int32((K-1)/2)),dtype = rx_gpu.dtype)
    rx_padded       = cp.concatenate([zero_padding,rx_gpu,zero_padding], axis = 1)
    rx_tensor       = cp.lib.stride_tricks.sliding_window_view(rx_padded,
                                                               window_shape=K, 
                                                               axis=1)
    return rx_tensor

def get_BeamFocus(rx_tensor, R_est, theta_est, antenna_x):

    delays          = propagation_delays(R_est, theta_est, antenna_x=antenna_x)
    StBeam          = get_GaussianBeam(delays, K, BW, SYSTEM.dt)
    FocusedSample   = cp.einsum('klm, km-> l', rx_tensor, StBeam.conj())
    
    return FocusedSample


# ============================================================
# Complete pipeline
# ============================================================

def localize_PhaseMatching(rx, time_axis, antenna_x, theta_hint=None, tau_hint=None, R_hint = None, 
                           use_radon = False):

    rx_tensor   = get_RxTensor(rx, K)       # Define Tensor Rx
    if use_radon:
        results       = localize(rx, time_axis, antenna_x, theta_hint, tau_hint)
        R_init        = results.R if R_hint is None else R_hint
        theta_init    = results.theta
    else:
        theta_init    = get_InitTheta(rx, antenna_x) if theta_hint is None else theta_hint
        R_init        = get_InitRange(rx_tensor, antenna_x, theta_init) if R_hint is None else R_hint
    
    if PRNT:
        print(f"Initial Range Estimated at: {R_init:.2f} !!")

    # Define Iteration Parameters 
    # =======================================================
    max_iter    = 50                        # maximum iterations 
    grad_tol    = 1e-9                      # tolerance of gradient
    step_size   = 0.85                       # Step-size of Newton's Update 
    R_est       = R_init                    # Define initial R
    theta_est   = theta_init                # Define initial Theta
    power_hist  = 0                         # Used to Compute Power Delta
    t_confine   = None                         # Confinement of Time-domain Samples                          

    for it in range(max_iter): 
        
        R_delta, theta_delta, grad_norm, Combined_Samples = get_NewtonGem_Step(rx_tensor, 
                                antenna_x, 
                                R_est, 
                                theta_est, 
                                confinement = t_confine) 

        if grad_norm <= grad_tol:
            if PRNT:
                print(f"Iteration Stop with Grad: {grad_norm:.3e} !!")
            break
        else: 
            R_est_new = R_est - step_size * R_delta 
            theta_est_new = theta_est - step_size * theta_delta 

        power_new = float(cp.mean(cp.abs(Combined_Samples)**2).item())
        power_delta = (power_new-power_hist)/(power_new + power_hist)

        if PRNT:
            print(f"Iter. {it:3d}, "
                f"Power Delta {power_delta:.3e}, "
                f"R: {R_est_new: .2f},  "
                f"T: {theta_est_new: .2f},  "
                f"rDelta: {abs(R_delta): .2e},  "
                f"tDelta: {abs(theta_delta): .2e},  "
                f"Grad_Norm: {grad_norm: .2e} !!")

        if power_delta < 1e-9: 
            break

        power_hist = power_new
        R_est      = R_est_new
        theta_est  = theta_est_new

        if 0: 
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(cp.asnumpy(cp.abs(Combined_Samples)))
            plt.grid()
            plt.savefig('figures/debug_p1.png')
            plt.close()
            breakpoint()

    # np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delay_axis_centered)
    rx_focus        = get_BeamFocus(rx_tensor, R_est, theta_est, antenna_x=antenna_x)
    tau_hat         = time_axis[int(cp.asnumpy(cp.abs(rx_focus).argmax()))]
    x_hat, y_hat    = polar_to_xy(R_est, theta_est)   


    return LocalizationResult(R_est, theta_est, tau_hat, x_hat, y_hat,
                              0,0,0,0,0,0)
