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
from signal_model.propagation import propagation_delays
from localization.pipeline import LocalizationResult

# ============================================================
# PRT settings
# ============================================================

PM_THETA_MIN, PM_THETA_MAX = -60.0, 60.0
PM_R_MIN, PM_R_MAX = 50.0, 400.0
PM_R_GRID          = np.linspace(PM_R_MIN, PM_R_MAX, 30)

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

def get_FracLPFBeam(delay_axis, K, bandwidth, dt, batch_delay = True):

    t                   = np.arange(-np.int32((K-1)/2), np.int32((K-1)/2) + 1) * dt
    # Standard Envelope Function
    envelope            = gaussian_pulse(t)
    envelope            = envelope / np.abs(envelope).max()

    # Find Length of the fractional-delay LPF filter  
    k                   = np.int32(np.sum(envelope >= 1e-4) * 2)

    # Construct Fractional Delay & Hamming Window
    window              = np.hamming(2 * k + 1)

    # Get Fractional Delay with LPF
    Cutoff_Freq         = bandwidth * dt

    if delay_axis.ndim == 1:
        delay_axis_centered = delay_axis - np.mean(delay_axis)

        FracLPFBeam         = np.zeros((len(delay_axis), K), dtype = np.complex128)

        # Compute Interger Delay 
        st_delay    = np.arange(-k, k+1)
        frac_delay  = (delay_axis_centered / dt) % 1
        int_delay   = np.int32(delay_axis_centered / dt - frac_delay)

        # Create Entry Mask in Slepian Interpolation 
        int_delay_idx = int_delay + np.int32((K-1)/2)
        entry_idx_y   = np.int32(int_delay_idx[:, None] + st_delay[None, :])
        entry_idx_x   = np.int32(np.arange(0, len(delay_axis))[:, None] * np.ones((1, 2*k + 1)))

        FracLPF             = 2 * Cutoff_Freq * np.sinc(2 * Cutoff_Freq * \
                                (frac_delay[:, None] + st_delay[None, :]))
        
        # Carrier Phase Compensation & Windoing
        carrier_phase       = np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delay_axis_centered)
        FracLPF             = FracLPF * window[None, :] * carrier_phase[:, None]

        # Assign FracLPF to Corresponding Integer Delay 
        FracLPFBeam[entry_idx_x.reshape(-1),
                    entry_idx_y.reshape(-1)] = FracLPF.reshape(-1)
        
    elif delay_axis.ndim == 2:
        # Enable Bach Processing For Acceleration
        delay_axis_centered = delay_axis - np.mean(delay_axis, axis = 0, keepdims = True)
        n_elem              = delay_axis_centered.shape[0]
        n_beams             = delay_axis_centered.shape[1]
        FracLPFBeam         = np.zeros((n_elem, n_beams, K) ,dtype = np.complex128)

        # Compute Interger Delay 
        st_delay            = np.arange(-k, k+1)
        frac_delay          = (delay_axis_centered / dt) % 1
        int_delay           = np.int32(delay_axis_centered / dt - frac_delay)
        
        # Create Entry Mask in Slepian Interpolation 
        int_delay_idx       = int_delay + np.int32((K-1)/2)
        entry_idx_y         = np.int32(int_delay_idx[:, :, None] + st_delay[None, None, :])
        entry_idx_x         = np.int32(np.arange(0, n_elem)[:, None, None] * np.ones((1, n_beams, 2*k + 1)))
        entry_idx_z         = np.int32(np.arange(0, n_beams)[None, :, None] * np.ones((n_elem, 1, 2*k + 1)))

        # Broadcast for Envelope Generation
        FracLPF             = 2 * Cutoff_Freq * np.sinc(2 * Cutoff_Freq * \
                                        (frac_delay[:, :, None] + st_delay[None, None, :]))
                
        # Carrier Phase Compensation & Windoing
        carrier_phase       = np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delay_axis_centered)
        FracLPF             = FracLPF * window[None, None, :] * carrier_phase[:, :, None]

        # Assign FracLPF to Corresponding Integer Delay 
        FracLPFBeam[entry_idx_x.reshape(-1),
                    entry_idx_z.reshape(-1),
                    entry_idx_y.reshape(-1)] = FracLPF.reshape(-1)
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
    
    k = 2.0 * np.pi * f_c / c
    deg2rad = np.pi / 180.0

    # Angle conversion
    th_rad = np.radians(theta)
    sin_th = np.sin(th_rad)
    cos_th = np.cos(th_rad)

    # Coordinates & distances
    x_i = xy_pos[:, 0]
    y_i = xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = np.sqrt(dx**2 + dy**2)

    # Array response vector a
    a = np.exp(-1j * k * d)  # Shape (N,)

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
    a_conj_v = np.conj(a) * v

    # Gradient calculations
    dJ_dr = -2.0 * k * np.imag(np.sum(dd_dr * a_conj_v))
    dJ_dth = -2.0 * k * np.imag(np.sum(dd_dth * a_conj_v))
    gradient = np.array([[dJ_dr], [dJ_dth]], dtype=np.float64)

    # Intermediate terms for Hessian
    w_r = A.T.conj() @ da_dr
    w_th = A.T.conj() @ da_dth

    term1_rr = 2.0 * np.real(np.vdot(w_r, w_r))
    term1_rth = 2.0 * np.real(np.vdot(w_r, w_th))
    term1_thth = 2.0 * np.real(np.vdot(w_th, w_th))

    # Hessian components
    H_rr = term1_rr - 2.0 * k * np.imag(np.sum(d2d_dr2 * a_conj_v)) \
           - 2.0 * (k**2) * np.real(np.sum((dd_dr**2) * a_conj_v))

    H_rth = term1_rth - 2.0 * k * np.imag(np.sum(d2d_drdth * a_conj_v)) \
            - 2.0 * (k**2) * np.real(np.sum((dd_dr * dd_dth) * a_conj_v))

    H_thth = term1_thth - 2.0 * k * np.imag(np.sum(d2d_dth2 * a_conj_v)) \
             - 2.0 * (k**2) * np.real(np.sum((dd_dth**2) * a_conj_v))

    hessian = np.array([[H_rr, H_rth], 
                        [H_rth, H_thth]], dtype=np.float64)

    return hessian, gradient


def polar_to_xy(R, theta_deg):
    t = np.deg2rad(theta_deg)
    return float(R * np.sin(t)), float(R * np.cos(t))


def get_InitTheta(rx, antenna_x):

    pass

def get_InitRange(rx_tensor, antenna_x, theta_init, batch_process = True):

    if batch_process:
        # Construct Delay Matrix 
        delay_mat               = np.zeros((len(antenna_x), len(PM_R_GRID)))

        for i in range(len(PM_R_GRID)):
            delay_mat[:, i]     = propagation_delays(PM_R_GRID[i], theta_init, antenna_x=antenna_x)

    
        ST_codebook             = get_FracLPFBeam(delay_mat, K, SYSTEM.bandwidth/4, SYSTEM.dt)
        r_power                 = np.linalg.norm(np.einsum('klm, knm -> ln', 
                                                           rx_tensor, 
                                                           ST_codebook.conj()), 2,
                                                           axis = 0)
        
    else:
        # Construct Range Codebook Based on Initial Theta
        r_power                 = np.zeros(len(PM_R_GRID))

        for i in range(len(PM_R_GRID)):
            FocusedSample       = get_BeamFocus(rx_tensor, PM_R_GRID[i], theta_init, antenna_x)
            r_power[i]          = np.sum(np.abs(FocusedSample)**2)

    return PM_R_GRID[r_power.argmax()]

def get_NewtonGem_Step(rx_tensor, antenna_x, R, theta):

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

    rx_adjoint = np.einsum('klm, km-> kl', rx_tensor, ST_envelope.conj())

    A_x        = rx_adjoint.conj().T @ PS_carrier / np.sqrt(len(delays))                     

    antenna_xy = np.stack([antenna_x, np.zeros(len(antenna_x))], axis = 1)
    PS_hess, PS_grad = f(rx_adjoint, R, theta, antenna_xy, SYSTEM.fc)

    # Solve for Newton's Method
    delta_r_theta      = np.linalg.solve(PS_hess, PS_grad)

    return delta_r_theta[0].item(), delta_r_theta[1].item(), np.linalg.norm(PS_grad), A_x


def get_RxTensor(rx, K):

    zero_padding    = np.zeros((rx.shape[0], np.int32((K-1)/2)),dtype = rx.dtype)
    rx_padded       = np.concatenate([zero_padding,rx,zero_padding], axis = 1)
    rx_tensor       = np.lib.stride_tricks.sliding_window_view(rx_padded, 
                                                               window_shape=K, 
                                                               axis=1)
    return rx_tensor

def get_BeamFocus(rx_tensor, R_est, theta_est, antenna_x, beam_type = 'FracLPF'):

    delays          = propagation_delays(R_est, theta_est, antenna_x=antenna_x)
    if beam_type == 'Gauss':
        StBeam     = get_GaussianBeam(delays, K, SYSTEM.bandwidth, SYSTEM.dt)
    elif beam_type == 'FracLPF':
        StBeam     = get_FracLPFBeam(delays, K, SYSTEM.bandwidth, SYSTEM.dt)

    FocusedSample   = np.einsum('klm, km-> l', rx_tensor, StBeam.conj())
    
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

    print(f"Initial Range Estimated at: {R_init:.2f} !!")

    # Define Iteration Parameters 
    # =======================================================
    max_iter    = 50                        # maximum iterations 
    grad_tol    = 1e-8                      # tolerance of gradient
    step_size   = 0.85                      # Step-size of Newton's Update 
    R_est       = R_init                    # Define initial R
    theta_est   = theta_init                # Define initial Theta

    for it in range(max_iter): 
        
        R_delta, theta_delta, grad_norm, Combined_Samples = get_NewtonGem_Step(rx_tensor, 
                                antenna_x, 
                                R_est, 
                                theta_est) 

        if grad_norm <= grad_tol:
            print(f"Iteration Stop with Grad: {grad_norm:.3e} !!")
            break
        else: 
            R_est = R_est - step_size * R_delta 
            theta_est = theta_est - step_size * theta_delta 

        print(f"Iter. {it:3d}, "
              f"Power {np.mean(np.abs(Combined_Samples)**2):.3e}, "
              f"R: {R_est: .2f},  "
              f"T: {theta_est: .2f},  "
              f"rDelta: {np.abs(R_delta): .2e},  " 
              f"tDelta: {np.abs(theta_delta): .2e},  "
              f"Grad_Norm: {grad_norm: .2e} !!")

    # np.exp(-1j * 2.0 * np.pi * SYSTEM.fc * delay_axis_centered)
    rx_focus        = get_BeamFocus(rx_tensor, R_est, theta_est, antenna_x=antenna_x)
    tau_hat         = time_axis[np.abs(rx_focus).argmax()]
    x_hat, y_hat    = polar_to_xy(R_est, theta_est)   
    
    return LocalizationResult(R_est, theta_est, tau_hat, x_hat, y_hat,
                              0,0,0,0,0,0)
