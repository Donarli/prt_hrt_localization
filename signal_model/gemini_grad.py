import numpy as np
from config import C
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


def f_RTheTau(A, r, theta, tau, xy_pos, f_c, delta_T, sigma):
    """
    Computes the 3x1 gradient and 3x3 Hessian of J(r, theta, tau) = |trace(A.H @ B)|^2.

    Parameters:
    -----------
    A : ndarray of shape (N, M), complex
    r : float, scalar distance
    theta : float, scalar angle in degrees
    tau : float, scalar time delay in seconds
    xy_pos : ndarray of shape (N, 2), sensor coordinates
    f_c : float, carrier frequency in Hz
    t : float, evaluation reference time in seconds
    delta_T : float, sampling interval in seconds
    sigma : float, Gaussian pulse width in seconds

    Returns:
    --------
    hessian : ndarray of shape (3, 3), real symmetric matrix
    gradient : ndarray of shape (3, 1), real column vector [dJ/dr, dJ/dtheta, dJ/dtau]^T
    """
    c = C
    k = 2.0 * np.pi * f_c / c
    deg2rad = np.pi / 180.0

    # Geometry & steering vector derivatives
    th_rad = np.radians(theta)
    sin_th, cos_th = np.sin(th_rad), np.cos(th_rad)

    x_i, y_i = xy_pos[:, 0], xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = np.sqrt(dx**2 + dy**2)  # Shape (N,)

    a = np.exp(-1j * k * d)     # Shape (N,)

    # Distance derivatives
    dd_dr = (dx * sin_th + dy * cos_th) / d
    dd_dth = deg2rad * r * (dx * cos_th - dy * sin_th) / d

    d2d_dr2 = (1.0 - dd_dr**2) / d
    d2d_drdth = - (dd_dr * dd_dth) / d + dd_dth / r
    d2d_dth2 = - (dd_dth**2) / d + (deg2rad**2 * r**2) / d - deg2rad**2 * r * dd_dr

    # Time indices & argument matrix T_nm (N, M)
    N, M = A.shape
    m_idx = np.arange(-(M-1)/2, (M-1)/2 + 1)
    T_nm = m_idx[None, :] * delta_T - tau - (d[:, None] / c)

    # Gaussian pulse s(T) and derivatives
    s_nm = np.exp(-0.5 * (T_nm / sigma)**2)
    ds_nm = - (T_nm / (sigma**2)) * s_nm
    d2s_nm = ((T_nm**2 - sigma**2) / (sigma**4)) * s_nm

    # Complex matrix B (N, M)
    B = a[:, None] * s_nm

    # Scalar z = trace(A^H B) = sum(A* . B)
    z = np.sum(np.conj(A) * B)

    # Intermediate combinations
    P_nm = -1j * k * s_nm - (1.0 / c) * ds_nm
    Q_nm = - (k**2) * s_nm + (2j * k / c) * ds_nm + (1.0 / (c**2)) * d2s_nm
    R_nm = 1j * k * ds_nm + (1.0 / c) * d2s_nm

    # First derivatives of B_nm
    dB_dr = dd_dr[:, None] * a[:, None] * P_nm
    dB_dth = dd_dth[:, None] * a[:, None] * P_nm
    dB_dtau = - a[:, None] * ds_nm

    # First derivatives of z
    dz_dr = np.sum(np.conj(A) * dB_dr)
    dz_dth = np.sum(np.conj(A) * dB_dth)
    dz_dtau = np.sum(np.conj(A) * dB_dtau)

    grad_z = np.array([[dz_dr], [dz_dth], [dz_dtau]], dtype=np.complex128)

    # Second derivatives of B_nm
    d2B_dr2 = d2d_dr2[:, None] * a[:, None] * P_nm + (dd_dr[:, None]**2) * a[:, None] * Q_nm
    d2B_drdth = d2d_drdth[:, None] * a[:, None] * P_nm + (dd_dr[:, None] * dd_dth[:, None]) * a[:, None] * Q_nm
    d2B_dth2 = d2d_dth2[:, None] * a[:, None] * P_nm + (dd_dth[:, None]**2) * a[:, None] * Q_nm

    d2B_drdtau = dd_dr[:, None] * a[:, None] * R_nm
    d2B_dthdtau = dd_dth[:, None] * a[:, None] * R_nm
    d2B_dtau2 = a[:, None] * d2s_nm

    # Second derivatives of z
    d2z_dr2 = np.sum(np.conj(A) * d2B_dr2)
    d2z_drdth = np.sum(np.conj(A) * d2B_drdth)
    d2z_dth2 = np.sum(np.conj(A) * d2B_dth2)
    d2z_drdtau = np.sum(np.conj(A) * d2B_drdtau)
    d2z_dthdtau = np.sum(np.conj(A) * d2B_dthdtau)
    d2z_dtau2 = np.sum(np.conj(A) * d2B_dtau2)

    hess_z = np.array([
        [d2z_dr2,    d2z_drdth,  d2z_drdtau],
        [d2z_drdth,  d2z_dth2,   d2z_dthdtau],
        [d2z_drdtau, d2z_dthdtau, d2z_dtau2]
    ], dtype=np.complex128)

    # Final real Gradient (3x1) and Hessian (3x3)
    gradient = 2.0 * np.real(np.conj(z) * grad_z)
    hessian = 2.0 * np.real(np.conj(z) * hess_z + grad_z @ grad_z.T.conj())

    return hessian, gradient



def compute_gradient_hessian_3d_ratio(A, r, theta, tau_ratio, xy_pos, f_c, t, delta_T, sigma):
    """
    Computes 3x1 gradient and 3x3 Hessian of J(r, theta, tau_ratio) = |trace(A.H @ B)|^2,
    where tau_ratio = tau / delta_T is a relative dimensionless delay.
    """
    c = 3e8
    k = 2.0 * np.pi * f_c / c
    deg2rad = np.pi / 180.0

    # Geometry & steering vector
    th_rad = np.radians(theta)
    sin_th, cos_th = np.sin(th_rad), np.cos(th_rad)

    x_i, y_i = xy_pos[:, 0], xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = np.sqrt(dx**2 + dy**2)

    a = np.exp(-1j * k * d)

    # Distance derivatives
    dd_dr = (dx * sin_th + dy * cos_th) / d
    dd_dth = deg2rad * r * (dx * cos_th - dy * sin_th) / d

    d2d_dr2 = (1.0 - dd_dr**2) / d
    d2d_drdth = - (dd_dr * dd_dth) / d + dd_dth / r
    d2d_dth2 = - (dd_dth**2) / d + (deg2rad**2 * r**2) / d - deg2rad**2 * r * dd_dr

    # Time indices & argument matrix T_nm (N, M)
    N, M = A.shape
    m_idx = np.arange(M)
    T_nm = t - (m_idx[None, :] + tau_ratio) * delta_T - (d[:, None] / c)

    # Gaussian pulse s(T) and derivatives
    s_nm = np.exp(-0.5 * (T_nm / sigma)**2)
    ds_nm = - (T_nm / (sigma**2)) * s_nm
    d2s_nm = ((T_nm**2 - sigma**2) / (sigma**4)) * s_nm

    # Complex matrix B
    B = a[:, None] * s_nm
    z = np.sum(np.conj(A) * B)

    # Intermediate combination factors
    P_nm = -1j * k * s_nm - (1.0 / c) * ds_nm
    Q_nm = - (k**2) * s_nm + (2j * k / c) * ds_nm + (1.0 / (c**2)) * d2s_nm
    R_nm = 1j * k * ds_nm + (1.0 / c) * d2s_nm

    # 1st derivatives of B_nm wrt [r, theta, tau_ratio]
    dB_dr = dd_dr[:, None] * a[:, None] * P_nm
    dB_dth = dd_dth[:, None] * a[:, None] * P_nm
    dB_dtau_ratio = - a[:, None] * ds_nm * delta_T  # Chain rule factor: * delta_T

    # 1st derivatives of z
    dz_dr = np.sum(np.conj(A) * dB_dr)
    dz_dth = np.sum(np.conj(A) * dB_dth)
    dz_dtau_ratio = np.sum(np.conj(A) * dB_dtau_ratio)

    grad_z = np.array([[dz_dr], [dz_dth], [dz_dtau_ratio]], dtype=np.complex128)

    # 2nd derivatives of B_nm
    d2B_dr2 = d2d_dr2[:, None] * a[:, None] * P_nm + (dd_dr[:, None]**2) * a[:, None] * Q_nm
    d2B_drdth = d2d_drdth[:, None] * a[:, None] * P_nm + (dd_dr[:, None] * dd_dth[:, None]) * a[:, None] * Q_nm
    d2B_dth2 = d2d_dth2[:, None] * a[:, None] * P_nm + (dd_dth[:, None]**2) * a[:, None] * Q_nm

    d2B_drdtau_ratio = dd_dr[:, None] * a[:, None] * R_nm * delta_T
    d2B_dthdtau_ratio = dd_dth[:, None] * a[:, None] * R_nm * delta_T
    d2B_dtau_ratio2 = a[:, None] * d2s_nm * (delta_T**2)

    # 2nd derivatives of z
    d2z_dr2 = np.sum(np.conj(A) * d2B_dr2)
    d2z_drdth = np.sum(np.conj(A) * d2B_drdth)
    d2z_dth2 = np.sum(np.conj(A) * d2B_dth2)
    d2z_drdtau_ratio = np.sum(np.conj(A) * d2B_drdtau_ratio)
    d2z_dthdtau_ratio = np.sum(np.conj(A) * d2B_dthdtau_ratio)
    d2z_dtau_ratio2 = np.sum(np.conj(A) * d2B_dtau_ratio2)

    hess_z = np.array([
        [d2z_dr2,          d2z_drdth,        d2z_drdtau_ratio],
        [d2z_drdth,        d2z_dth2,         d2z_dthdtau_ratio],
        [d2z_drdtau_ratio, d2z_dthdtau_ratio, d2z_dtau_ratio2]
    ], dtype=np.complex128)

    # Real Gradient (3x1) and Hessian (3x3)
    gradient = 2.0 * np.real(np.conj(z) * grad_z)
    hessian = 2.0 * np.real(np.conj(z) * hess_z + grad_z @ grad_z.T.conj())

    return hessian, gradient