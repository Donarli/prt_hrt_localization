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