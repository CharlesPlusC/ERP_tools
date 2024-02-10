"""
ERP_estimation.py

Author: Benjamin Hanson

Date: 2/7/2024

Miscellaneous estimation functions used for ERP research
"""

import numpy as np
import sys
import scipy
###########################################################################################################################################
def EKF(x0, xest0, Pest0, dt, T, Q, R, f, h, F=[], H=[], const=[]):
    """
    Perform Extended Kalman Filter estimation given dynamics and measurement models

    Parameters:
        x0:      Initial true state
        xest0:   Initial estimate
        Pest0:   Initial covariance
        T:       Period of propagation
        dt:      Time step or [True timestep, Measurement timestep]
        Q:       Process noise matrix
        R:       Measurement noise matrix
        f:       Dynamics model
        h:       Measurement model
        F:       Dynamics Jacobian (optional, finite difference approximation if not included)
        H:       Measurement Jacobian (optional, finite difference approximation if not included)
        const:   Miscellaneous constants (optional)

    Outputs:
        x:       True states
        xest:    Estimated states
        z:       Measurements
        Pest:    Estimated covariances
        tspan_x: True time span
        tspan_z: Measurement time span
    """

    # Checks and Balances
    [rows, columns] = np.shape(Q)
    if(rows!=columns):
        print("ERROR: Process noise matrix is not square.")
        sys.exit()

    if(len(xest0)!=len(x0)):
        print(xest0)
        print(x0)
        print(len(xest0))
        print(len(x0))
        print("ERROR: Initial true state and initial estimate state dimensions do not match.")
        sys.exit()

    if(len(xest0)!=rows):
        print("ERROR: State vector and process noise matrix are not compatible dimensions.")
        sys.exit()

    [rows, columns] = np.shape(R)
    if(rows!=columns):
        print("ERROR: Measurement noise matrix is not square.")
        sys.exit()

    [rows, columns] = np.shape(Pest0)
    if(rows!=columns):
        print("ERROR: Covariance matrix is not square.")
        sys.exit()

    if(len(xest0)!=rows):
        print("ERROR: State vector and covariance matrix are not compatible dimensions.")
        sys.exit()

    if not(np.allclose(Pest0, Pest0.T, rtol=1e-05, atol=1e-05)):
        print("ERROR: Covariance matrixx is not symmetric.")
        sys.exit()

    if not(callable(f)):
        print("ERROR: Input f must be callable.")
        sys.exit()

    if not(callable(h)):
        print("ERROR: Input h must be callable.")
        sys.exit()

    # Timespans
    if(type(dt) is list)or(type(dt) is np.ndarray):
        if(float(dt[1])/dt[0] % 1 != 0):
            print("Measurement time step is not a multiple of true time step. Rounding to nearest multiple.")
            dt[1] = round(dt[1]/dt[0])*dt[0]; 
            
        tspan_x = np.arange(0,T+dt[0],dt[0])
        tspan_z = np.arange(dt[1],T+dt[1],dt[1])
        dt_ratio = dt[1]/dt[0]
        dt = dt[0]
    else:
        tspan_x = np.arange(0,T+dt,dt)  # Defining true state timespan
        tspan_z = np.arange(dt,T+dt,dt) # Defining measurement timespan
        dt_ratio = 1

    # Approximating Jacobians 
    if(F == []):
        def F(x, dt, const):
            dx=1e-8
            n = len(x)
            func = f(x, dt, const)
            jac = np.zeros((n, n))
            for j in range(n):  
                Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                jac[:, j] = (f(x_plus, dt, const) - func)/Dxj
            return jac

    if(H == []):
        def H(x):
            dx=1e-8
            n = len(x)
            func = h(x)
            jac = np.zeros((n, n))
            for j in range(n):  
                Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                jac[:, j] = (h(x_plus) - func)/Dxj
            return jac

    d1      = len(Q)                      # Dimension of process noise matrix
    d2      = len(R)                      # Dimension of measurement noise matrix
    n1      = len(tspan_x)                # Length of true timespan
    n2      = len(tspan_z)                # Length of measurement timespan
    x       = np.full((n1,d1),np.nan)     # True states 
    x[0]    = x0                          # Initializing first true state
    xest    = np.full((n1,d1),np.nan)     # Estimated states
    xest[0] = xest0                       # Initialized first estimated state
    Pest    = np.full((n1,d1,d1),np.nan)  # Estimated covariance
    Pest[0] = Pest0                       # Initializing first estimated covariance
    z       = np.full((n2,d2),np.nan)     # Measurements
    count   = 0
    for i in range(1,n1):

        # Initialization
        w    = np.linalg.multi_dot([np.sqrt(Q),np.random.randn(d1)])    # Process noise
        x[i] = f(x[i-1],dt,const) + w                                   # True state

        # Prediction
        xpred = f(xest[i-1],dt,const)                                                                             # Predicted state
        Ppred = np.linalg.multi_dot([F(xest[i-1],dt,const),Pest[i-1],np.transpose(F(xest[i-1],dt,const))]) + Q    # Predicted covariance
        zpred = h(xpred)                                                                                          # Predicted measurement

        # Correction
        if((i) % dt_ratio == 0):
            v       = np.matmul(np.sqrt(R), np.random.randn(d2))                                                                   # Measurement noise
            z[count]  = h(x[i]) + v                                                                                                # Actual measurement
            K       = np.matmul(Ppred,np.transpose(H(xpred)),np.linalg.inv(np.matmul(H(xpred),Ppred,np.transpose(H(xpred)))+R))    # Kalman Gain
            xest[i] = xpred + np.matmul(K,z[count]-zpred)                                                                          # Estimated state
            Pest[i] = np.matmul((np.identity(d1)-np.matmul(K,H(xpred))),Ppred)                                                     # Estimated covariance
            count += 1
        else:
            xest[i] = xpred                                                                
            Pest[i] = Ppred       

    return x, xest, z, Pest, tspan_x, tspan_z
###########################################################################################################################################
def UKF(x0, xest0, Pest0, dt, T, Q, R, f, h, const=[], alpha = [], beta = [], kappa = []):
    """
    Perform Extended Kalman Filter estimation given dynamics and measurement models

    Parameters:
        x0:      Initial true state
        xest0:   Initial estimate
        Pest0:   Initial covariance
        T:       Period of propagation
        dt:      Time step or [True timestep, Measurement timestep]
        Q:       Process noise matrix
        R:       Measurement noise matrix
        f:       Dynamics model
        h:       Measurement model
        const:   Miscellaneous constants (optional)
        alpha:   Alpha, spread of the sigma points around x0 (optional)
        beta:    Beta, incorporates prior knowledge (optional)
        kappa:   Kappa, secondary scaling parameter (optional)


    Outputs:
        x:       True states
        xest:    Estimated states
        z:       Measurements
        Pest:    Estimated covariances
        tspan_x: True time span
        tspan_z: Measurement time span
    """

    # Checks and Balances
    [rows, columns] = np.shape(Q)
    if(rows!=columns):
        print("ERROR: Process noise matrix is not square.")
        sys.exit()

    if(len(xest0)!=len(x0)):
        print(xest0)
        print(x0)
        print(len(xest0))
        print(len(x0))
        print("ERROR: Initial true state and initial estimate state dimensions do not match.")
        sys.exit()

    if(len(xest0)!=rows):
        print("ERROR: State vector and process noise matrix are not compatible dimensions.")
        sys.exit()

    [rows, columns] = np.shape(R)
    if(rows!=columns):
        print("ERROR: Measurement noise matrix is not square.")
        sys.exit()

    [rows, columns] = np.shape(Pest0)
    if(rows!=columns):
        print("ERROR: Covariance matrix is not square.")
        sys.exit()

    if(len(xest0)!=rows):
        print("ERROR: State vector and covariance matrix are not compatible dimensions.")
        sys.exit()

    if not(np.allclose(Pest0, Pest0.T, rtol=1e-05, atol=1e-05)):
        print("ERROR: Covariance matrixx is not symmetric.")
        sys.exit()

    if not(callable(f)):
        print("ERROR: Input f must be callable.")
        sys.exit()

    if not(callable(h)):
        print("ERROR: Input h must be callable.")
        sys.exit()

    # UT Scaling Parameters
    if(alpha==[]): alpha = 1e-3
    if(beta==[]):  beta  = 2
    if(kappa==[]): kappa = 0

    # Timespans
    if(type(dt) is list)or(type(dt) is np.ndarray):
        if(float(dt[1])/dt[0] % 1 != 0):
            print("Measurement time step is not a multiple of true time step. Rounding to nearest multiple.")
            dt[1] = round(dt[1]/dt[0])*dt[0]; 
            
        tspan_x = np.arange(0,T+dt[0],dt[0])
        tspan_z = np.arange(dt[1],T+dt[1],dt[1])
        dt_ratio = dt[1]/dt[0]
        dt = dt[0]
    else:
        tspan_x = np.arange(0,T+dt,dt) # Defining true state timespan
        tspan_z = np.arange(dt,T+dt,dt) # Defining measurement timespan
        dt_ratio = 1

    d1      = len(Q)                      # Dimension of process noise matrix
    d2      = len(R)                      # Dimension of measurement noise matrix
    n1      = len(tspan_x)                # Length of true timespan
    n2      = len(tspan_z)                # Length of measurement timespan
    x       = np.full((n1,d1),np.nan)     # True states 
    x[0]    = x0                          # Initializing first true state
    xest    = np.full((n1,d1),np.nan)     # Estimated states
    xest[0] = xest0                       # Initialized first estimated state
    Pest    = np.full((n1,d1,d1),np.nan)  # Estimated covariance
    Pest[0] = Pest0                       # Initializing first estimated covariance
    z       = np.full((n2,d2),np.nan)     # Measurements
    count   = 0

    # Weights and Sigma Points
    n   = len(x0)                  # Dimension of x0          
    lam = alpha**2*(n+kappa)-n     # Lambda
    Wm  = np.full(2*n+1,np.nan)    # Weights, m
    Wc  = np.full(2*n+1,np.nan)    # Weights, c

    Wm[0] = lam/(n + lam)
    Wc[0] = (lam/(n + lam)) + (1-alpha**2+beta)
    for i in range(1,2*n+1):
        Wm[i] = 1/(2*(n + lam))
        Wc[i] = Wm[i]

    for i in range(1,n1):
        # Initialization
        w    = np.linalg.multi_dot([np.sqrt(Q),np.random.randn(d1)])    # Process noise
        x[i] = f(x[i-1],dt,const) + w                                   # True state

        # Calculate Sigma Points
        A = np.full((2*n+1,n),np.nan)
        L = scipy.linalg.sqrtm((n + lam) * Pest[i-1])
        A[0] = xest[i-1]
        for j in range(1,n+1):
            A[j] = xest[i-1] + L[j-1]
        for j in range(n+1, 2*n+1):
            A[j] = xest[i-1] - L[j-1-n]

        # Prediction
        for j in range(0,2*n+1):
            A[j] = f(A[j], dt, const)                                 # Sigma Points
        xpred = np.zeros(d1)
        for j in range(0, 2*n+1):
            xpred += Wm[j]*A[j]                                       # Mean
        Ppred = Q  
        for j in range(0,2*n+1):
            Ppred = Ppred + Wc[j]*np.outer((A[j]-xpred),(A[j]-xpred)) # Covariance
        Z = np.full((2*n+1,d2),np.nan)                  
        for j in range(0,2*n+1):
            Z[j] = h(A[j])                                            # Sigma measurements
        zpred = np.zeros(d2)
        for j in range(0,2*n+1):
            zpred += Wm[j]*Z[j]                                       # Measurement
        
        # Correction
        if((i) % dt_ratio == 0):
            v        = np.matmul(np.sqrt(R), np.random.randn(d2))   # Measurement noise
            z[count] = h(x[i]) + v                                  # Actual measurement

            P_zz = np.zeros([d2,d2])
            for j in range(0,2*n+1):
                P_zz += Wc[j]*np.outer((Z[j]-zpred),(Z[j]-zpred))

            P_xz = np.zeros([d1,d2])
            for j in range(0,2*n+1):
                P_xz += Wc[j]*np.outer((A[j]-xpred),(Z[j]-zpred))

            S = R + P_zz
            K = np.linalg.multi_dot([P_xz,np.linalg.inv(S)])
            xest[i] = xpred + np.linalg.multi_dot([K,(z[count]-zpred)])
            Pest[i] = Ppred - np.linalg.multi_dot([K,S,np.transpose(K)])
            count += 1
        else:
            xest[i] = xpred                                                                
            Pest[i] = Ppred  

    return x, xest, z, Pest, tspan_x, tspan_z
###########################################################################################################################################
def plot_gaussian_ellipsoid(m, C, sd=1, p=[], ax=[]):
    """
    Plot Gaussian ellipsoids representative of a 2D/3D covariance matrix

    Parameters:
        m:   Mean
        C:   Covariance
        sd:  Number of standard deviations 
        ax:  Figure axis 
        p:   Plot parameters (optional)
            -type:    'fill' for filled in ellipse, 'line' for line plot ellipse
            -display: print to legend (1) or not (else)
            -name:    If display==1, name added to legend
            -color:   If 'line' type, color of line
            -lw:      If 'line' type, linewidth of ellipse
            -ls:      If 'line' type, line style of ellipse
            - means:  Plot means
    Outputs:
        Plots Gaussian covariance matrix as an ellipse
    """
    if(p == []):
        class ellipse: # Plotting Gaussians Ellipsoids
            def __init__(self, display, name, color, lw, ls, means):
                self.display = display
                self.name = name
                self.color = color
                self.lw = lw
                self.ls = ls
                self.means = means
        if(len(m))==2:
            p = ellipse(0, [], (0, 0, 1), 1, '-', 0)
        else:
            p = ellipse(0, [], (0, 0, 1, 0.2), [], [], 0)

    if(len(m)==2):
        npts = 50
        tt = np.linspace(0,2*np.pi, npts)
        x = np.cos(tt)
        y = np.sin(tt)
        ap = np.vstack([x,y])
        d, v = np.linalg.eig(C)
        d = np.diag(d)
        v = sd * np.sqrt(v)
        bp = np.linalg.multi_dot([v,d,ap])+np.transpose(np.tile(m,(np.size(ap[1]),1)))
        if(p.display):
            ax.plot(bp[0,:], bp[1,:], color = p.color, lw = p.lw, ls = p.ls, label = p.name)
        else:
            ax.plot(bp[0,:], bp[1,:], color = p.color)
        if(p.means):
            ax.scatter(m[0], m[1], s = 50,  marker = 'o', color = p.color)
    elif(len(m)==3):
        npts = 20
        theta = np.linspace(-npts,npts,(npts+1))/npts*np.pi
        phi = np.linspace(-npts,npts,(npts+1))/npts*np.pi/2
        cosphi = np.cos(phi)
        cosphi[0] = 0
        cosphi[npts] = 0
        sintheta = np.sin(theta)
        sintheta[0] = 0
        sintheta[npts] = 0
        x = np.outer(cosphi,np.cos(theta))
        y = np.outer(cosphi, sintheta)
        z = np.outer(np.sin(phi),np.full((1,npts+1),1))
        ap = np.vstack([np.ndarray.flatten(np.transpose(x)),np.ndarray.flatten(np.transpose(y)),np.ndarray.flatten(np.transpose(z))])
        d, v = np.linalg.eig(C)
        d = np.diag(d)
        d = sd * np.sqrt(d)
        bp = np.linalg.multi_dot([v,d,ap])+np.transpose(np.tile(m,(np.size(ap[1]),1)))
        xp = np.transpose(np.reshape(bp[0,:], np.shape(x)))
        yp = np.transpose(np.reshape(bp[1,:], np.shape(y)))
        zp = np.transpose(np.reshape(bp[2,:], np.shape(z)))
        ax.plot_surface(xp, yp, zp, color=p.color)
        if(p.means):
            ax.scatter(m[0], m[1], m[2], s = 50, marker = 'o', color = p.color)
    else:
        print("PLOT_GAUSSIAN_ELLIPSOIDS requires 2D or 3D means/covariances.")
        sys.exit()

    return
###########################################################################################################################################

