# ERP_estimation.py
# Author: Benjamin Hanson
# Date: 2/7/2024

import numpy as np
import sys
import scipy
###########################################################################################################################################
def EKF(xest0, Pest0, dt, z, Q, R, f, time, h = [], F=[], H=[], method = 'RK4', const=[]):

    # Perform Extended Kalman Filter estimation given dynamics and measurement models

    # Parameters:
    #     xest0:   Initial estimate (required)
    #     Pest0:   Initial covariance (required)
    #     dt:      Initial time step (required)
    #     z:       Measurements with epochs (required)
    #       - Measurements can be in the following two formats
    #           * z = [T] where T is the period, meaning no measurements
    #           * z = [[t1, t2, t3, ..., T],[z1,z2,z3,...,zn]] where ti are the measurement epochs and zi are the measurements
    #     Q:       Process noise matrix (required)
    #     R:       Measurement noise matrix (required)
    #     f:       Dynamics model (continuous-time, required, function handle)
    #     time:    Time frame of dynamics model, either "CT" or "DT" (required)
    #     h:       Measurement model (optional if no measurements, function handle)
    #     F:       Dynamics Jacobian (continuous-time, optional, function handle)
    #     H:       Measurement Jacobian (optional, function handle)
    #     method:  Time-marching method (optional)
    #       - 'EE':    Explicit Euler - Dynamics Jacobian is used
    #       - 'RK4':   Runge-Kutta 4 (default) - Dynamics Jacobian is estimated
    #       - 'RK45':  Adaptive Runge-Kutta 4/5 - Dynamics Jacobian is estimated
    #     const:   Miscellaneous constants (optional)

    # Outputs:
    #     xest:    Estimated states
    #     Pest:    Estimated covariances
    #     tspan:   Time span
    
    # Note: RK45 does not work currently
        
    # Checks and Balances
    [rows, columns] = np.shape(Q)
    if(rows!=columns):
        print("ERROR: Process noise matrix is not square.")
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

    if len(z)==1:
        [rows, columns] = np.shape(R)
        z = [z, [list(np.full((rows),np.nan))]]
    else:
        if h==[]:
            print("ERROR: Included measurements but missing measurement model.")
            sys.exit()

     # Estimating Dynamics Jacobian if not included based on time-marching scheme
    if(time == 'CT'):
        if(method == 'EE'):
            def df(f, x, dt, const):
                if const == []:
                    x1 = x + dt*f(x)
                else:
                    x1 = x + dt*f(x,const)
                return x1, dt
            if (F!=[]):
                def dF(f, df, F, x, dt, const):
                    if const == []:
                        return np.identity(len(x)) + dt*np.array(F(x))
                    else:
                        return np.identity(len(x)) + dt*np.array(F(x,const))
            else:
                def dF(f, df, F, x, dt, const):
                    dx = 1e-8
                    n = len(x)
                    df0, dx0 = df(f, x, dt, const)
                    jac = np.zeros((n, n))
                    for j in range(n):  
                        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                        dfi, dti = df(f,x_plus, dt, const)
                        jac[:, j] = (dfi - df0)/Dxj
                    return jac
        elif(method == 'RK4'):
            def df(f, x, dt, const):
                if const == []:
                    f1 = f(x)
                    f2 = f(x+(dt/2)*f1)
                    f3 = f(x+(dt/2)*f2)
                    f4 = f(x+dt*f3)
                else:
                    f1 = f(x,const)
                    f2 = f(x+(dt/2)*f1,const)
                    f3 = f(x+(dt/2)*f2,const)
                    f4 = f(x+dt*f3,const)
                x1 = x + dt*((f1/6)+(f2/3)+(f3/3)+(f4/6))
                return x1, dt
            def dF(f, df, F, x, dt, const):
                dx = 1e-8
                n = len(x)
                df0, dx0 = df(f, x, dt, const)
                jac = np.zeros((n, n))
                for j in range(n):  
                    Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                    x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                    dfi, dti = df(f,x_plus, dt, const)
                    jac[:, j] = (dfi - df0)/Dxj
                return jac
        elif(method == 'RK45'):
            print("ERROR: RK45 is not working yet.")
            sys.exit()
        else:
            print("ERROR: Invalid time-marching scheme.")
            sys.exit()
    elif(time == 'DT'):
        def df(f, x, dt, const):
            if const == []:
                x1 = f(x,dt)
            else:
                x1 = f(x,dt,const)
            return x1, dt
        if (F!=[]):
            def dF(f, df, F, x, dt, const):
                if const == []:
                    return F(x,dt)
                else:
                    return F(x,dt,const)
        else:
            def dF(f, df, F, x, dt, const):
                dx = 1e-8
                n = len(x)
                df0, dx0 = df(f, x, dt, const)
                jac = np.zeros((n, n))
                for j in range(n):  
                    Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                    x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                    dfi, dti = df(f,x_plus, dt, const)
                    jac[:, j] = (dfi - df0)/Dxj
                return jac

    # Estimating Measurement Jacobian if not included
    if(H == [])and(h!=[]):
        def dH(h,H,x):
            dx=1e-8
            n = len(x)
            func = h(x)
            jac = np.zeros((n, n))
            for j in range(n):  
                Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
                x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
                jac[:, j] = (h(x_plus) - func)/Dxj
            return jac
    else:
        def dH(h,H,x):
            return H(x)

    tz = z[0]
    xest = np.array([xest0])
    Pest = np.array([Pest0])
    t0 = 0
    tspan = [t0]
    for i, tk1 in enumerate(tz):
        zk1 = z[1][i] 
        t0 = tspan[-1]

        # Prediction
        dtx = dt
        while t0 < tk1:
            dtx = min(dtx, tk1-t0)
            t0 += dtx
            tspan.append(t0)
            x1, dtx = df(f,xest[-1],dtx,const)
            xest = np.concatenate((xest,[x1]),axis=0)                                                                                                                      # Estimated state      
            Pest = np.concatenate((Pest,np.array([np.linalg.multi_dot([dF(f,df,F,xest[-1],dtx,const),Pest[-1],np.transpose(dF(f,df,F,xest[-1],dtx,const))]) + Q])),axis=0) # Estimated covariance                                                                                                           

        # Correction
        if not(np.isnan(zk1[0])):
            zpred = h(xest[-1])                                                                                                                             # Predicted Measurement
            K     = np.matmul(Pest[-1],np.transpose(dH(h,H,xest[-1])),np.linalg.inv(np.matmul(dH(h,H,xest[-1]),Pest[-1],np.transpose(dH(h,H,xest[-1])))+R)) # Kalman Gain
            xest[-1] = xest[-1] + np.matmul(K,zk1-zpred)                                                                                                    # Estimated state
            Pest[-1] = np.matmul((np.identity(len(xest0))-np.matmul(K,dH(h,H,xest[-1]))),Pest[-1])                                                          # Estimated covariance

    return xest, Pest, tspan
###########################################################################################################################################
def UKF(xest0, Pest0, dt, z, Q, R, f, time, h = [], alpha = 1e-3, beta = 2, kappa = 0, method = 'RK4', const=[]):

    # Perform Extended Kalman Filter estimation given dynamics and measurement models

    # Parameters:
    #     x0:      Initial true state (required)
    #     xest0:   Initial estimate (required)
    #     Pest0:   Initial covariance (required)
    #     dt:      Initial time step (required)
    #     z:       Measurements with epochs (required)
    #       - Measurements can be in the following two formats
    #           * z = [T] where T is the period, meaning no measurements
    #           * z = [[t1, t2, t3, ..., T],[z1,z2,z3,...,zn]] where ti are the measurement epochs and zi are the measurements
    #     Q:       Process noise matrix (required)
    #     R:       Measurement noise matrix (required)
    #     f:       Dynamics model (continuous-time, required, function handle)
    #     time:    Time frame of dynamics model, either "CT" or "DT" (required)
    #     h:       Measurement model (optional if no measurements, function handle)
    #     alpha:   Alpha, spread of the sigma points around x0 (optional)
    #     beta:    Beta, incorporates prior knowledge (optional)
    #     kappa:   Kappa, secondary scaling parameter (optional)
    #     method:  Time-marching method (optional)
    #       - 'EE':    Explicit Euler - Dynamics Jacobian is used
    #       - 'RK4':   Runge-Kutta 4 (default) - Dynamics Jacobian is estimated
    #       - 'RK45':  Adaptive Runge-Kutta 4/5 - Dynamics Jacobian is estimated
    #     const:   Miscellaneous constants (optional)

    # Outputs:
    #     xest:    Estimated states
    #     Pest:    Estimated covariances
    #     tspan:   Time span
        
    # Checks and Balances
    [rows, columns] = np.shape(Q)
    if(rows!=columns):
        print("ERROR: Process noise matrix is not square.")
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

    if len(z)==1:
        [rows, columns] = np.shape(R)
        z = [z, [list(np.full((rows),np.nan))]]
    else:
        if h==[]:
            print("ERROR: Included measurements but missing measurement model.")
            sys.exit()

     # Estimating Dynamics Jacobian if not included based on time-marching scheme
    if(time == 'CT'):
        if(method == 'EE'):
            def df(f, x, dt, const):
                x1 = x + dt*f(x,const)
                return x1, dt
        elif(method == 'RK4'):
            def df(f, x, dt, const):
                f1 = f(x,const)
                f2 = f(x+(dt/2)*f1,const)
                f3 = f(x+(dt/2)*f2,const)
                f4 = f(x+dt*f3,const)
                x1 = x + dt*((f1/6)+(f2/3)+(f3/3)+(f4/6))
                return x1, dt
        elif(method == 'RK45'):
            print("ERROR: RK45 is not working yet.")
            sys.exit()
        else:
            print("ERROR: Invalid time-marching scheme.")
            sys.exit()
    elif(time == 'DT'):
         def df(f, x, dt, const):
            if const == []:
                x1 = f(x,dt)
            else:
                x1 = f(x,dt,const)
            return x1, dt

    # Weights and Sigma Points
    n   = len(xest0)               # Dimension of x0    
    n2  = np.shape(R)[0]           # Dimension of measurement
    lam = alpha**2*(n+kappa)-n     # Lambda
    Wm  = np.full(2*n+1,np.nan)    # Weights, m
    Wc  = np.full(2*n+1,np.nan)    # Weights, c

    Wm[0] = lam/(n + lam)
    Wc[0] = (lam/(n + lam)) + (1-alpha**2+beta)
    for i in range(1,2*n+1):
        Wm[i] = 1/(2*(n + lam))
        Wc[i] = Wm[i]

    tz = z[0]
    xest = np.array([xest0])
    Pest = np.array([Pest0])
    t0 = 0
    tspan = [t0]
    for i, tk1 in enumerate(tz):
        zk1 = z[1][i] 
        t0 = tspan[-1]

        # Prediction
        dtx = dt
        while t0 < tk1:
            dtx = min(dtx, tk1-t0)
            t0 += dtx
            tspan.append(t0)

            # Calculate Sigma Points
            A = np.full((2*n+1,n),np.nan)
            L = scipy.linalg.sqrtm((n + lam) * Pest[i-1])
            A[0] = xest[-1]
            for j in range(1,n+1):
                A[j] = xest[-1] + L[j-1]
            for j in range(n+1, 2*n+1):
                A[j] = xest[-1] - L[j-1-n]

            # Prediction
            for j in range(0,2*n+1):
                dfi, dti = df(f,A[j],dt,const)
                A[j] = dfi                               # Sigma Points

            xest = np.concatenate((xest,[np.zeros(n)]),axis=0)   
            for j in range(0, 2*n+1):
                xest[-1] += Wm[j]*A[j]                                       # Mean

            Pest = np.concatenate((Pest,np.array([Q])),axis=0) # Estimated covariance                                                                                                           
            for j in range(0,2*n+1):
                Pest[-1] = Pest[-1] + Wc[j]*np.outer((A[j]-xest[-1]),(A[j]-xest[-1])) # Covariance
            
        # Correction
        if not(np.isnan(zk1[0])):
            Z = np.full((2*n+1,n2),np.nan)                  
            for j in range(0,2*n+1):
                Z[j] = h(A[j])                                            # Sigma measurements
            zpred = np.zeros(n2)
            for j in range(0,2*n+1):
                zpred += Wm[j]*Z[j]                                       # Measurement
            
            P_zz = np.zeros([n2,n2])
            for j in range(0,2*n+1):
                P_zz += Wc[j]*np.outer((Z[j]-zpred),(Z[j]-zpred))

            P_xz = np.zeros([n,n2])
            for j in range(0,2*n+1):
                P_xz += Wc[j]*np.outer((A[j]-xest[-1]),(Z[j]-zpred))

            S = R + P_zz
            K = np.linalg.multi_dot([P_xz,np.linalg.inv(S)])
            xest[-1] = xest[-1] + np.linalg.multi_dot([K,(zk1-zpred)])
            Pest[-1] = Pest[-1] - np.linalg.multi_dot([K,S,np.transpose(K)])
    
    return xest, Pest, tspan
###########################################################################################################################################def plot_gaussian_ellipsoid(m, C, sd=1, p=[], ax=[]):
def plot_gaussian_ellipsoid(m, C, sd, p, ax):

    # Plot Gaussian ellipsoids representative of a 2D/3D covariance matrix

    # Parameters:
    #     m:   Mean
    #     C:   Covariance
    #     sd:  Number of standard deviations 
    #     ax:  Figure axis 
    #     p:   Plot parameters (optional)
    #         -type:    'fill' for filled in ellipse, 'line' for line plot ellipse
    #         -display: print to legend (1) or not (else)
    #         -name:    If display==1, name added to legend
    #         -color:   If 'line' type, color of line
    #         -lw:      If 'line' type, linewidth of ellipse
    #         -ls:      If 'line' type, line style of ellipse
    #         - means:  Plot means
    # Outputs:
    #     Plots Gaussian covariance matrix as an ellipse

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
def get_measurements(x0, dt, t, f, h, Q, R, time, method = 'RK4', const = []):

    # Given a dynamics model, initial state, and time parameters, return the
    # true states and measurements over time.
     
    # Inputs: 
    #    x0:      Initial true state (required)
    #    dt:      Time step (required)
    #    t:       Time parameter (required)
    #        - Time parameter can be in the following two fortmats:
    #            * t = [T] where T is period
    #            * t = [t1, t2, ..., T] where ti are the epochs
    #    f:       Dynamics model (required, continuous-time, function handle)
    #    h:       Measurement model (required, function handle)
    #    Q:       Process noise matrix (required)
    #    R:       Measurement noise matrix (required)
    #    time:    Time frame of dynamics model, either "CT" or "DT" (required)
    #    method:  Time-marching method (optional)
    #        - 'EE':    Explicit Euler
    #        - 'RK4':   Runge-Kutta 4 (default)
    #        - 'RK45':  Adaptive Runge-Kutta 4/5
    #    const:   Miscellaneous constants (optional)

    #  Outputs:
    #    x:       True states
    #    z:       Measurements
    #    tx:      True time span
    #    tz:      Measurement time span

    if(time == 'CT'):
        if(method == 'EE'):
            def df(f, x, dt, const):
                if const == []:
                    x1 = x + dt*f(x)
                else:
                    x1 = x + dt*f(x,const)
                return x1, dt
        elif(method == 'RK4'):
            def df(f, x, dt, const):
                if const == []:
                    f1 = f(x)
                    f2 = f(x+(dt/2)*f1)
                    f3 = f(x+(dt/2)*f2)
                    f4 = f(x+dt*f3)
                else:
                    f1 = f(x,const)
                    f2 = f(x+(dt/2)*f1,const)
                    f3 = f(x+(dt/2)*f2,const)
                    f4 = f(x+dt*f3,const)
                x1 = x + dt*((f1/6)+(f2/3)+(f3/3)+(f4/6))
                return x1, dt
        elif(method == 'RK45'):
            print("ERROR: RK45 is not working yet.")
            sys.exit()
        else:
            print("ERROR: Invalid time-marching scheme.")
            sys.exit()
    elif(time == 'DT'):
        def df(f, x, dt, const):
            if const == []:
                x1 = f(x,dt)
            else:
                x1 = f(x,dt,const)
            return x1, dt

    dtz = dt
    if len(t)==1:
        T = t[0]
        t = []
        t0 = 0
        while (t0 < T):
            t0 += dtz
            t.append(t0)
            dtz = min(dtz, T-t0)

    tx = [0]
    x = np.array([x0])
    tz = t
    t0 = 0
    cx = 0
    for i, tk1 in enumerate(tz):

        dtx = dt
        # True State
        while t0 < tk1:
            dtx = min(dtx, tk1-t0)
            t0 += dtx
            tx.append(t0)
            w  = np.linalg.multi_dot([np.sqrt(Q),np.random.randn(len(x0))])
            x1, dtx = df(f,x[cx,:],dtx,const)
            x = np.concatenate((x,[x1+w]),axis=0) 
            cx += 1

        # Measurement 
        d = R.shape[0]
        v = np.linalg.multi_dot([np.sqrt(R),np.random.randn(d)])
        if i==0:
            z = [h(x[cx,:]) + v]
        else:
            z = np.concatenate((z,[h(x[cx,:]) + v]),axis=0)

    return np.array(x), np.array(z), np.array(tx), np.array(tz)
###########################################################################################################################################
