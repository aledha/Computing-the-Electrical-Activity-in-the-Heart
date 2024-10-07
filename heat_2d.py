import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay 
from scipy.sparse import coo_array
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def area(p, triangle):
    points = p[triangle]
    x, y = points[:,0], points[:,1]
    return 1/2 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))

def buildmatrices(p, t, dt, theta):
    m = np.size(t,0)
    n = np.size(p,0)

    Arow = []
    Acol = []
    Adata = []
    F = np.zeros((m,3,3))

    for k in range(m):
        T_k = area(p, t[k])
        x_loc = p[t[k]]
        V = np.hstack((np.ones((3,1)), x_loc))
        C = np.linalg.inv(V)
        H = lambda x, alpha: C[0, alpha] + C[1, alpha]*x[0] + C[2, alpha]*x[1]
        xg = x_loc/2 + np.sum(x_loc, 0)/6

        for alpha in range(3):
            for beta in range(3):
                Arow.append(t[k,alpha])
                Acol.append(t[k,beta]) 
                Adata.append(T_k * (np.sum([H(x, alpha) * H(x,beta) for x in xg])/3
                        + dt * theta * (C[1,alpha]*C[1,beta] + C[2,alpha]*C[2,beta])))
                F[k, alpha, beta] = T_k * (np.sum([H(x, alpha) * H(x,beta) for x in xg])/3
                                            - dt * (1-theta) * (C[1,alpha]*C[1,beta] + C[2,alpha]*C[2,beta]))
    A = coo_array((Adata, (Arow, Acol)))
    return A, F


def heat(p, t, dt, theta, v0, T):
    m = np.size(t,0)
    n = np.size(p,0)
    timesteps = round(T/dt)
    A, F = buildmatrices(p, t, dt, theta)
    vs = np.zeros((timesteps+1, n))
    vs[0] = v0
    for step in range(timesteps):
        f = np.zeros(n)
        for k in range(0, m):
            for alpha in range(3):
                f[t[k, alpha]] += np.sum([v0[t[k,beta]] * F[k,alpha,beta] for beta in range(3)])
        v = spsolve(A.tocsc(), f)
        v0 = v.copy()
        vs[step+1] = v 
    return vs

def animate_solution(p, vs):
    timesteps = vs.shape[0]
    fig,ax = plt.subplots()

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    def animate(i):
        ax.clear()
        tf = ax.tricontourf(p[:,0], p[:,1], vs[i])
        ax.set_title('%03d'%(i)) 
        cax.cla()
        fig.colorbar(tf, cax=cax)

    ani = animation.FuncAnimation(fig, animate, timesteps, blit=False)

    plt.show()

def v_exact(x,y,t):  
   return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.exp(-8*np.pi**2 *t)

def test_heat(N, T, dt, theta):
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xx, yy = np.meshgrid(x,y)
    p = np.vstack((xx.flatten(), yy.flatten())).T
    
    tri = Delaunay(p) 
    t = tri.simplices

    v0 = np.cos(2*np.pi*p[:,0]) * np.cos(2*np.pi*p[:,1])
    vs = heat(p, t, dt, theta, v0, T)

    ts = np.arange(0, T+dt, dt)
    errors = np.zeros(len(ts))
    for i in range(len(ts)):
        errors[i] = np.max(np.abs(vs[i]-v_exact(p[:,0], p[:,1], ts[i])))
    #print(errors)

    animate_solution(p, vs)

test_heat(50, 0.5, 0.005, 0.5)

##plt.tricontourf(p[:,0],p[:,1], v)

#plt.show()