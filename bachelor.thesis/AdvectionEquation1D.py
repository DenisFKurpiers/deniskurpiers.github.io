import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Parametry
velocity = 8
t_start = 0.0
t_end = 1.0
#body
nx = 200 
#delka                     
L = 2.0 
#delka kousicku                    
dx = L / (nx - 1)             

# Initial conditions
def set_initial_condition(nx, dx, condition_type="discontinuous"):
    q = np.zeros(nx)
    for i in range(nx):
        x = i * dx
        if condition_type == "discontinuous":
            q[i] = 1.0 if 0.4 < x < 0.6 else 0.0 
        elif condition_type == "continuous":
            q[i] = np.exp(-100 * (x - 0.5)**2)
    return q

def compute_spatial_derivatives(t, q, nx, dx, velocity, scheme):
    dqdt = np.zeros(nx)
    
    if scheme == "fdm":
        for i in range(1, nx - 1):
            dqdt[i] = -velocity * (q[i + 1] - q[i - 1]) / (2 * dx)
        dqdt[0] = 0.0
        dqdt[nx - 1] = 0.0
                       
    elif scheme == "upwind":
        if velocity > 0:
            for i in range(1, nx):
                dqdt[i] = -velocity * (q[i] - q[i - 1]) / dx
            dqdt[0] = 0.0
        else:
            for i in range(0, nx - 1):
                dqdt[i] = -velocity * (q[i + 1] - q[i]) / dx
            dqdt[nx - 1] = 0.0
                
    return dqdt

def euler_step(t, q, nx, dx, velocity, scheme, dt):
    dqdt = compute_spatial_derivatives(t, q, nx, dx, velocity, scheme)
    return q + dt * dqdt

def solve_loop(t_start, t_end, dt, nx, dx, velocity, scheme, q_initial):
    solution = []
    current_time = t_start
    q_current = q_initial.copy()
    
    solution.append((current_time, q_current.copy()))
    
    while current_time < t_end:
        q_current = euler_step(current_time, q_current, nx, dx, velocity, scheme, dt)
        current_time += dt
        solution.append((current_time, q_current.copy()))
        
    return solution

if __name__ == "__main__":
    scheme = "upwind"
    cond_type = "discontinuous" 
    
    #natvrdo cfl
    cfl_number = 1.0 #cfl=1 funguje perfektne, jeste aby ne
    dt = cfl_number * dx / abs(velocity)
    
    q = set_initial_condition(nx, dx, condition_type=cond_type)
    M = np.max(q) if np.max(q) > 0 else 1.0
    
    global_solution = solve_loop(
        t_start,
        t_end,
        dt,
        nx,
        dx,
        velocity,
        scheme,
        q
    )

    t_vals, q_vals = zip(*global_solution)
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.linspace(0, L, nx), q_vals[0])
    ax.set_ylim(-0.1, 1.2 * M)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mass / State (q)")

    def animate(i):
        line.set_ydata(q_vals[i])
        return (line,)

    ani = animation.FuncAnimation(fig, animate, frames=len(t_vals), interval=30, blit=True)
    plt.show()