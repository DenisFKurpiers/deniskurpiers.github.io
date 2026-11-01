import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parametry
velocity = 4.0
initial_time = 0.0
final_time = 1.0
# pocet kousku
size = 200
# delka objektu
L = 2.0
# delka mezi dvema body
h = L / (size - 1)                    

# pocatecni podminky
def set_initial_condition(size, h, condition_type="discontinuous"):
    u = np.zeros(size)
    for i in range(size):
        x = i * h
        if condition_type == "discontinuous":
            
            u[i] = 1.0 if 0.4 < x < 0.6 else 0.0
        elif condition_type == "continuous":
            
            u[i] = np.exp(-100 * (x - 0.5)**2)
    return u

# du/dt = fu
def function_f(t, u, size, h, velocity, scheme, tau):
    fu = np.zeros(size)
    
    # okrajovy podmínky (Dirichlet)
    u[0] = 0.0
    u[size - 1] = 0.0

    if scheme == "fdm":
        for i in range(1, size - 1):
            fu[i] = -velocity * (u[i + 1] - u[i - 1]) / (2 * h)
                       
    elif scheme == "upwind":
        for i in range(1, size - 1):
            if velocity > 0:
                fu[i] = -velocity * (u[i] - u[i - 1]) / h
            else:
                fu[i] = -velocity * (u[i + 1] - u[i]) / h
                
    return fu

def euler_step(t, u, size, h, velocity, scheme, tau):
    fu = function_f(t, u, size, h, velocity, scheme, tau)
    return u + tau * fu

def solve_loop(initial_time, final_time, time_step, size, h, velocity, scheme, u_initial):
    solution = []
    current_time = initial_time
    u_current = u_initial.copy()
    
    solution.append((current_time, u_current.copy()))
    
    while current_time < final_time:
        u_current = euler_step(current_time, u_current, size, h, velocity, scheme, time_step)
        current_time += time_step
        solution.append((current_time, u_current.copy()))
        
    return solution


if __name__ == "__main__":
    scheme = "upwind"
    cond_type = "continuous" 
    
    integration_time_step = 0.1 * L / size
    
    # Set up problem
    u = set_initial_condition(size, h, condition_type=cond_type)
    M = np.max(u) if np.max(u) > 0 else 1.0
    
    global_solution = solve_loop(
        initial_time,
        final_time,
        integration_time_step,
        size,
        h,
        velocity,
        scheme,
        u
    )

    t_vals, u_vals = zip(*global_solution)
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.linspace(0, L, size), u_vals[0])
    ax.set_ylim(-0.1, 1.2 * M)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mass")

    def animate(i):
        line.set_ydata(u_vals[i])
        return (line,)

    ani = animation.FuncAnimation(fig, animate, frames=len(t_vals), interval=50, blit=True)
    plt.show()