
#upe stejny jako 1D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

import sys
from os.path import dirname

sys.path.append("..")

from Euler import Euler
from Merson import Merson
from RungeKutta import RK_second_order
from ODE.ODE import *

# 2D, takze chci x,y, jinak anal.
velocity_x = 3.0
velocity_y = 3.0
initial_time = 0.0
final_time = 1
size = 100
L = 2
integration_time_step = 0.1 * L / size
time_step = integration_time_step

global_solution = None

#scheme = "fdm"
#scheme = "lax-fridrichs"
scheme = "upwind"


# array misto line, 
def animate(i):
    t, u = zip(*global_solution)
    im.set_array(u[i])
    return (im,)


class ExplicitTransportEquationProblem2D:
    def __init__(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def get_degrees_of_freedom(self):
        # 2D 
        return self.size * self.size

    def set_parameters(self, size, L):
        self.size = size
        self.h = L / (size - 1)

  
    def set_initial_condition(self, u):
        for i in range(self.size):
            for j in range(self.size):
                y = i * self.h
                x = j * self.h
                # tady jsem chtel udelat kolecko
                u[i, j] = 1 if (x - 0.5)**2 + (y - 0.5)**2 < 0.15**2 else 0.0


    def function_f(self, t, u):
        #fu je casova derivace
        fu = np.zeros((self.size, self.size))
        
        #bacha u[a,b], kde a==radek, b==sloupec
        u[0, :] = 0.0
        u[self.size - 1, :] = 0.0
        u[:, 0] = 0.0
        u[:, self.size - 1] = 0.0

        tau = integration_time_step
        if scheme == "fdm":
            for i in range(1, self.size - 1):
                for j in range(1, self.size - 1):
                    #anal co 1D pro x,y
                    flow_x = -velocity_x * (u[i, j + 1] - u[i, j - 1]) / (2 * self.h)
                    flow_y = -velocity_y * (u[i + 1, j] - u[i - 1, j]) / (2 * self.h)
                    #superpozice
                    fu[i, j] = flow_x + flow_y
                    
        elif scheme == "lax-fridrichs":
            for i in range(1, self.size - 1):
                for j in range(1, self.size - 1):
                    #anal co 1D pro x,y
                    flow_x = -velocity_x * (u[i, j + 1] - u[i, j - 1]) / (2 * self.h) + (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / (2 * tau)
                    flow_y = -velocity_y * (u[i + 1, j] - u[i - 1, j]) / (2 * self.h) + (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (2 * tau)
                    #superpozice
                    fu[i, j] = flow_x + flow_y
                    
        elif scheme == "upwind":
            for i in range(1, self.size - 1):
                for j in range(1, self.size - 1):
                    # x
                    if velocity_x > 0:
                        flow_x = -velocity_x * (u[i, j] - u[i, j - 1]) / self.h
                    else:
                        flow_x = -velocity_x * (u[i, j + 1] - u[i, j]) / self.h
                        
                    # y
                    if velocity_y > 0:
                        flow_y = -velocity_y * (u[i, j] - u[i - 1, j]) / self.h
                    else:
                        flow_y = -velocity_y * (u[i + 1, j] - u[i, j]) / self.h
                        
                    fu[i, j] = flow_x + flow_y
                    
        return fu


if __name__ == "__main__":

    problem = ExplicitTransportEquationProblem2D(size, L)

    #uprava na 2D, jinak opsat
    u = np.zeros((size, size))
    problem.set_initial_condition(u)
    M = np.max(u)

    integrator = Euler()
    global_solution = solve_loop(
        initial_time,
        final_time,
        time_step,
        integration_time_step,
        problem,
        integrator,
        u,
    )

    #tomuhle nerozumim, Gemini ocividne jo
    t, u = zip(*global_solution)
    fig, ax = plt.subplots()
    im = ax.imshow(u[0], cmap="magma", origin="lower", extent=[0, L, 0, L], vmin=0, vmax=1.1 * M)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Transport Equation 2D Solution")
    fig.colorbar(im, ax=ax, label="Mass")
    
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
    plt.show()