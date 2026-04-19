# FDTD.py: FDTD Maxwell's equations in 1-D with periodic BC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    Xm = 201
    beta = 0.01

    Ex = np.zeros((Xm, 2))
    Hy = np.zeros((Xm, 2))

    # Initial fields
    z = np.arange(Xm)
    Ex[:Xm, 0] = 0.1 * np.sin(2 * np.pi * z / 100.0)
    Hy[:Xm, 0] = 0.1 * np.sin(2 * np.pi * z / 100.0)

    # Set up plot
    fig, (ax_e, ax_h) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('E: cyan, H: red. Periodic BC')

    line_e, = ax_e.plot(z, Ex[:Xm, 0], color='cyan', linewidth=1.5)
    ax_e.set_ylabel('Ex')
    ax_e.set_ylim(-0.15, 0.15)
    ax_e.grid(True, alpha=0.3)

    line_h, = ax_h.plot(z, Hy[:Xm, 0], color='red', linewidth=1.5)
    ax_h.set_xlabel('Z')
    ax_h.set_ylabel('Hy')
    ax_h.set_ylim(-0.15, 0.15)
    ax_h.grid(True, alpha=0.3)

    def update(frame):
        # FDTD time stepping
        Ex[1:Xm - 1, 1] = Ex[1:Xm - 1, 0] + beta * (Hy[0:Xm - 2, 0] - Hy[2:Xm, 0])
        Hy[1:Xm - 1, 1] = Hy[1:Xm - 1, 0] + beta * (Ex[0:Xm - 2, 0] - Ex[2:Xm, 0])

        # Periodic BC
        Ex[0, 1] = Ex[0, 0] + beta * (Hy[Xm - 2, 0] - Hy[1, 0])
        Ex[Xm - 1, 1] = Ex[Xm - 1, 0] + beta * (Hy[Xm - 2, 0] - Hy[1, 0])
        Hy[0, 1] = Hy[0, 0] + beta * (Ex[Xm - 2, 0] - Ex[1, 0])
        Hy[Xm - 1, 1] = Hy[Xm - 1, 0] + beta * (Ex[Xm - 2, 0] - Ex[1, 0])

        # Update plot
        line_e.set_ydata(Ex[:Xm, 1])
        line_h.set_ydata(Hy[:Xm, 1])

        # Swap time levels
        Ex[:Xm, 0] = Ex[:Xm, 1]
        Hy[:Xm, 0] = Hy[:Xm, 1]

        return line_e, line_h

    anim = FuncAnimation(fig, update, frames=2000, interval=5, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
