import numpy as np
import matplotlib.pyplot as plt

# Parameters
U0 = 1.0
rho = 1000
T = 0.1
mu = 1
omega = 2*np.pi/T
k = np.sqrt(rho*omega / (2*mu))
dz = 1/1000
H = 0.05
Nz = int(round(H/dz)+1)

nu = mu / rho


dt = 0.4 * dz**2 / (2*nu)
Nt = int(10*T/dt)
length = range(0,Nz)
j = np.linspace(0, H, Nz)
t = np.linspace(0,10,Nt)


nuField = np.zeros((Nt,Nz))


def dzz(u, dz):
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dz**2
    return d2u

# Time stepping
for n in range(0, Nt-1):
    u = nuField[n, :].copy()
    d2u = dzz(u, dz)
    
    # Update interior points
    u_new = u + dt * nu * d2u
    
    # Boundary conditions
    u_new[0] = U0 * np.sin(omega * t[n])   # oscillating plate
    u_new[-1] = 0.0                        # stationary upper plate
    
    nuField[n+1, :] = u_new

time = np.linspace(0, 1, nuField.shape[1])
z = np.linspace(0, 1, nuField.shape[0])
Time, Z = np.meshgrid(time, z)

plt.contourf(Z, Time, nuField, levels=20, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Scalar Field (contourf)')
plt.xlabel('Time')
plt.ylabel('Z')
plt.show()