import numpy as np
from pathlib import Path
import time
from numba import jit, njit, prange, set_num_threads
from nbody.particles import Particles
import matplotlib.pyplot as plt

"""

This program solve 3D direct N-particles simulations 
under gravitational forces. 

This file contains two classes:

1) Particles: describes the particle properties
2) NbodySimulation: describes the simulation

Usage:

    Step 1: import necessary classes

    from nbody import Particles, NbodySimulation

    Step 2: Write your own initialization function

    
        def initialize(particles:Particles):
            ....
            ....
            particles.set_masses(mass)
            particles.set_positions(pos)
            particles.set_velocities(vel)
            particles.set_accelerations(acc)

            return particles

    Step 3: Initialize your particles.

        particles = Particles(N=100)
        initialize(particles)


    Step 4: Initial, setup and start the simulation

        simulation = Simulation(particles)
        simulation.setip(...)
        simulation.evolve(dt=0.001, tmax=10)


Author: Yuan-Yen Peng (editted from Prof. Kuo-Chuan Pan, NTHU 2022.10.30)
Dept. of Physics, NTHU
Date: Npv. 28, 2022
For the course, computational physics lab

"""

def ACC_norm(N, posx, posy, posz, G, mass, rsoft):
    '''
    Acceleration with normal for loops.
    
    :param N: number of particles
    :param posx: position x
    :param posy: position y
    :param posz: position z
    :param G: gravitational constant
    :param mass: mass
    '''
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if j > i:
                x = posx[i] - posx[j]
                y = posy[i] - posy[j]
                z = posz[i] - posz[j]
                r = np.sqrt(x**2 + y**2 + z**2) + rsoft
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                F = - G * mass[i, 0] * mass[j, 0] / np.square(r)
                # Fx = F * np.cos(phi)
                # Fy = F * np.sin(phi)
                # Fz = 0
                Fx = F * np.cos(theta) * np.cos(phi)
                Fy = F * np.cos(theta) * np.sin(phi)
                Fz = F * np.sin(theta)

                acc[i, 0] += Fx / mass[i, 0]
                acc[j, 0] -= Fx / mass[j, 0]

                acc[i, 1] += Fy / mass[i, 0]
                acc[j, 1] -= Fy / mass[j, 0]
                
                acc[i, 2] += Fz / mass[i, 0]
                acc[j, 2] -= Fz / mass[j, 0]
    return acc

@jit(nopython=True)
def ACC_jit(N, posx, posy, posz, G, mass, rsoft):
    '''
    Acceleration with numba jit for loops
    
    :param N: number of particles
    :param posx: position x
    :param posy: position y
    :param posz: position z
    :param G: gravitational constant
    :param mass: mass
    '''
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if j > i:
                x = posx[i] - posx[j]
                y = posy[i] - posy[j]
                z = posz[i] - posz[j]
                r = np.sqrt(x**2 + y**2 + z**2) + rsoft
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                F = - G * mass[i, 0] * mass[j, 0] / np.square(r)
                Fx = F * np.cos(theta) * np.cos(phi)
                Fy = F * np.cos(theta) * np.sin(phi)
                Fz = F * np.sin(theta)

                acc[i, 0] += Fx / mass[i, 0]
                acc[j, 0] -= Fx / mass[j, 0]

                acc[i, 1] += Fy / mass[i, 0]
                acc[j, 1] -= Fy / mass[j, 0]
                
                acc[i, 2] += Fz / mass[i, 0]
                acc[j, 2] -= Fz / mass[j, 0]
                
    return acc

set_num_threads(8)
@njit(parallel=True)
def ACC_njit(N, posx, posy, posz, G, mass, rsoft):
    '''
    Acceleration with numba njit for loops (parallel)
    
    :param N: number of particles
    :param posx: position x
    :param posy: position y
    :param posz: position z
    :param G: gravitational constant
    :param mass: mass
    '''
    acc = np.zeros((N, 3))
    for i in prange(N):
        for j in prange(N):
            if j > i:
                x = posx[i] - posx[j]
                y = posy[i] - posy[j]
                z = posz[i] - posz[j]
                r = np.sqrt(x**2 + y**2 + z**2) + rsoft
                theta = np.arccos(z / r)
                phi = np.arctan2(y, x)
                F = - G * mass[i, 0] * mass[j, 0] / np.square(r)
                Fx = F * np.cos(theta) * np.cos(phi)
                Fy = F * np.cos(theta) * np.sin(phi)
                Fz = F * np.sin(theta)

                acc[i, 0] += Fx / mass[i, 0]
                acc[j, 0] -= Fx / mass[j, 0]

                acc[i, 1] += Fy / mass[i, 0]
                acc[j, 1] -= Fy / mass[j, 0]
                
                acc[i, 2] += Fz / mass[i, 0]
                acc[j, 2] -= Fz / mass[j, 0]
    return acc

class NbodySimulation:
    """
    
    The N-body Simulation class.
    
    """
    
    def __init__(self,particles:Particles):
        """
        Initialize the N-body simulation with given Particles.

        :param particles: A Particles class.  
        
        """

        # store the particle information
        self.nparticles = particles.nparticles
        self.particles  = particles

        # Store physical information
        self.time  = 0.0  # simulation time

        # Set the default numerical schemes and parameters
        self.setup()
        
        return

    def setup(self, G=1, 
                    rsoft=0.01, 
                    method="Euler", 
                    io_freq=10, 
                    io_title="particles",
                    io_screen=True,
                    visualized=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_title: the output header
        :param io_screen: print message on screen or not.
        :param visualized: on the fly visualization or not. 
        
        """
        # TODO:
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_title = io_title
        self.io_screen = io_screen
        self.visualized = visualized
        return

    def evolve(self, dt:float=0.01, tmax:float=1):
        """

        Start to evolve the system

        :param dt: time step
        :param tmax: the finial time
        
        """
        # TODO:
        method = self.method
        if method=="Euler":
            _update_particles = self._update_particles_euler
        elif method=="RK2":
            _update_particles = self._update_particles_rk2
        elif method=="RK4":
            _update_particles = self._update_particles_rk4    
        else:
            print("No such update meothd", method)
            quit() 

        # prepare an output folder for lateron output
        io_folder = "data_"+self.io_title
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        io_folder_fig = "fig_" + self.io_title
        Path(io_folder_fig).mkdir(parents=True, exist_ok=True)
        
        # ====================================================
        #
        # The main loop of the simulation
        #
        # =====================================================

        # TODO:
        particles = self.particles # call the class: Particles
        n = 0
        t = particles.get_time()
        t1 = time.time()
        while t < tmax:
            # update particles
            _update_particles(dt, particles)
            # update io
            if (n % self.io_freq == 0):
                if self.io_screen:
                    # print('n = ', n, 'time = ', t, 'dt = ', dt)
                    # output
                    fn = io_folder+"/data_"+self.io_title+"_"+str(n).zfill(5)+".txt"
                    print(fn)
                    self.particles.output(fn, t)
                    
                    # savefig
                    scale = 60
                    fig, ax = plt.subplots()
                    fig.set_size_inches(10.5, 10.5, forward=True)
                    fig.set_dpi(72)
                    ax.set_xlim(-1*scale,1*scale)
                    ax.set_ylim(-1*scale,1*scale)
                    ax.set_aspect('equal')
                    ax.set_xlabel('X [code unit]')
                    ax.set_ylabel('Y [code unit]')
                    pos = particles.get_positions()
                    plt.title(f'Time = {np.round(t, 0)}')
                    FIG = f'{io_folder_fig}/fig_{self.io_title}_{str(int(0.01 * n)).zfill(1)}.png'
                    ax.scatter(pos[:, 0], pos[:, 1], s = 10, alpha = .3)
                    plt.savefig(FIG)
                    plt.show()
            
            # update time
            if t + dt > tmax:
                dt = tmax - t
            t += dt
            n += 1
        t2 = time.time()
        print("Time diff: ", t2 - t1)
        print("Done!")
        return

    def _calculate_acceleration(self, mass, pos):
        """
        Calculate the acceleration.
        """
        # TODO:
        G = self.G
        rsoft = self.rsoft
        posx = pos[:, 0]
        posy = pos[:, 1]
        posz = pos[:, 2]
        N = self.nparticles
        
        return ACC_jit(N, posx, posy, posz, G, mass,rsoft)
        # return ACC_njit(N, posx, posy, posz, G, mass, rsoft)

    def _update_particles_euler(self, dt, particles:Particles):
        # TODO:
        mass = particles.get_masses()
        pos = particles.get_positions()
        vel = particles.get_velocities()
        acc = self._calculate_acceleration(mass, pos)
        y0 = np.array([pos, vel])
        yder = np.array([vel, acc])
        
        y0 = np.add(y0, yder * dt)
        
        particles.set_positions(y0[0])
        particles.set_velocities(y0[1])
        particles.set_accelerations(acc)
        
        return particles

    def _update_particles_rk2(self, dt, particles:Particles):
        # TODO:
        mass = particles.get_masses()
        pos = particles.get_positions()
        vel = particles.get_velocities()
        acc = self._calculate_acceleration(mass, pos)
        
        y0 = np.array([pos, vel])
        yder = np.array([vel, acc])
        k1 = yder
        y_temp = y0 + dt * k1 
        acc = self._calculate_acceleration(mass, y_temp[0])
        k2 = np.array([y_temp[1], acc])
        y0 = np.add(y0, (dt / 2) * (k1 + k2))
        
        particles.set_positions(y0[0])
        particles.set_velocities(y0[1])
        particles.set_accelerations(acc)
        return particles

    def _update_particles_rk4(self, dt, particles:Particles):
        # TODO:
        mass = particles.get_masses()
        pos = particles.get_positions()
        vel = particles.get_velocities()
        acc = self._calculate_acceleration(mass, pos)
        
        y0 = np.array([pos, vel])
        yder = np.array([vel, acc])
        k1 = yder
        y_temp = y0 + 0.5 * dt * k1 
        acc = self._calculate_acceleration(mass, y_temp[0])
        k2 = np.array([y_temp[1], acc])
        y_temp = y0 + 0.5 * dt * k2
        acc = self._calculate_acceleration(mass, y_temp[0])
        k3 = np.array([y_temp[1], acc])
        y_temp = y0 + dt * k3
        acc = self._calculate_acceleration(mass, y_temp[0])
        k4 = np.array([y_temp[1], acc])
        
        y0 = np.add(y0, (1/6) * dt * (k1 + 2*k2 + 2*k3 + k4))
        
        particles.set_positions(y0[0])
        particles.set_velocities(y0[1])
        particles.set_accelerations(acc)
        return particles


if __name__=='__main__':

    # test Particles() here
    particles = Particles(N=10)
    # test NbodySimulation(particles) here
    sim = NbodySimulation(particles=particles)
    sim.setup(G = 6.67e-8, io_freq=2, io_screen=True, io_title="test")
    sim.evolve(dt = 1, tmax = 10)
    print(sim.G)
    print("Done")