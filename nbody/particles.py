import numpy as np

class Particles:
    """
    
    The Particles class handle all particle properties

    for the N-body simulation. 

    """
    def __init__(self, N:int = 100):
        """
        Prepare memories for N particles

        :param N: number of particles.

        By default: particle properties include:
                nparticles: int. number of particles
                _masses: (N,1) mass of each particle
                _positions:  (N,3) x,y,z positions of each particle
                _velocities:  (N,3) vx, vy, vz velocities of each particle
                _accelerations:  (N,3) ax, ay, az accelerations of each partciel
                _tags:  (N)   tag of each particle
                _time: float. the simulation time 

        """
        self.nparticles = N
        self._time = 0 # initial time = 0
        self._masses = np.ones((N, 1))
        self._positions = np.zeros((N, 3))
        self._velocities = np.zeros((N, 3))
        self._accelerations = np.zeros((N, 3))
        self._tags = np.linspace(1, N, N)
        
        return


    def get_time(self):
        return self._time
    
    def get_masses(self):
        return self._masses
    
    def get_positions(self):
        return self._positions
    
    def get_velocities(self):
        return self._velocities
    
    def get_accelerations(self):
        return self._accelerations
    
    def get_tags(self):
        return self._tags
    
    def get_time(self):
        return self._time


    def set_time(self, time):
        self._time = time
        return
    
    def set_masses(self, mass):
        self._masses = mass
        return
    
    def set_positions(self, pos):
        self._positions = pos
        return
    
    def set_velocities(self, vel):
        self._velocities = vel
        return
    
    def set_accelerations(self, acc):
        self._accelerations = acc
        return
    
    def set_tags(self, IDs):
        self._tags = IDs
        return
    
    def output(self, fn, time):
        """
        Write simulation data into a file named "fn"


        """
        mass = self._masses
        pos  = self._positions
        vel  = self._velocities
        acc  = self._accelerations
        tag  = self._tags
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics Lab

                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        np.savetxt(fn,(tag[:],mass[:,0],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)

        return