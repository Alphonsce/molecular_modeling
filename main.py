# Minimal parts : <=100

SIGMA = 1
EPSILON = 1
M = 1

TIME_STEPS = 500
N = 20
L = N ** (1 / 3)

dt = 0.001
#rho = 
#T = 

class Particle:
    def __init__(self, x, y, z, vx, vy, vz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.fx = 0
        self.fy = 0
        self.fz = 0

    def move(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        self.vx += (self.fx / M) * dt
        self.vy += (self.fy / M) * dt
        self.vz += (self.fz / M) * dt

# def initialize_particles():
#     for _ in range(N):

def main_function(n_particles):
    '''это главный цикл, в котором будет происходить расчет взаимодействия частиц и их движение
    '''
    pass
