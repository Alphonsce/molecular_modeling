# ---------------------------------- #
# Argon is used as a reference gas, his parameters in SI: m = 6.7E-26 kg; epsilon = 1.6E-21 m; sigma = 3.4E-10 J

SIGMA = 1
EPSILON = 1
M = 1

TIME_STEPS = 10000
N = 2
if N == 2:
    L = 30000 * N ** (1 / 3)
else:
    L = 3 * N ** (1 / 3)
r_cut = 50505055050505     # the distance of cut for the LJ potential

dt = 0.001      # 0.001
# T is in epsilon / k_b units; we need to rescale velocities for temperature to be our set value
T_thermostat = 0.8

# ---------------------------------- #