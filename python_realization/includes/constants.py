# ---------------------------------- #
# Argon is used as a reference gas, his parameters in SI: m = 6.7E-26 kg; epsilon = 1.6E-21 J; sigma = 3.4E-10 m

SIGMA = 1
EPSILON = 1
M = 1

TIME_STEPS = 1
N = 1000
L = 3 * N ** (1 / 3)
# the distance of cut for the LJ potential
r_cut = 10
#rho = N / L ** 3
dt = 0.0005      # 0.001
# T is in epsilon / k_b units;
# ---------------------------------- #