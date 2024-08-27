import numpy as np
import math

asun = 0.349066
aearth = 0.418879
theta = 0.523599

a = (math.cos(aearth) - (math.cos(asun)*math.cos(theta)))/math.sin(theta)
b = (math.sin(asun)**2 - a**2)**0.5

p1 = np.array([0, 0, 1])
p2 = np.array([math.sin(theta), 0, math.cos(theta)])
p3 = np.array([a, -b, math.cos(asun)])
p4 = np.array([a, b, math.cos(asun)])

nb = np.cross(p1, p4 - p1)/np.linalg.norm(np.cross(p1, p4 - p1))
nc = np.cross(p1, p3 - p1)/np.linalg.norm(np.cross(p1, p3 - p1))
phi1 = math.acos(np.vdot(nb, nc))
nb = np.cross(p2, p4 - p2)/np.linalg.norm(np.cross(p2, p4 - p2))
nc = np.cross(p2, p3 - p2)/np.linalg.norm(np.cross(p2, p3 - p2))
phi2 = math.acos(np.vdot(nb, nc))
nb = np.cross(p4, p1 - p4)/np.linalg.norm(np.cross(p4, p1 - p4))
nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
psi1 = math.acos(np.vdot(nb, nc))
nb = np.cross(p4, p2 - p4)/np.linalg.norm(np.cross(p4, p2 - p4))
nc = np.cross(p4, p3 - p4)/np.linalg.norm(np.cross(p4, p3 - p4))
psi2 = math.acos(np.vdot(nb, nc))

digon = 2*math.pi - 2*(psi1 + psi2) - phi1 * math.cos(asun) - phi2 * math.cos(aearth)
sunsolid = (math.pi* (1**2))/satsundist**2

print(digon)
print(sunsolid)