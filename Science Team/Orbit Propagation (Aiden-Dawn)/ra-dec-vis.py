import numpy as np
import matplotlib.pyplot as plt

#rubindata = np.genfromtxt('satcoord-rubin.csv',delimiter=',',skip_header=1)
#gmudata = np.genfromtxt('satcoord-gmu.csv',delimiter=',',skip_header=1)
#palomardata = np.genfromtxt('satcoord-palomar.csv',delimiter=',',skip_header=1)
#snifsdata = np.genfromtxt('satcoord-snifs.csv',delimiter=',',skip_header=1)
#stardata = np.genfromtxt('exoplanet-host-stars.csv',delimiter=',',skip_header=9)
data = np.genfromtxt('satcoord.csv',delimiter=',',skip_header=1)


#rubin_ra = rubindata[:96,1]*(np.pi/180)
#gmu_ra = gmudata[:96,1]*(np.pi/180)
#palomar_ra = palomardata[:96,1]*(np.pi/180)
#snifs_ra = snifsdata[:96,1]*(np.pi/180)
#star_ra = stardata[:,0]*(np.pi/180)
ra = data[:50,1]*(np.pi/180)
"""
for i in range(len(rubin_ra)):
    if rubin_ra[i] >= np.pi:
        rubin_ra[i] = rubin_ra[i] - 2*np.pi
for i in range(len(rubin_ra)):
    if gmu_ra[i] >= np.pi:
        gmu_ra[i] = gmu_ra[i] - 2*np.pi
for i in range(len(rubin_ra)):
    if palomar_ra[i] >= np.pi:
        palomar_ra[i] = palomar_ra[i] - 2*np.pi
for i in range(len(rubin_ra)):
    if snifs_ra[i] >= np.pi:
        snifs_ra[i] = snifs_ra[i] - 2*np.pi
for i in range(len(star_ra)):
    if star_ra[i] >= np.pi:
        star_ra[i] = star_ra[i] - 2*np.pi
"""
for i in range(len(ra)):
    if ra[i] >= np.pi:
        ra[i] = ra[i] - 2*np.pi
        
#rubin_dec = rubindata[:96,2]*(np.pi/180)
#gmu_dec = gmudata[:96,2]*(np.pi/180)
#palomar_dec = palomardata[:96,2]*(np.pi/180)
#snifs_dec = snifsdata[:96,2]*(np.pi/180)
#star_dec = stardata[:,1]*(np.pi/180)
dec = data[:50,2]*(np.pi/180)

rubincountindex = []
gmucountindex = []
palomarcountindex = []
snifscountindex = []

"""
for i in range(len(rubin_ra)):
    rubincountindex = np.append(rubincountindex, np.where((abs(rubin_ra[i] - star_ra) <= 50*0.000291) & (abs(rubin_dec[i] - star_dec) <= 10*0.000291))).astype(int)
    gmucountindex = np.append(gmucountindex, np.where((abs(gmu_ra[i] - star_ra) <= 50*0.000291) & (abs(gmu_dec[i] - star_dec) <= 10*0.000291))).astype(int)
    palomarcountindex = np.append(palomarcountindex, np.where((abs(palomar_ra[i] - star_ra) <= 50*0.000291) & (abs(palomar_dec[i] - star_dec) <= 10*0.000291))).astype(int)
    snifscountindex = np.append(snifscountindex, np.where((abs(snifs_ra[i] - star_ra) <= 50*0.000291) & (abs(snifs_dec[i] - star_dec) <= 10*0.000291))).astype(int)
"""
plt.figure(figsize=[6.4,6.4])
ax = plt.subplot(projection='aitoff')
#ax.plot(rubin_ra,rubin_dec,'.',label='Rubin',ms=3,c='darkred')
#ax.plot(gmu_ra,gmu_dec,'.',label='GMU',ms=3,c='darkgreen')
#ax.plot(palomar_ra,palomar_dec,'.',label='Palomar',ms=3,c='purple')
#ax.plot(snifs_ra,snifs_dec,'.',label='SNIFS',ms=3,c='darkblue')
#ax.plot(star_ra[rubincountindex],star_dec[rubincountindex],'.',ms=1,label='Observable Exoplanet Host Stars',c='yellow')
ax.plot(ra,dec,'.',ms=1)
ax.plot(ra[::3600],dec[::3600],'.',ms=5)
ax.grid()
#ax.set_facecolor('k')
#plt.title('RAAN = 120$^{\circ}$, Ecc = 0, Incl = -20$^{\circ}$', pad=30)
#plt.legend(loc='upper right', bbox_to_anchor = [0.775,-.1])
plt.savefig('retrograde-1day.png')
plt.show()
plt.close()

"""
plt.figure()
ax = plt.subplot(projection='aitoff')
ax.plot(rubin_ra[0:96],rubin_dec[0:96],'.',ms=3,c='pink',label='First orbit')
#ax.plot(rubin_ra[100*96:101*96],rubin_dec[100*96:101*96],'.',ms=0.5,c='r',label='100th Orbit')
#ax.plot(rubin_ra[200*96:201*96],rubin_dec[200*96:201*96],'.',ms=0.5,c='g', label='200th Orbit')
ax.plot(rubin_ra[300*96:301*96],rubin_dec[300*96:301*96],'.',ms=0.5,c='darkblue',label='300th Orbit (~10 months later)')
plt.grid()
plt.legend()
plt.savefig('precession.png')
plt.show()
plt.close()
"""