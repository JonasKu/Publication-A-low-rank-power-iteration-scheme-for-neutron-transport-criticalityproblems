from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

import create

import numpy as np
import re

def memoryFull(Nx,G): 
    return Nx*Nx*G+G*G*Nx*2

def memoryDLRA(Nx,G,r): 
    return (Nx*Nx+G*G)*r*r

NCells =[400]
geo = 'sphere'
problem = "uranium_01"
ranks = [5, 10, 15, 20, 25]  

# determine energy group centers
G = [87]
markers = ['>','o']

keff = 9.208736732340545572e-01 # Nx = 400, uranium_01
#keff =  9.077073771627990340e-01 # Nx = 400, uranium_02

area = (10)**2  # 0 to 10 point radii
c = np.sqrt(area)
gCounter = -1
xCounter = -1

plt.figure(figsize=(12, 11), dpi=80)
for nG in G:
    gCounter = gCounter+1
    xCounter = -1
    for nX in NCells:
        xCounter = xCounter + 1

        if nG < 618:


            kHistory = np.genfromtxt('output/kEffectiveProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', delimiter=',')
            kefferr = np.abs(kHistory[-1]-keff)
            
            print('Nx = ',nX,' ,G = ',G,': ',memoryFull(nX,nG))
            plt.scatter(memoryFull(nX,nG),kefferr,marker='X',s = area,label='full , $N_x$ '+str(nX)+', G '+str(nG), alpha=1.0)

    nX = 400
    for r in ranks:
        # open csv file
        kHistory = np.genfromtxt('output/kEffectiveDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
        plt.scatter(memoryDLRA(nX,nG,r),np.abs(kHistory[-1]-keff),marker=markers[gCounter],s = area,label='rank '+str(r)+', $N_x$ '+str(nX)+', G '+str(nG), alpha=1.0)
            
plt.yscale('log')
plt.ylabel(r'$\vert k_{eff}-k_{eff}^*\vert$', fontsize=20)
plt.xlabel('memory', fontsize=20)
#plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)
plt.savefig(f"{problem}/memoryMaterial{problem}G{G}NCells{NCells}Rank{r}.png", bbox_inches='tight')


