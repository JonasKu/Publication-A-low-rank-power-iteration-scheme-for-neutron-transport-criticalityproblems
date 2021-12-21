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
    return (Nx^2+G^2)*r^2

NCells =[100]
geo = 'sphere'
problem = ["plutonium_mix_02"]
ranks = [5, 10, 15, 20]

# determine energy group centers
G = [70,618]
markers = ['>','o']

keff = 0.7265133840194548#7.265133935673440124e-01 # plutonium_mix_02, G = 618, I = 100

area = (10)**2  # 0 to 10 point radii
c = np.sqrt(area)
gCounter = -1


for p in problem:
    plt.figure(figsize=(12, 11), dpi=80)
    for nG in G:
        gCounter = gCounter+1;
        for nX in NCells:

            if nG < 618 and nX < 100:
                kHistory = np.genfromtxt('output/kEffectiveProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', delimiter=',')

                if nG == 618:
                    kHistory = np.load('output/phi_diffusion_plutonium_mix_02_' + str(nX))
                print(kHistory)
                
                plt.scatter(memoryFull(nX,nG),np.abs(kHistory[-1]-keff),marker='X',s = area,label='full', alpha=1.0)

            for r in ranks:
                # open csv file
                kHistory = np.genfromtxt('output/kEffectiveDLRProblem'+p+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
                print(np.abs(kHistory[-1]-keff))
                plt.scatter(memoryDLRA(nX,nG,r),np.abs(kHistory[-1]-keff),marker=markers[gCounter],s = area,label='rank '+str(r)+', $N_x$ '+str(nX)+', G '+str(nG), alpha=1.0)

    plt.yscale('log')
    plt.ylabel(r'$\vert k_{eff}-k_{eff}^*\vert$', fontsize=20)
    plt.xlabel('memory', fontsize=20)
    #plt.xlim([0,R])
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f"{p}/memoryProblem{p}G{G}NCells{NCells}Rank{r}.png", bbox_inches='tight')


