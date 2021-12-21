from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

import create

import numpy as np
import re

NCells =[400]
geo = 'sphere'
problem = "plutonium_carbon_02"
ranks = [5, 10, 15, 20]

# determine energy group centers
G = [618]#[70]

#keff = 0.9962781679989708 # plutonium_01, G = 70, I = 20
#keff = 0.9962248494385907 # plutonium_01, G = 618, I = 20
#keff = 0.7334307928804227 # plutonium_mix_02, G = 70, I = 20
#keff = 0.7322480800969234 # plutonium_mix_02, G = 618, I = 20
#keff = 0.7277073607021792290 # plutonium_mix_02, G = 70, I = 100
#keff = 7.265133840194547554e-01 # plutonium_mix_02, G = 618, I = 400
#keff = 9.991935071313471362e-01 # plutonium_mix_02, G = 618, I = 400, r = 25
keff = 1.010016455736819418e+00 ## plutonium_carbon_02,G = 618, I = 400, r = 20

plt.figure(figsize=(12, 11), dpi=80)
maxIt = -1

for nG in G:
    for nX in NCells:

        if nG < 618 or nX < 100:
            kHistory = np.genfromtxt('output/kEffectiveProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', delimiter=',')
            if maxIt < kHistory.shape[0]:
                maxIt = kHistory.shape[0]
            it = np.linspace(0.0,kHistory.shape[0],kHistory.shape[0])
            plt.plot(it,np.abs(kHistory-keff), "k:", linewidth=2,label='full', alpha=1.0)

        for r in ranks:
            # open csv file
            kHistory = np.genfromtxt('output/kEffectiveDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
            if maxIt < kHistory.shape[0]:
                maxIt = kHistory.shape[0]
            itDLR = np.linspace(0.0,kHistory.shape[0],kHistory.shape[0])
            plt.plot(itDLR,np.abs(kHistory-keff), "-", linewidth=2,label='rank '+str(r), alpha=0.8)

plt.yscale('log')
plt.ylabel(r'$\vert k_{eff}-k_{eff}^*\vert$', fontsize=20)
plt.xlabel('iteration', fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)
maxIt = 40
plt.xlim([0,maxIt])
x, labels = plt.xticks()
xint = range(int(x[0]), int(x[-1])+1,5)
plt.xticks(xint)      
plt.grid(True)      
plt.rc('grid', linestyle=":", color='gray')

plt.savefig(f"{problem}/historyMaterial{problem}G{G}NCells{NCells}.png", bbox_inches='tight')


