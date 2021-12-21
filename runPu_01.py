
from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt

import create

import numpy as np
import re

I = 20
NCells = I
geo = 'sphere'
problem = "plutonium_01"

# determine energy group centers
G = 70

variables = create.selection(problem,G,I)

example = Diffusion(*variables,geo=geo)
example.geometry()

D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb = example.create_matrix_DLRA_format_harmonic()

phi,keff,kHistory = example.solver_matrix_inv(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)

phiDLR,keffDLR,kHistoryDLR = example.solver_matrix_inv_DLR_harmonic(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)

#phiDLR,keffDLR,kHistoryDLR = example.solver_matrix_inv_DLR_adaptive(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)

#phi,keff,kHistory = example.solver_matrix_inv(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)
keff = 0.8814716866525122

R = example.R
problem = re.findall(r'^[A-Za-z]+',problem)[0]

# plot phi
plt.figure(figsize=(12, 8), dpi=80)
x = np.linspace(0.0,R,NCells)
#plt.plot(x,phi, ":", linewidth=3, alpha=0.8)
plt.plot(x,phiDLR, "-", linewidth=2, alpha=0.4)
plt.ylabel(r'$\phi$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.savefig(f"phiRank{example.r}Material{problem}G{G}.png", bbox_inches='tight')

# determine energy group centers
group_centers = np.loadtxt('{}group_centers_{}G_Pu_20pct240.csv'.format('data/',G))

# compute solutions in different energy regions
phiThermal = np.zeros((NCells))
phiEpiThermal = np.zeros((NCells))
phiFast = np.zeros((NCells))
for j in range(NCells):
    for g in range(G):
        if group_centers[g]*1e6 < 5:
            phiThermal[j] += phiDLR[j,g]
        else:
            if group_centers[g]*1e6 > 0.5*1e6:
                phiFast[j] += phiDLR[j,g]
            else:
                phiEpiThermal[j] += phiDLR[j,g]



# plot phi groups
plt.figure(figsize=(12, 8), dpi=80)
x = np.linspace(0.0,R,NCells)
plt.plot(x,phiThermal, ":", linewidth=3,label='thermal', alpha=0.8)
plt.plot(x,phiEpiThermal, ":", linewidth=3,label='epithermal', alpha=0.8)
plt.plot(x,phiFast, ":", linewidth=3,label='fast', alpha=0.8)
plt.ylabel(r'$\phi$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)
plt.savefig(f"phiGroupsRank{example.r}Material{problem}G{G}.png", bbox_inches='tight')

# plot history k
plt.figure(figsize=(12, 8), dpi=80)
#it = np.linspace(0.0,kHistory.shape[0],kHistory.shape[0])
itDLR = np.linspace(0.0,kHistoryDLR.shape[0],kHistoryDLR.shape[0])
#plt.plot(it,np.abs(kHistory-keff), ":", linewidth=3,label='conventional', alpha=0.8)
plt.plot(itDLR,np.abs(kHistoryDLR-keff), "-", linewidth=2,label='DLRA', alpha=0.8)
plt.yscale('log')
plt.ylabel(r'$\vert k_{eff}-k_{eff}^*\vert$', fontsize=20)
plt.xlabel('iteration', fontsize=20)
#plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)
plt.savefig(f"historyRank{example.r}Material{problem}G{G}.png", bbox_inches='tight')

# save to csv file
np.savetxt('output/phiDLRProblem'+problem+'Nx'+str(NCells)+'G'+str(G)+'Rank'+str(example.r)+'.csv', phiDLR, delimiter=',')
np.savetxt('output/kEffectiveDLRProblem'+problem+'Nx'+str(NCells)+'G'+str(G)+'Rank'+str(example.r)+'.csv', kHistoryDLR, delimiter=',')
