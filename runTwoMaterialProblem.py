
from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt

import numpy as np
import re

R = 25.0; I = 100
NCells = I
geo = 'sphere'
problem = 'Pu70'
G = int(re.findall(r'\d+',problem)[0])
# problem combines materials and groups
# materials 'Pu', 'PuC', 'PuPuC'
# groups '12', '70', '618'


# run full problem
variables = dfp.selection(problem,R,I)
example = Diffusion(*variables,geo=geo)
example.geometry()
D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb = example.create_matrix_DLRA_format_harmonic()
phi,keff,kHistory = example.solver_matrix_inv(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)  

ranks = [3, 5,7,10,13,15,20]# 9 12 15 20]
nRanks = len(ranks)

phiDLR = np.zeros((nRanks,NCells,G))
keffDLR = np.zeros((nRanks))
nIterations = np.zeros((nRanks))
kHistoryRanks = []

for i in range(nRanks):
    print('run rank ',ranks[i])
    variables = dfp.selection(problem,R,NCells)

    example = Diffusion(*variables,geo=geo)
    example.geometry()
    example.r = ranks[i]

    D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb = example.create_matrix_DLRA_format_harmonic()

    phiDLR[i,:,:],keffDLR[i],kHistoryDLR = example.solver_matrix_inv_DLR_harmonic(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)
    nIterations[i] = len(kHistoryDLR)
    kHistoryRanks = np.append(kHistoryRanks,kHistoryDLR, axis=0)

# determine energy group centers
G = int(re.findall(r'\d+',problem)[0])
problem = re.findall(r'^[A-Za-z]+',problem)[0]
group_centers = np.loadtxt('{}group_centers_{}G_Pu_20pct240.csv'.format('data/',G))

# compute solutions in different energy regions
phiThermal = np.zeros((NCells))
phiEpiThermal = np.zeros((NCells))
phiFast = np.zeros((NCells))
for j in range(NCells):
    for g in range(G):
        if group_centers[g]*1e6 < 5:
            phiThermal[j] += phi[j,g]
        else:
            if group_centers[g]*1e6 > 0.5*1e6:
                phiFast[j] += phi[j,g]
            else:
                phiEpiThermal[j] += phi[j,g]

# plot phi
plt.figure(figsize=(12, 8), dpi=80)
x = np.linspace(0.0,R,NCells)
plt.plot(x,phi, ":", linewidth=3, alpha=0.8)
plt.plot(x,phiDLR[0,:,:], "-", linewidth=2, alpha=0.4)
plt.ylabel(r'$\phi$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.savefig(f"phiRank{example.r}Material{problem}G{G}.png", bbox_inches='tight')

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
it = np.linspace(1,kHistory.shape[0],kHistory.shape[0])
itDLR = np.linspace(0.0,kHistoryDLR.shape[0],kHistoryDLR.shape[0])
plt.plot(it,np.abs(kHistory-keff), ":", linewidth=3,label='conventional', alpha=0.8)
startIt = 0
for i in range(nRanks):
    endIt = int(startIt+nIterations[i])
    it = np.linspace(1,nIterations[i],int(nIterations[i]))
    print(int(nIterations[i]))
    print(it.shape)
    print((np.abs(kHistoryRanks[startIt:endIt]-keff)).shape)
    plt.plot(it,np.abs(kHistoryRanks[startIt:endIt]-keff), "-", linewidth=2,label=f"DLRA, r = {ranks[i]}", alpha=0.8)
    startIt = endIt
plt.yscale('log')
plt.ylabel(r'$\vert k_{eff}-k_{eff}^*\vert$', fontsize=20)
plt.xlabel('iteration', fontsize=20)
#plt.xlim([0,R])
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=20)
plt.savefig(f"historyRank{example.r}Material{problem}G{G}.png", bbox_inches='tight')
