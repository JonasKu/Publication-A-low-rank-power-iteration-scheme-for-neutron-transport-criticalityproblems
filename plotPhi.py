from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

import create

import numpy as np
import re

NCells =[100]
geo = 'sphere'
problem = "plutonium_mix_02"
ranks = [5, 10, 15, 20]

# determine energy group centers
G = [618]

variables = create.selection(problem,G[0],NCells[0])
example = Diffusion(*variables,geo=geo)
example.geometry()
R = example.R

# plot phi groups
for nG in G:
    # determine energy group centers
    group_centers = np.loadtxt('{}group_centers_{}G_Pu_20pct240.csv'.format('data/',nG))
    for nX in NCells:
        if nG < 618 or nX < 100:
            phi = np.genfromtxt('output/phiProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', delimiter=',')
            phiFullThermal = np.zeros((nX))
            phiFullEpiThermal = np.zeros((nX))
            phiFullFast = np.zeros((nX))
            for j in range(nX):
                for g in range(nG):
                    if group_centers[g]*1e6 < 5:
                        phiFullThermal[j] += phi[j,g]
                    else:
                        if group_centers[g]*1e6 > 0.5*1e6:
                            phiFullFast[j] += phi[j,g]
                        else:
                            phiFullEpiThermal[j] += phi[j,g]
        for r in ranks:
            # open csv file
            phi = np.genfromtxt('output/phiDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
            # compute solutions in different energy regions
            phiThermal = np.zeros((nX))
            phiEpiThermal = np.zeros((nX))
            phiFast = np.zeros((nX))
            for j in range(nX):
                for g in range(nG):
                    if group_centers[g]*1e6 < 5:
                        phiThermal[j] += phi[j,g]
                    else:
                        if group_centers[g]*1e6 > 0.5*1e6:
                            phiFast[j] += phi[j,g]
                        else:
                            phiEpiThermal[j] += phi[j,g]
            x = np.linspace(0.0,R,nX)
            plt.figure(figsize=(12, 11), dpi=80)
            sgn = np.sign(phiFast[0])
            plt.plot(x,sgn*phiThermal, "b:", linewidth=3,label='thermal, r = '+str(r), alpha=1.0)
            plt.plot(x,sgn*phiEpiThermal, "r:", linewidth=3,label='epithermal, r = '+str(r), alpha=1.0)
            plt.plot(x,sgn*phiFast, "g:", linewidth=3,label='fast, r = '+str(r), alpha=1.0)
            if nG < 618 or nX < 100:
                plt.plot(x,phiFullThermal, "b-", linewidth=3,label='thermal, full', alpha=0.4)
                plt.plot(x,phiFullEpiThermal, "r-", linewidth=3,label='epithermal, full', alpha=0.4)
                plt.plot(x,phiFullFast, "g-", linewidth=3,label='fast, full', alpha=0.4)
            plt.ylabel(r'$\phi$', fontsize=20)
            plt.xlabel(r'$x$', fontsize=20)
            plt.xlim([0,R])
            plt.tick_params(axis='both', labelsize=20)
            plt.legend(fontsize=20)
            plt.savefig(f"{problem}/phiGroupsMaterial{problem}G{G}NCells{NCells}Rank{r}.png", bbox_inches='tight')

            plt.figure(figsize=(12, 11), dpi=80)
            plt.plot(phi[97,:], "b:", linewidth=3,label='thermal, r = '+str(r), alpha=1.0)
            plt.savefig(f"{problem}/phiMaterial{problem}G{G}NCells{NCells}Rank{r}.png", bbox_inches='tight')
            



