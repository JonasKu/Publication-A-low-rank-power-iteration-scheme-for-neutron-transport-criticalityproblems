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
ranks = [20]

# determine energy group centers
G = [618]

variables = create.selection(problem,G[0],NCells[0])
example = Diffusion(*variables,geo=geo)
example.geometry()
R = example.R

nBasisFuns = 5


# plot phi groups
for nG in G:
    # determine energy group centers
    group_centers = np.loadtxt('{}group_centers_{}G_Pu_20pct240.csv'.format('data/',nG))
    group_edges = np.loadtxt('{}group_edges_{}G_Pu_20pct240.csv'.format('data/',nG))
    group_centers = group_centers[::-1]*1e6
    group_edges = group_edges[::-1]*1e6
    # plot basis
    dEInv = np.zeros((nG,nG))
    for g in range(nG):
        dEInv[g,g] = 1.0/(group_edges[g+1] - group_edges[g])
        dEInv[g,g] = 1.0

    for nX in NCells:
        for r in ranks:
            # open csv file
            phi = np.genfromtxt('output/phiDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
            # compute basis
            U,S,VT = np.linalg.svd(phi,full_matrices=True)
            print(S)
            V = VT.transpose()
            
            x = np.linspace(0.0,R,nX)
            plt.figure(figsize=(12, 11), dpi=80)
            for i in range(nBasisFuns):
                maxX = np.max(U[:,i])
                minX = np.min(U[:,i])
                signX = 1.0
                if np.abs(minX) > maxX:
                    signX = -1.0
                plt.plot(x,signX*U[:,i], linewidth=3,label=r'$\widehat{X}_{'+str(i+1)+r'}$', alpha=1.0)

            plt.xlabel(r'radius (cm)', fontsize=20)
            plt.xlim([0,R])
            plt.tick_params(axis='both', labelsize=20)
            plt.legend(fontsize=20)
            plt.savefig(f"{problem}/phiBasisXMaterial{problem}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

            plt.figure(figsize=(12, 11), dpi=80)
            x = np.linspace(1,G,nG)
            for i in range(nBasisFuns):
                #plt.plot(group_centers,V[:,i],'-', linewidth=3,label=r'$\widehat{W}_{'+str(i+1)+r'}$', alpha=1.0)
                VE = dEInv@V[:,i]
                maxV = np.max(VE)
                minV = np.min(VE)
                signV = 1.0
                if np.abs(minV) > maxV:
                    signV = -1.0
                plt.plot(group_centers,signV*VE,'-', linewidth=3,label=r'$\widehat{W}_{'+str(i+1)+r'}$', alpha=1.0)

            plt.xlabel('E (eV)', fontsize=20)
            plt.xscale('log')
            plt.xlim([group_centers[0],group_centers[-1]])
            plt.tick_params(axis='both', labelsize=20)
            plt.legend(fontsize=20)
            plt.savefig(f"{problem}/phiBasisWLogMaterial{problem}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

            plt.figure(figsize=(12, 11), dpi=80)
            x = np.linspace(1,G,nG)
            for i in range(nBasisFuns):
                #plt.plot(group_centers,V[:,i],'-', linewidth=3,label=r'$\widehat{W}_{'+str(i+1)+r'}$', alpha=1.0)
                plt.plot(V[:,i],'-', linewidth=3,label=r'$\widehat{W}_{'+str(i+1)+r'}$', alpha=1.0)

            plt.xlabel('groups', fontsize=20)
            #plt.xlim([group_centers[0],group_centers[-1]])
            plt.xlim([1,nG])
            plt.tick_params(axis='both', labelsize=20)
            plt.legend(fontsize=20)
            plt.savefig(f"{problem}/phiBasisWMaterial{problem}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

            plt.figure(figsize=(12, 11), dpi=80)
            x = np.linspace(1,r,r)
            for i in range(nBasisFuns):
                #plt.plot(group_centers,V[:,i],'-', linewidth=3,label=r'$\widehat{W}_{'+str(i+1)+r'}$', alpha=1.0)
                plt.plot(x,S[:r],'ko', linewidth=5, alpha=1.0)

            plt.xlabel('index', fontsize=20)
            plt.ylabel('singular values', fontsize=20)
            plt.yscale('log')
            #plt.xlim([group_centers[0],group_centers[-1]])
            plt.xlim([1,r])
            x, labels = plt.xticks()
            xint = range(int(x[0]), int(x[-1])+1,5)
            plt.xticks(xint)      
            plt.grid(True)  
            plt.tick_params(axis='both', labelsize=20)
            #plt.legend(fontsize=20)
            plt.savefig(f"{problem}/phiSingularValuesMaterial{problem}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')
            



