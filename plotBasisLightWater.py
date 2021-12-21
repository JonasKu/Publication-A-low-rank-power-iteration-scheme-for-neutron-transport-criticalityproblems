from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

import create

import numpy as np
import re

NCells =[400]
geo = 'sphere'
problem = ["light_water_reactor"]
ranks = [5, 10, 15, 20, 25]

# determine energy group centers
G = [361]#[70]
R = 79.06925;

nBasisFuns = 5

# plot basis
for p in problem:
    for nG in G:
        # determine energy group centers
        group_centers = np.loadtxt('{}group_centers_{}G_SHEM.csv'.format('data/',nG))
        group_edges = np.loadtxt('{}group_edges_{}G_SHEM.csv'.format('data/',nG))
        dEInv = np.zeros((nG,nG))
        for g in range(nG):
            #dEInv[g,g] = 1.0/(group_edges[g+1] - group_edges[g])
            dEInv[g,g] = 1.0

        #group_centers = group_centers[::-1]
        for nX in NCells:
            for r in ranks:
                # open csv file
                phi = np.genfromtxt('output/phiDLRProblem'+p+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', delimiter=',')
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
                plt.savefig(f"{p}/phiBasisXMaterial{p}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

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
                plt.savefig(f"{p}/phiBasisWLogMaterial{p}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

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
                plt.savefig(f"{p}/phiBasisWMaterial{p}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')

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
                plt.savefig(f"{p}/phiSingularValuesMaterial{p}G{nG}NCells{nX}Rank{r}.png", bbox_inches='tight')
            



