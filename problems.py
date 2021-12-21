""" Setting up diffusion problems """

import numpy as np
import re

DATA_PATH = 'data/'

def selection(problem,R,I):
    G = int(re.findall(r'\d+',problem)[0])
    problem = re.findall(r'^[A-Za-z]+',problem)[0]
    prob = getattr(Problems,problem)
    return prob(G,R,I)

class Problems:
    def compound(G,name=''):
        if G == 87 or G==618:
            diffusion = np.loadtxt('{}D_{}G_{}.csv'.format(DATA_PATH,G,name))
            chiNuFission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,name),delimiter=',')
            absorb = np.loadtxt('{}Siga_{}G_{}.csv'.format(DATA_PATH,G,name))
            scatter = np.loadtxt('{}Scat_{}G_{}.csv'.format(DATA_PATH,G,name),delimiter=',')
            removal = [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
        else:
            diffusion = np.loadtxt('{}D_{}G_Pu_20pct{}{}.csv'.format(DATA_PATH,G,PuType,ext))
            chi = np.loadtxt('{}chi_{}G_Pu_20pct{}{}.csv'.format(DATA_PATH,G,PuType,ext))
            fission = np.loadtxt('{}nuSigf_{}G_Pu_20pct{}{}.csv'.format(DATA_PATH,G,PuType,ext))
            absorb = np.loadtxt('{}Siga_{}G_Pu_20pct{}{}.csv'.format(DATA_PATH,G,PuType,ext))
            scatter = np.loadtxt('{}Scat_{}G_Pu_20pct{}{}.csv'.format(DATA_PATH,G,PuType,ext),delimiter=',')
            removal = [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
            chiNuFission = np.zeros((G,G))
            for g in range(G):
                for gpr in range(G):
                    chiNuFission[g,gpr] = chi[g]*fission[gpr] #set up the fission diagonal 

        np.fill_diagonal(scatter,0)
        return diffusion,scatter,chiNuFission,removal

    def Pu(G,R,I):
        diffusion_inner,scat_inner,chiNuFission_inner,removal_inner = Problems.compound(G,ext='')
        nMaterials = 1
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                scatter[0,g,gpr] = scat_inner[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_inner
        # material 2 start
        split = 1e10

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def PuC(G,R,I):
        diffusion_inner,scat_inner,chiNuFission_inner,removal_inner = Problems.compound(G,ext='C')

        nMaterials = 1
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                scatter[0,g,gpr] = scat_inner[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_inner
        # material 2 start
        split = 1e10

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def SS(G,R,I):
        diffusion_inner,scat_inner,chiNuFission_inner,removal_inner = Problems.compound(G,ext='SS')
        nMaterials = 1
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                scatter[0,g,gpr] = scat_inner[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_inner
        # material 2 start
        split = 1e10

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def PuPuC(G,R,I):
        nMaterials = 2
        diffusion_outer,scatter_outer,chiNuFission_outer,removal_outer = Problems.compound(G,ext='')
        diffusion_inner,scatter_inner,chiNuFission_inner,removal_inner = Problems.compound(G,ext='C')
        # store material properties for all materials
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]
            diffusion[1,g] = diffusion_outer[g]
            removal[1,g] = removal_outer[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                chiNuFission[1,g,gpr] = chiNuFission_outer[g,gpr]
                scatter[0,g,gpr] = scatter_inner[g,gpr]
                scatter[1,g,gpr] = scatter_outer[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_outer

        # material 2 start
        split = 10

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def PuSS(G,R,I):
        # 87 Group Problem:      Vacuum Boundary -- | -- 45 cm Carbon or Stainless Steel -- | -- 35 cm Enriched Uranium Hydride -- | -- 20 cm Depleted Uranium Hydride -- | -- Reflected Boundary
        nMaterials = 3
        diffusion_outer,scatter_outer,chiNuFission_outer,removal_outer = Problems.compound(G,ext='SS')
        diffusion_center,scatter_center,chiNuFission_center,removal_center = Problems.compound(G,ext='',PuType='239')
        diffusion_inner,scatter_inner,chiNuFission_inner,removal_inner = Problems.compound(G,ext='')
        # store material properties for all materials
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]

            diffusion[1,g] = diffusion_center[g]
            removal[1,g] = removal_center[g]

            diffusion[2,g] = diffusion_outer[g]
            removal[2,g] = removal_outer[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                chiNuFission[1,g,gpr] = chiNuFission_center[g,gpr]
                chiNuFission[2,g,gpr] = chiNuFission_outer[g,gpr]
                scatter[0,g,gpr] = scatter_inner[g,gpr]
                scatter[1,g,gpr] = scatter_center[g,gpr]
                scatter[2,g,gpr] = scatter_outer[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_outer

        # material zones
        #split = [0.0,45.0,35.0+45.0,20.0+35.0+45.0]
        split = [0.0,20.0,20.0+35.0,20.0+35.0+45.0]

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def PuCPu(G,R,I):
        #618 Group Problem:    Vacuum Boundary -- | -- 5 cm Carbon -- | -- 1.5 cm Enriched Plutonium -- | -- 3.5 cm Plutonium-240 -- | -- Reflected Boundary
        nMaterials = 3
        diffusion_outer,scatter_outer,chiNuFission_outer,removal_outer = Problems.compound(G,name='Pu_20pct240C_mat')
        diffusion_center,scatter_center,chiNuFission_center,removal_center = Problems.compound(G,name='Pu_239')
        diffusion_inner,scatter_inner,chiNuFission_inner,removal_inner = Problems.compound(G,name='Pu_240')
        # store material properties for all materials
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]

            diffusion[1,g] = diffusion_center[g]
            removal[1,g] = removal_center[g]

            diffusion[2,g] = diffusion_outer[g]
            removal[2,g] = removal_outer[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                chiNuFission[1,g,gpr] = chiNuFission_center[g,gpr]
                chiNuFission[2,g,gpr] = chiNuFission_outer[g,gpr]
                scatter[0,g,gpr] = scatter_inner[g,gpr]
                scatter[1,g,gpr] = scatter_center[g,gpr]
                scatter[2,g,gpr] = scatter_outer[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_outer

        # material zones
        Pu240length = 3.5
        Pu239length = 1.5
        Clength = 5.0
        split = [0.0,Pu240length,Pu240length+Pu239length,Pu240length+Pu239length+Clength]

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

    def PuCPuNoCrit(G,R,I):
        #618 Group Problem:    Vacuum Boundary -- | -- 5 cm Carbon -- | -- 0.75 cm Enriched Plutonium -- | -- 3.5 cm Plutonium-240 -- | -- Reflected Boundary
        nMaterials = 3
        diffusion_outer,scatter_outer,chiNuFission_outer,removal_outer = Problems.compound(G,name='Pu_20pct240C_mat')
        diffusion_center,scatter_center,chiNuFission_center,removal_center = Problems.compound(G,name='Pu_239')
        diffusion_inner,scatter_inner,chiNuFission_inner,removal_inner = Problems.compound(G,name='Pu_240')
        # store material properties for all materials
        diffusion = np.zeros((nMaterials,G))
        removal = np.zeros((nMaterials,G))
        scatter = np.zeros((nMaterials,G,G))
        chiNuFission = np.zeros((nMaterials,G,G))
        for g in range(G):
            diffusion[0,g] = diffusion_inner[g]
            removal[0,g] = removal_inner[g]

            diffusion[1,g] = diffusion_center[g]
            removal[1,g] = removal_center[g]

            diffusion[2,g] = diffusion_outer[g]
            removal[2,g] = removal_outer[g]
            for gpr in range(G):
                chiNuFission[0,g,gpr] = chiNuFission_inner[g,gpr]
                chiNuFission[1,g,gpr] = chiNuFission_center[g,gpr]
                chiNuFission[2,g,gpr] = chiNuFission_outer[g,gpr]
                scatter[0,g,gpr] = scatter_inner[g,gpr]
                scatter[1,g,gpr] = scatter_center[g,gpr]
                scatter[2,g,gpr] = scatter_outer[g,gpr]

        # Boundary Conditions
        BC = np.zeros((G,2)) + 0.25
        BC[:,1] = 0.5*diffusion_outer

        # material zones
        #split = [0.0,45.0,35.0+45.0,20.0+35.0+45.0]
        Pu239length = 0.5
        Pu240length = 3.5
        Clength = 5.0
        split = [0.0,Pu240length,Pu240length+Pu239length,Pu240length+Pu239length+Clength]

        return G,R,I,diffusion,scatter,chiNuFission,removal,nMaterials,split,BC

