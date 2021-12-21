from diffusion import Diffusion
import problems as dfp
import matplotlib.pyplot as plt

import create

import numpy as np
import re

I = [400]
NCells = I
geo = 'sphere'
problem = "light_water_reactor"
ranks = [8]

# determine energy group centers
G = [361]

for nG in G:
    for nX in NCells:
        # run full problem
        #variables = create.selection(problem,nG,nX)
        #example = Diffusion(*variables,geo=geo)
        #example.geometry()

        #D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb = example.create_matrix_DLRA_format_harmonic()

        #phi,keff,kHistory = example.solver_matrix_inv(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)

        # save to csv file
        #np.savetxt('output/phiProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', phi, delimiter=',')
        #np.savetxt('output/kEffectiveProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'.csv', kHistory, delimiter=',')
        for r in ranks:
            variables = create.selection(problem,nG,nX)
            example = Diffusion(*variables,geo=geo)
            example.geometry()
            example.r = r

            D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb = example.create_matrix_DLRA_format_harmonic()

            phiDLR,keffDLR,kHistoryDLR = example.solver_matrix_inv_DLR_harmonic(D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb)

            # save to csv file
            np.savetxt('output/phiDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', phiDLR, delimiter=',')
            np.savetxt('output/kEffectiveDLRProblem'+problem+'Nx'+str(nX)+'G'+str(nG)+'Rank'+str(r)+'.csv', kHistoryDLR, delimiter=',')

