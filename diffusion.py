""" Multigroup Diffusion Code """
import create 

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class Diffusion:
    # Keywords Arguments allowed
    __allowed = ("geo")

    def __init__(self,G,R,I,D,scatter,fission,removal,nMaterials,split,BC,**kwargs):
        self.G = G # number of energy groups
        self.R = R # length of problem (cm)
        self.I = I # number of spatial cells
        self.nMaterials = nMaterials
        self.D = D # Diffusion Coefficient
        self.scatter = scatter # scatter XS
        self.fission = fission # nu * fission
        self.removal = removal # removal XS
        self.BC = BC # boundary conditions
        self.geo = 'sphere' # geometry (slab, sphere, cylinder)
        self.split = split
        self.chiNuFission = fission
        
        self.r = 5
        self.harmonic = True
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, track, geometry" 
            setattr(self, key, value)

    @classmethod
    def run(cls,problem,R,I,**kwargs):
        # This is for running the diffusion problems
        # Returns phi and keff
        attributes = create.selection(problem,G,I)
        initialize = cls(*attributes)
        # For geometry
        if 'geo' in kwargs:
            initialize.geo = kwargs['geo']
        # Create Geometry
        initialize.geometry()
        # Create lhs and rhs matrices
        A,B = initialize.create_matrix()
        # Solve for keff
        return initialize.solver(A,B)

    def geometry(self):
        """ Creates the grid and assigns the surface area and volume """
        # For the grid
        self.delta = float(self.R)/ self.I
        self.centers = np.arange(self.I) * self.delta + 0.5 * self.delta
        self.edges = np.arange(self.I + 1) * self.delta
        # for surface area and volume
        if (self.geo == 'slab'): 
            self.SA = 0.0*self.edges + 1 # 1 everywhere except at the left edge
            self.SA[0] = 0.0 #to enforce Refl BC
            self.V = 0.0*self.centers + self.delta # dr
        elif (self.geo == 'cylinder'): #
            self.SA = 2.0*np.pi*edges # 2pi r
            self.V = np.pi*(self.edges[1:(self.I+1)]**2 - self.edges[0:self.I]**2) # pi(r^2-r^2)
        elif (self.geo == 'sphere'): 
            self.SA = 4.0*np.pi*self.edges**2 # 4 pi^2        
            self.V = 4.0/3.0*np.pi*(self.edges[1:(self.I+1)]**3 - self.edges[0:self.I]**3) # 4/3 pi(r^3-r^3)

    def change_space(self,ii,gg): 
        """ Change the cell spatial position 
        include left and right of each
        """
        left = gg * (self.I+1) + ii - 1
        middle = gg * (self.I+1) + ii
        right = gg * (self.I+1) + ii + 1
        return left,middle,right

    def matrix2vector(self,mat):
        y = np.zeros(((self.I+1)*self.G))
        for i in range(self.I+1):
            for g in range(self.G):
                y[g * (self.I+1) + i] = mat[i,g]
        return y

    def indicator(self,r,material):
        if (r >= self.split[material] and r <= self.split[material+1]):
            return 1.0
        if r >= self.R and material == self.nMaterials-1:
            return 1.0
        if r <= 0.0 and material == 0:
            return 1.0
        else:
            return 0.0

    def create_matrix_DLRA_format_harmonic(self):
        """ Creates the left and right matrices for 1-D neutron diffusion eigenvalue problem
        of form Ax = (1/k)Bx with harmonic mean
        Returns:
            A: left hand side matrix - removal cross-section
            B: right hand side matrix - fission cross-section
        """
        D = np.zeros((self.nMaterials,self.nMaterials,self.I+1,self.I+1))
        M = np.zeros((self.nMaterials,self.nMaterials,self.G,self.G))
        Sigma = np.zeros((self.nMaterials,self.G,self.G))
        SigmaF = np.zeros((self.nMaterials,self.G,self.G))
        I = np.zeros((self.nMaterials,self.I+1,self.I+1))
        removal = np.zeros((self.nMaterials,self.G,self.G))
        delta = self.delta
        
        for l in range(self.nMaterials):
            # Iterate over energy groups
            for g in range(self.G):
                for k in range(self.nMaterials):
                    M[l,k,g,g] = self.D[l,g]*self.D[k,g]/(self.D[l,g]+self.D[k,g]) #(self.D[l,g]+self.D[k,g])/4#self.D[l,g]*self.D[k,g]/(self.D[l,g]+self.D[k,g])

                # removal matrix
                removal[l,g,g] = self.removal[l,g]

                #in scattering
                for gpr in range(self.G):
                    if (gpr != g): #skip the same group scattering
                        Sigma[l,g,gpr] = -self.scatter[l,g,gpr] #scattering diagonal
                    SigmaF[l,g,gpr] = self.chiNuFission[l,g,gpr] #set up the fission diagonal 

            # Iterate over spatial cells
            for ii in range(self.I):
                r = self.centers[ii] #determine the physical distance
                cell = ii
                minus = ii-1
                plus = ii+1
                for k in range(self.nMaterials):
                    # Matrix D
                    D[l,k,cell,cell] = self.indicator(r,l)*self.indicator(r + delta,k)*(self.indicator(r,l)+self.indicator(r+delta,k))/(delta * self.V[ii])*self.SA[ii+1]
                    D[l,k,cell,plus] = -self.indicator(r,l)*self.indicator(r + delta,k)*(self.indicator(r,l)+self.indicator(r+delta,k))/(delta * self.V[ii])*self.SA[ii+1] 
                    if ii > 0:
                        D[l,k,cell,minus] = -self.indicator(r,l)*self.indicator(r - delta,k)*(self.indicator(r,l)+self.indicator(r - delta,k))/(delta * self.V[ii]) * self.SA[ii] 
                        D[l,k,cell,cell] += self.indicator(r,l)*self.indicator(r - delta,k)*(self.indicator(r,l)+self.indicator(r - delta,k))/(delta * self.V[ii]) * self.SA[ii]

                # Matrix I
                I[l,ii,ii] = self.indicator(r,l)
            #r = self.centers[self.I-1]+delta
            #I[l,self.I,self.I] = self.indicator(r,l)

            # sets the boundary conditions
            B1 = np.zeros((self.I+1,self.I+1))
            B2 = np.zeros((self.I+1,self.I+1))
            B1[self.I,self.I] = 1.0
            B2[self.I,self.I-1] = 1.0
            Ba = np.zeros((self.G,self.G))
            Bb = np.zeros((self.G,self.G))
            for g in range(self.G):
                Ba[g,g] =  self.BC[g,0]*0.5 + self.BC[g,1]/delta
                Bb[g,g] =  self.BC[g,0]*0.5 - self.BC[g,1]/delta 

        return D,M,Sigma,SigmaF,removal,I,sparse.csr_matrix(B1),sparse.csr_matrix(B2),Ba,Bb

    def invertInvPowerOptHarmonic(self,D,M,Sigma,removal,I,B1,B2,Ba,Bb,rhs):
        A = np.zeros(((self.I+1)*self.G,(self.I+1)*self.G))
        y = np.zeros(((self.I+1)*self.G))

        # diagonal part in energy groups
        for i in range(self.I+1):
            for g in range(self.G):
                y[g * (self.I+1) + i] = rhs[i,g]
                for j in [i-1,i,i+1]:
                    if j > self.I or j<0:
                        continue
                    for l in range(self.nMaterials):
                        for k in range(self.nMaterials):
                            A[g * (self.I+1) + i,g * (self.I+1) + j] += D[l,k,i,j]*M[l,k,g,g]

        # diagonal part in spatial cells
        for gg in range(self.G):
            # Iterate over spatial cells
            for ii in range(self.I):
                for gpr in range(self.G):
                    for l in range(self.nMaterials):
                        A[gg * (self.I+1) + ii,gpr * (self.I+1) + ii] += I[l,ii,ii]*(Sigma[l,gg,gpr]+removal[l,gg,gpr]) #scattering diagonal

        # boundary terms
        i = self.I
        j = self.I-1
        for g in range(self.G):
            A[g * (self.I+1) + i,g * (self.I+1) + j] += B2[i,j]*Bb[g,g]

        for g in range(self.G):
            A[g * (self.I+1) + i,g * (self.I+1) + i] += B1[i,i]*Ba[g,g]

        phi = spsolve(sparse.csr_matrix(A), y)

        return np.reshape(phi,(self.I+1,self.G),order='F')

    def invertKStep(self,D,MHat,SigmaHat,removalHat,I,B1,B2,BaHat,BbHat,rhs,r):
        A = np.zeros(((self.I+1)*r,(self.I+1)*r))
        y = np.zeros(((self.I+1)*r))

        # diagonal part in energy groups
        for i in range(self.I+1):
            for g in range(r):
                y[g * (self.I+1) + i] = rhs[i,g]
                for j in [i-1,i,i+1]:
                    if j > self.I or j<0:
                        continue
                    for gpr in range(r):
                        for l in range(self.nMaterials):
                            if self.harmonic:
                                for k in range(self.nMaterials):
                                    A[g * (self.I+1) + i,gpr * (self.I+1) + j] += D[l,k,i,j]*MHat[l,k,gpr,g]
                            else:
                                A[g * (self.I+1) + i,gpr * (self.I+1) + j] += D[l,i,j]*MHat[l,gpr,g]

        # diagonal part in spatial cells
        for gg in range(r):
            # Iterate over spatial cells
            for ii in range(self.I):
                for gpr in range(r):
                    for l in range(self.nMaterials):
                        A[gg * (self.I+1) + ii,gpr * (self.I+1) + ii] += I[l,ii,ii]*(SigmaHat[l,gpr,gg]+removalHat[l,gpr,gg]) #scattering diagonal

        # boundary terms
        i = self.I
        j = self.I-1
        for g in range(r):
            for gpr in range(r):
                A[g * (self.I+1) + i,gpr * (self.I+1) + j] += B2[i,j]*BbHat[gpr,g]

        for g in range(r):
            for gpr in range(r):
                A[g * (self.I+1) + i,gpr * (self.I+1) + i] += B1[i,i]*BaHat[gpr,g]

        # solve
        K1 = spsolve(sparse.csr_matrix(A), y)

        return np.reshape(K1,(self.I+1,r),order='F')
        
    def invertLStep(self,DHat,M,Sigma,removal,IHat,B1Hat,B2Hat,Ba,Bb,rhs,r):
        A = np.zeros((self.G*r,self.G*r))
        y = np.zeros((self.G*r))

        # diagonal part in energy groups
        for i in range(r):
            for g in range(self.G):
                y[g * r + i] = rhs[i,g]
                for j in range(r):
                    for gpr in range(self.G):
                        for l in range(self.nMaterials):
                            if self.harmonic:
                                for k in range(self.nMaterials):
                                    A[g * r + i,gpr * r + j] += DHat[l,k,i,j]*M[l,k,gpr,g]
                            else:
                                A[g * r + i,gpr * r + j] += DHat[l,i,j]*M[l,gpr,g]

        # diagonal part in spatial cells
        for gg in range(self.G):
            # Iterate over spatial cells
            for ii in range(r):
                for j in range(r):
                    for gpr in range(self.G):
                        for l in range(self.nMaterials):
                            A[gg * r + ii,gpr * r + j] += IHat[l,ii,j]*(Sigma[l,gg,gpr]+removal[l,gg,gpr]) 

        # boundary terms
        for i in range(r):
            for j in range(r):
                for g in range(self.G):
                    for gpr in range(self.G):
                        A[g * r + i,gpr * r + j] += B2Hat[i,j]*Bb[gpr,g]

        for i in range(r):
            for j in range(r):
                for g in range(self.G):
                    for gpr in range(self.G):
                        A[g * r + i,gpr * r + j] += B1Hat[i,j]*Ba[gpr,g]

        # solve
        L1 = spsolve(sparse.csr_matrix(A), y)

        return np.reshape(L1,(self.G,r),order='C')

    def invertSStep(self,DHat,MHat,SigmaHat,removalHat,IHat,B1Hat,B2Hat,BaHat,BbHat,rhs,r):
        A = np.zeros((r*r,r*r))
        y = np.zeros((r*r))

        # diagonal part in energy groups
        for i in range(r):
            for g in range(r):
                y[g * r + i] = rhs[i,g]
                for j in range(r):
                    for gpr in range(r):
                        for l in range(self.nMaterials):
                            if self.harmonic:
                                for k in range(self.nMaterials):
                                    A[g * r + i,gpr * r + j] += DHat[l,k,i,j]*MHat[l,k,gpr,g]
                            else:
                                A[g * r + i,gpr * r + j] += DHat[l,i,j]*MHat[l,gpr,g]

        # diagonal part in spatial cells
        for gg in range(r):
            # Iterate over spatial cells
            for ii in range(r):
                for j in range(r):
                    for gpr in range(r):
                        for l in range(self.nMaterials):
                            A[gg * r + ii,gpr * r + j] += IHat[l,ii,j]*(SigmaHat[l,gpr,gg]+removalHat[l,gpr,gg]) 

        # boundary terms
        for i in range(r):
            for j in range(r):
                for g in range(r):
                    for gpr in range(r):
                        A[g * r + i,gpr * r + j] += B2Hat[i,j]*BbHat[gpr,g]

        for i in range(r):
            for j in range(r):
                for g in range(r):
                    for gpr in range(r):
                        A[g * r + i,gpr * r + j] += B1Hat[i,j]*BaHat[gpr,g]

        # solve
        L1 = spsolve(sparse.csr_matrix(A), y)

        return np.reshape(L1,(r,r),order='C')

    def solver_matrix_inv(self,D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb,tol=1E-8,MAX_ITS=200):
        """ Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue 
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """
        # Initialize Phi
        phi_old = np.random.rand((self.I+1),self.G)
        #phi_old = np.ones(((self.I+1),self.G))
        phi_old /= np.linalg.norm(phi_old)

        Rhs = np.zeros((self.I+1,self.G))
        kHistory = []

        converged = 0; count = 1
        while not(converged):
            Rhs = np.zeros((self.I+1,self.G))

            for l in range(self.nMaterials):
                Rhs += I[l,:,:]@phi_old@SigmaF[l,:,:].transpose()


            phi = self.invertInvPowerOptHarmonic(D,M,Sigma,removal,I,B1,B2,Ba,Bb,Rhs)

            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm(phi - phi_old)
            converged = (change < tol) or (count >= MAX_ITS)
            print('Iteration: {} Change {}\tKeff {}'.format(count,change,keff))

            count += 1
            phi_old = phi.copy()
            kHistory = np.append(kHistory,keff)
            #break

        #phi = np.reshape(phi,(self.I+1,self.G),order='F')
        return phi[:self.I+1],keff,kHistory
    
    def svd_reduce(matrix,rank):
        # Taking SVD of a matrix
        U, S, V = np.linalg.svd(matrix,full_matrices=True)        
        S[rank:] = 0
        return U @ np.diag(S) @ V

    def solver_matrix_inv_DLR_harmonic(self,D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb,tol=1E-8,MAX_ITS=200):
        """ Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue 
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """

        rank = self.r; # rank for DLRA

        # Initialize Phi
        phi_old = np.random.rand((self.I+1),self.G)
        #phi_old = np.ones(((self.I+1),self.G))
        phi_old /= np.linalg.norm(phi_old)

        U,S,VT = np.linalg.svd(phi_old,full_matrices=True)         # compute SVD of initial condition
        V = VT.transpose()
        
        # reshape SVD factors
        U0 = U[:,range(rank)]
        S0 = np.diag(S[range(rank)])
        V0 = V[:,range(rank)]
        
        keff_old = 100;
        phi = phi_old

        SigmaFHat = np.zeros((self.nMaterials,self.r,self.r))
        SigmaHat = np.zeros((self.nMaterials,self.r,self.r))
        IHat = np.zeros((self.nMaterials,self.r,self.r))
        DHat = np.zeros((self.nMaterials,self.nMaterials,self.r,self.r))
        removalHat = np.zeros((self.nMaterials,self.r,self.r))
        MHat = np.zeros((self.nMaterials,self.nMaterials,self.r,self.r))
        BaHat = np.zeros((self.r,self.r))
        BbHat = np.zeros((self.r,self.r))

        kHistory = []

        converged = 0; count = 1
        while not(converged):
            
            # K-step
            K0 = U0 @ S0
            for l in range(self.nMaterials):
                SigmaHat[l,:,:] = V0.transpose()@Sigma[l,:,:].transpose()@V0
                removalHat[l,:,:] = V0.transpose()@removal[l,:,:].transpose()@V0
                SigmaFHat[l,:,:] = V0.transpose()@SigmaF[l,:,:].transpose()@V0
                for k in range(self.nMaterials):
                    MHat[l,k,:,:] = V0.transpose()@M[l,k,:,:]@V0
            BaHat = V0.transpose()@Ba@V0
            BbHat = V0.transpose()@Bb@V0

            Rhs = np.zeros((self.I+1,self.r))
            for l in range(self.nMaterials):
                Rhs += I[l,:,:]@K0@SigmaFHat[l,:,:]
            K1 = self.invertKStep(D,MHat,SigmaHat,removalHat,I,B1,B2,BaHat,BbHat,Rhs,self.r)

            U1,R = np.linalg.qr(K1,mode='reduced')
            #U1 = U1[:,range(rank)]
            N = U1.transpose() @ U0

            # L-step
            L0 = V0 @ S0.transpose()
            for l in range(self.nMaterials):
                IHat[l,:,:] = U0.transpose()@I[l,:,:]@U0
                for k in range(self.nMaterials):
                    DHat[l,k,:,:] = U0.transpose()@D[l,k,:,:]@U0
            B1Hat = U0.transpose()@B1@U0
            B2Hat = U0.transpose()@B2@U0

            Rhs = np.zeros((self.r,self.G))
            for l in range(self.nMaterials):
                Rhs += IHat[l,:,:]@L0.transpose()@SigmaF[l,:,:].transpose()      

            L1 = self.invertLStep(DHat,M,Sigma,removal,IHat,B1Hat,B2Hat,Ba,Bb,Rhs,self.r)
            V1,R = np.linalg.qr(L1,mode='reduced')
            #V1 = V1[:,range(rank)]
            N2 = V1.transpose() @ V0

            # S-step
            S0 = N @ S0 @ N2.transpose()
            # recompute matrices
            for l in range(self.nMaterials):
                SigmaFHat[l,:,:] = V1.transpose()@SigmaF[l,:,:].transpose()@V1
                IHat[l,:,:] = U1.transpose()@I[l,:,:]@U1
                SigmaHat[l,:,:] = V1.transpose()@Sigma[l,:,:].transpose()@V1
                removalHat[l,:,:] = V1.transpose()@removal[l,:,:].transpose()@V1
                for k in range(self.nMaterials):
                    DHat[l,k,:,:] = U1.transpose()@D[l,k,:,:]@U1
                    MHat[l,k,:,:] = V1.transpose()@M[l,k,:,:]@V1
            B1Hat = U1.transpose()@B1@U1
            B2Hat = U1.transpose()@B2@U1
            BaHat = V1.transpose()@Ba@V1
            BbHat = V1.transpose()@Bb@V1
            
            Rhs = np.zeros((self.r,self.r))
            for l in range(self.nMaterials):
                Rhs += IHat[l,:,:]@S0@SigmaFHat[l,:,:]
            S1 = self.invertSStep(DHat,MHat,SigmaHat,removalHat,IHat,B1Hat,B2Hat,BaHat,BbHat,Rhs,self.r)                                        # compute rhs at phi = U1*S0*V1'
                          
            keff = np.linalg.norm(S1)
            
            S1 /= keff

            change = np.linalg.norm(S1 - S0)

            # postprocessing
            U0 = U1
            S0 = S1
            V0 = V1
            
            #change = np.abs(keff - keff_old)
            converged = (change < tol) or (count >= MAX_ITS)
            print('Iteration: {} Change {}\tKeff {}'.format(count,change,keff))

            count += 1
            keff_old = keff
            kHistory = np.append(kHistory,keff)

        phi = U0@S0@V0.T

        return phi[:self.I],keff,kHistory

    def solver_matrix_inv_DLR_adaptive(self,D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb,tol=1E-6,MAX_ITS=100,tolAdapt=1E-9):
        """ Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue 
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """

        rank = 2*self.r; # set rank for allocation to be twice the maximum rank

        # Initialize Phi
        phi_old = np.random.rand((self.I+1),self.G)
        #phi_old = np.ones(((self.I+1),self.G))
        phi_old /= np.linalg.norm(phi_old)

        U,S,VT = np.linalg.svd(phi_old,full_matrices=True)         # compute SVD of initial condition
        V = VT.transpose()

        r = 3#self.r
        rMaxTotal = self.r

        # adapt truncation tolerance according to current convergence level
        adaptTruncation = False
        
        # reshape SVD factors
        U0 = U[:,range(r)]
        S0 = np.diag(S[range(r)])
        V0 = V[:,range(r)]
        
        keff_old = 100;
        phi = phi_old

        SigmaFHat = np.zeros((self.nMaterials,rank,rank))
        SigmaHat = np.zeros((self.nMaterials,rank,rank))
        IHat = np.zeros((self.nMaterials,rank,rank))
        DHat = np.zeros((self.nMaterials,self.nMaterials,rank,rank))
        removalHat = np.zeros((self.nMaterials,rank,rank))
        MHat = np.zeros((self.nMaterials,self.nMaterials,rank,rank))
        BaHat = np.zeros((rank,rank))
        BbHat = np.zeros((rank,rank))

        kHistory = []

        change = 0.001

        converged = 0; count = 1
        while not(converged):
            
            # K-step
            K0 = U0 @ S0
            for l in range(self.nMaterials):
                SigmaHat[l,0:r,0:r] = V0.transpose()@Sigma[l,:,:].transpose()@V0
                removalHat[l,0:r,0:r] = V0.transpose()@removal[l,:,:].transpose()@V0
                SigmaFHat[l,0:r,0:r] = V0.transpose()@SigmaF[l,:,:].transpose()@V0
                for k in range(self.nMaterials):
                    MHat[l,k,0:r,0:r] = V0.transpose()@M[l,k,:,:]@V0
            BaHat = V0.transpose()@Ba@V0
            BbHat = V0.transpose()@Bb@V0

            Rhs = np.zeros((self.I+1,r))
            for l in range(self.nMaterials):
                Rhs += I[l,:,:]@K0@SigmaFHat[l,0:r,0:r]
            K1 = self.invertKStep(D,MHat,SigmaHat,removalHat,I,B1,B2,BaHat,BbHat,Rhs,r)

            K1 = np.concatenate((K1, U0), axis=1)
            U1,R = np.linalg.qr(K1,mode='reduced')
            #U1 = U1[:,:2*r]
            N = U1.transpose() @ U0

            # L-step
            L0 = V0 @ S0.transpose()
            for l in range(self.nMaterials):
                IHat[l,0:r,0:r] = U0.transpose()@I[l,:,:]@U0
                for k in range(self.nMaterials):
                    DHat[l,k,0:r,0:r] = U0.transpose()@D[l,k,:,:]@U0
            B1Hat = U0.transpose()@B1@U0
            B2Hat = U0.transpose()@B2@U0

            Rhs = np.zeros((r,self.G))
            for l in range(self.nMaterials):
                Rhs += IHat[l,0:r,0:r]@L0.transpose()@SigmaF[l,:,:].transpose() 
     

            L1 = self.invertLStep(DHat,M,Sigma,removal,IHat,B1Hat,B2Hat,Ba,Bb,Rhs,r)
            L1 = np.concatenate((L1, V0), axis=1)
            V1,R = np.linalg.qr(L1,mode='reduced')
            #V1 = V1[:,:2*r]
            N2 = V1.transpose() @ V0

            # S-step
            S0 = N @ S0 @ N2.transpose()
            # recompute matrices
            for l in range(self.nMaterials):
                SigmaFHat[l,:2*r,:2*r] = V1.transpose()@SigmaF[l,:,:].transpose()@V1
                IHat[l,:2*r,:2*r] = U1.transpose()@I[l,:,:]@U1
                SigmaHat[l,:2*r,:2*r] = V1.transpose()@Sigma[l,:,:].transpose()@V1
                removalHat[l,:2*r,:2*r] = V1.transpose()@removal[l,:,:].transpose()@V1
                for k in range(self.nMaterials):
                    DHat[l,k,:2*r,:2*r] = U1.transpose()@D[l,k,:,:]@U1
                    MHat[l,k,:2*r,:2*r] = V1.transpose()@M[l,k,:,:]@V1
            B1Hat = U1.transpose()@B1@U1
            B2Hat = U1.transpose()@B2@U1
            BaHat = V1.transpose()@Ba@V1
            BbHat = V1.transpose()@Bb@V1
            
            Rhs = np.zeros((2*r,2*r))
            for l in range(self.nMaterials):
                Rhs += IHat[l,0:2*r,0:2*r]@S0@SigmaFHat[l,0:2*r,0:2*r]
            S1 = self.invertSStep(DHat,MHat,SigmaHat,removalHat,IHat,B1Hat,B2Hat,BaHat,BbHat,Rhs,2*r)
            

            ################## truncate ##################

            # Compute singular values of S1 and decide how to truncate:
            U,Diag,VT = np.linalg.svd(S1,full_matrices=True)         # compute SVD of initial condition
            V = VT.transpose()
            rmax = -1
            S = np.zeros((S1.shape))


            tolA = tolAdapt*np.linalg.norm(Diag)

            tmp = 0.0
            rmax = 2*r
            
            for j in range(2*r):
                tmp = np.sqrt(sum(Diag[j:2*r])*sum(Diag[j:2*r]))
                if (tmp<tolA):
                    rmax = j
                    break

            
            rmax = min(rmax,rMaxTotal)
            rmax = max(rmax,3)

            for l in range(rmax):
                S[l,l] = Diag[l]

            # if 2*r was actually not enough move to highest possible rank
            if rmax == -1:
                rmax = rMaxTotal;

            # update solution with new rank
            U1 = U1@U
            V1 = V1@V

            # update solution with new rank
            #epsSolution = np.linalg.norm(S0-S1)
            #keff = np.linalg.norm(S)
            S0 = S[0:rmax,0:rmax]
            U0 = U1[:,0:rmax]
            V0 = V1[:,0:rmax]

            # update rank
            r = rmax
            keff = np.linalg.norm(S0)
            
            S0 /= keff
            
            change = np.abs(keff - keff_old)
            converged = (change < tol) or (count >= MAX_ITS)
            print('Iteration: {} Change {}\tKeff {} \tRank {}'.format(count,change,keff,r))

            count += 1
            keff_old = keff
            kHistory = np.append(kHistory,keff)
            #if adaptTruncation:
            #    tolAdapt = epsSolution*change*0.01#min(epsSolution*0.01,1E-9)

        phi = U0@S0@V0.T

        return phi[:self.I],keff,kHistory


    def solver_matrix_inv_DLR_harmonic_restart(self,D,M,Sigma,SigmaF,removal,I,B1,B2,Ba,Bb,phi_old,tol=1E-8,MAX_ITS=200):
        """ Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue 
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """

        rank = self.r; # rank for DLRA

        # Initialize Phi
        #phi_old = np.ones(((self.I+1),self.G))
        phi_old /= np.linalg.norm(phi_old)
        print(phi_old.shape)

        U,S,VT = np.linalg.svd(phi_old,full_matrices=True)         # compute SVD of initial condition
        V = VT.transpose()
        
        # reshape SVD factors
        U0 = U[:,range(rank)]
        S0 = np.diag(S[range(rank)])
        V0 = V[:,range(rank)]
        
        keff_old = 100;
        phi = phi_old

        SigmaFHat = np.zeros((self.nMaterials,self.r,self.r))
        SigmaHat = np.zeros((self.nMaterials,self.r,self.r))
        IHat = np.zeros((self.nMaterials,self.r,self.r))
        DHat = np.zeros((self.nMaterials,self.nMaterials,self.r,self.r))
        removalHat = np.zeros((self.nMaterials,self.r,self.r))
        MHat = np.zeros((self.nMaterials,self.nMaterials,self.r,self.r))
        BaHat = np.zeros((self.r,self.r))
        BbHat = np.zeros((self.r,self.r))

        kHistory = []

        converged = 0; count = 1
        while not(converged):
            
            # K-step
            K0 = U0 @ S0
            for l in range(self.nMaterials):
                SigmaHat[l,:,:] = V0.transpose()@Sigma[l,:,:].transpose()@V0
                removalHat[l,:,:] = V0.transpose()@removal[l,:,:].transpose()@V0
                SigmaFHat[l,:,:] = V0.transpose()@SigmaF[l,:,:].transpose()@V0
                for k in range(self.nMaterials):
                    MHat[l,k,:,:] = V0.transpose()@M[l,k,:,:]@V0
            BaHat = V0.transpose()@Ba@V0
            BbHat = V0.transpose()@Bb@V0

            Rhs = np.zeros((self.I+1,self.r))
            for l in range(self.nMaterials):
                Rhs += I[l,:,:]@K0@SigmaFHat[l,:,:]
            K1 = self.invertKStep(D,MHat,SigmaHat,removalHat,I,B1,B2,BaHat,BbHat,Rhs,self.r)

            U1,R = np.linalg.qr(K1,mode='reduced')
            #U1 = U1[:,range(rank)]
            N = U1.transpose() @ U0

            # L-step
            L0 = V0 @ S0.transpose()
            for l in range(self.nMaterials):
                IHat[l,:,:] = U0.transpose()@I[l,:,:]@U0
                for k in range(self.nMaterials):
                    DHat[l,k,:,:] = U0.transpose()@D[l,k,:,:]@U0
            B1Hat = U0.transpose()@B1@U0
            B2Hat = U0.transpose()@B2@U0

            Rhs = np.zeros((self.r,self.G))
            for l in range(self.nMaterials):
                Rhs += IHat[l,:,:]@L0.transpose()@SigmaF[l,:,:].transpose()      

            L1 = self.invertLStep(DHat,M,Sigma,removal,IHat,B1Hat,B2Hat,Ba,Bb,Rhs,self.r)
            V1,R = np.linalg.qr(L1,mode='reduced')
            #V1 = V1[:,range(rank)]
            N2 = V1.transpose() @ V0

            # S-step
            S0 = N @ S0 @ N2.transpose()
            # recompute matrices
            for l in range(self.nMaterials):
                SigmaFHat[l,:,:] = V1.transpose()@SigmaF[l,:,:].transpose()@V1
                IHat[l,:,:] = U1.transpose()@I[l,:,:]@U1
                SigmaHat[l,:,:] = V1.transpose()@Sigma[l,:,:].transpose()@V1
                removalHat[l,:,:] = V1.transpose()@removal[l,:,:].transpose()@V1
                for k in range(self.nMaterials):
                    DHat[l,k,:,:] = U1.transpose()@D[l,k,:,:]@U1
                    MHat[l,k,:,:] = V1.transpose()@M[l,k,:,:]@V1
            B1Hat = U1.transpose()@B1@U1
            B2Hat = U1.transpose()@B2@U1
            BaHat = V1.transpose()@Ba@V1
            BbHat = V1.transpose()@Bb@V1
            
            Rhs = np.zeros((self.r,self.r))
            for l in range(self.nMaterials):
                Rhs += IHat[l,:,:]@S0@SigmaFHat[l,:,:]
            S1 = self.invertSStep(DHat,MHat,SigmaHat,removalHat,IHat,B1Hat,B2Hat,BaHat,BbHat,Rhs,self.r)                                        # compute rhs at phi = U1*S0*V1'
                          
            keff = np.linalg.norm(S1)
            
            S1 /= keff

            change = np.linalg.norm(S1 - S0)

            # postprocessing
            U0 = U1
            S0 = S1
            V0 = V1
            
            #change = np.abs(keff - keff_old)
            converged = (change < tol) or (count >= MAX_ITS)
            print('Iteration: {} Change {}\tKeff {}'.format(count,change,keff))

            count += 1
            keff_old = keff
            kHistory = np.append(kHistory,keff)

        phi = U0@S0@V0.T

        return phi[:self.I],keff,kHistory
