""" Setting up diffusion problems """


import numpy as np
from collections import OrderedDict
import json
import pkg_resources
import importlib.resources as importlib_resources

# DATA_PATH = pkg_resources.resource_filename('NeutronDiffusion','data/')

DATA_PATH = 'data/'

problem_dictionary = json.load(open(DATA_PATH+'problem_setup.json','r'))

def selection(problem_name,G,I):
    # Call the ordered dictionary
    problem = problem_dictionary[problem_name]
    
    R = []
    diffusion = []
    scatter = []
    chi = []
    fission = []
    removal = []
    interfaces = [0.0]

    for mat,r in problem.items():
        t1, t2, t3, t4, t5 = loading_data(G,mat)
        diffusion.append(t1)
        scatter.append(t2)
        chi.append(t3)
        fission.append(t4)
        removal.append(t5)
        R.append(r)
        interfaces.append(interfaces[-1]+r)

    BC = np.zeros((G,2)) + 0.25
    BC[:,1] = 0.5*diffusion[-1] # outer edge

    diffusion = np.array(diffusion)
    scatter = np.array(scatter)
    #chi = np.array(chi)
    #fission = np.array(fission)
    removal = np.array(removal)
    interfaces = np.array(interfaces)

    nMaterials = removal.shape[0]

    chiNuFission = np.zeros((nMaterials,G,G))
    for g in range(G):
        for gpr in range(G):
            for l in range(nMaterials):
                if len(fission[l].shape) == 1:
                    chiNuFission[l,g,gpr] = chi[l][g]*fission[l][gpr] #set up the fission diagonal 
                else: # full matrix fission
                    chiNuFission[l,g,gpr] = fission[l][g,gpr]

    return G,float(interfaces[-1]),I,diffusion,scatter,chiNuFission,removal,nMaterials,interfaces,BC

def loading_data(G,material):
    diffusion = np.loadtxt('{}D_{}G_{}.csv'.format(DATA_PATH,G,material))
    absorb = np.loadtxt('{}Siga_{}G_{}.csv'.format(DATA_PATH,G,material))
    scatter = np.loadtxt('{}Scat_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    centers = np.loadtxt('{}group_centers_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    if np.argmax(centers) == 0:
        removal = [absorb[gg] + np.sum(scatter,axis=0)[gg] - scatter[gg,gg] for gg in range(G)]
    else:
        removal = [absorb[gg] + np.sum(scatter,axis=1)[gg] - scatter[gg,gg] for gg in range(G)]
    if material != 'SHEM':
        np.fill_diagonal(scatter,0)
    else:
        removal =np.loadtxt('{}Removal_{}G_{}.csv'.format(DATA_PATH,G,material))
    # Some have fission matrix, others have birth rate and fission vector
    try:
        chi = np.loadtxt('{}chi_{}G_{}.csv'.format(DATA_PATH,G,material))
        fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material))
    except OSError:
        chi = None
        fission = np.loadtxt('{}nuSigf_{}G_{}.csv'.format(DATA_PATH,G,material),delimiter=',')
    return diffusion,scatter,chi,fission,removal

def add_problem_to_file(layers,materials,name):
    """ Adding problem to json for easy access 
    Inputs:
        layers: list of numbers (width of each material)
        materials: list of string (for loading csv data)
        name: string of name to be called 
    Returns:
        status string    """

    # Load current dictionary
    with open('problem_setup.json','r') as fp:
        problems = json.load(fp)

    # Check to see duplicate names
    if name in problems.keys():
        return "Name already exists"

    # Working inside (center) outward to edge
    od = OrderedDict()
    for layer,material in zip(layers,materials):
        od[material] = layer

    # Add to existing dictionary
    problems[name] = od

    # Save new dictionary
    with open('problem_setup.json', 'w') as fp:
        json.dump(problems, fp, sort_keys=True, indent=4)

    return "Successfully Added Element"

    
