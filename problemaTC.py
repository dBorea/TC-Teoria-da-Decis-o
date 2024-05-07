
#Importa os módulos usados

import numpy as np
import matplotlib.pyplot as plt
import copy


# Define um tipo de dado similar ao Pascal "record" or C "struct"

class Struct:
    pass

def probdef(nPA=80**2, nCL=495):
    
    probdata = Struct()

    probdata.CL_cons = 0# importar consumo dos clientes do .csv

    probdata.PA_cap = 54 # em Mbps
    probdata.PA_raio = 84 # em metros
    probdata.CL_min_p = 0.05 # porcentagem minima de clientes
    probdata.exp_coef = 1 # coeficiente de exposição
    probdata.falloff = 1 # fator de decaimento

    # CALCULAR DEPOIS
    probdata.dist_CL_PA = np.zeros((nCL, nPA)) # distância entre clientes e PAs
    probdata.exp_CL_PA = np.zeros((nCL, nPA)) # exposição dos clientes aos PAs

    probdata.nPA = nPA
    probdata.nCL = nCL
    probdata.nPA_max = 30

    return probdata

# Solução inicial
def sol_inicial(probdata, apply_heur):
    
    if True: #apply_heur == False: # EDITAR DEPOIS DE IMPLEMENTAR METAHEURISTICA
        x = Struct()
        x.solution = np.random.randint(0,1,size=(probdata.nCL, probdata.nPA))
        y = Struct()
        y.solution = np.random.randint(0,1,size=(1, probdata.nPA))
    
    else:
        x = Struct()
        x.solution = np.zeros((probdata.nCL, probdata.nPA))
        y = Struct()
        y.solution = np.zeros((probdata.nPA))

        # IMPLEMENTAR METAHEURÍSTICA CONSTRUTIVA AQUI
    
    return x, y

# Função de penalidades para todas as restrições
def penalties(x, y, probdata):

    # percentual mínimo de clientes
    pen_CLmin = probdata.nCL * probdata.CL_min_p - np.sum(x.solution) 
    pen_CLmin = np.sum(np.where(pen_CLmin <= 0, 0, pen_CLmin)**2)

    # limite de consumo dos PAs
    pen_PAcap = np.zeros(probdata.nPA)
    for i in probdata.nPA:
        pen_PAcap[i] = np.dot(np.array(probdata.CL_cons), np.array(x.solution[:[i]])) - y.solution[i] * probdata.PA_cap
        pen_PAcap[i] = np.sum(np.where(pen_PAcap[i] <= 0, 0, pen_PAcap[i])**2)
    
    # limite de distância entre PAs e clientes
    pen_dist = np.zeros((probdata.nCL, probdata.nPA))
    for i in probdata.nPA:
        for j in probdata.nCL:
            pen_dist[j,i] = probdata.dist_CL_PA[j,i] * x.solution[j,i] - y.solution[i] * probdata.PA_raio
            pen_dist[j,i] = np.sum(np.where(pen_dist[j,i] <= 0, 0, pen_dist[j,i])**2)

    # pelo menos 5% de exposição à rede
    pen_CLmin = np.zeros(probdata.nCL)
    for j in probdata.nCL:
        pen_CLmin[j] = 0.05 * 1 - np.dot(np.array(probdata.exp_CL_PA[j]), np.array(y.solution))
        pen_CLmin[j] = np.sum(np.where(pen_CLmin[j] <= 0, 0, pen_CLmin[j])**2)
    
    # no máximo um PA por cliente
    pen_PAperCL = np.zeros(probdata.nCL)
    for j in probdata.nCL:
        pen_PAperCL[j] = sum(x.solution[j]) - 1
        pen_PAperCL[j] = np.sum(np.where(pen_PAperCL[j] <= 0, 0, pen_PAperCL[j])**2)

    # número máximo de PAs
    pen_PAmax = sum(y.solution) - probdata.nPA_max

    # return all multiplied by U
    return 100 * (pen_CLmin + sum(pen_PAcap) + sum(pen_dist) + sum(pen_CLmin) + sum(pen_PAperCL) + pen_PAmax)


# Função objetivo 1: Minimizar número de PAs ativos
def fobj_minPA(x, y, probdata):

    sol = np.transpose(np.array(y.solution))

    y.fitness = np.sum(sol)
    y.penalidade = penalties(x, y, probdata)
    y.fitness_penalizado = y.fitness + y.penalidade

    return y


# Função objetivo 2: Minimizar distância cumulativa de clientes e PAs
def fobj_mindist(x, y, probdata):

    sol = x.solution

    fit_matrix = np.multiply(sol, probdata.dist_CL_PA)
    x.fitness = sum(fit_matrix)
    x.penalidade = penalties(x, y, probdata)
    x.fitness_penalizado = x.fitness + x.penalidade

    return x


# NeighborhoodChange implementation
def neighborhoodChange(x, xlinha, k):
    
    if xlinha.fitness_penalizado < x.fitness_penalizado:
        x = copy.deepcopy(xlinha)
        k  = 1
    else:
        k += 1
        
    return x, k


# Shake implementation // TODO: MUDAR HEURISTICAS
def shake(x,k,probdata):
    
    y = copy.deepcopy(x)
    r = np.random.permutation(probdata.n)       
    
    if k == 1:             # apply not operator in one random position
        y.solution[r[0]] = not(y.solution[r[0]])
        
    elif k == 2:           # apply not operator in two random positions        
        y.solution[r[0]] = not(y.solution[r[0]])
        y.solution[r[1]] = not(y.solution[r[1]])
        
    elif k == 3:           # apply not operator in three random positions
        y.solution[r[0]] = not(y.solution[r[0]])
        y.solution[r[1]] = not(y.solution[r[1]])
        y.solution[r[2]] = not(y.solution[r[2]])        
    
    return y

