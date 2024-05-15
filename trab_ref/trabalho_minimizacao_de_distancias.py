'''
Importa os módulos usados
'''
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy
from typing import Dict, List
import random

'''
Define um tipo de dado similar ao Pascal "record" or C "struct"
'''
class Struct:
    pass

NAO_DEFINIDO: int = -1

class Cliente:
    x: float
    y: float
    consumo_de_banda: float
    id: int
    ponto_de_acesso_id: int = NAO_DEFINIDO
    
    def __init__(self, x: str, y: str, consumo_de_banda: str) -> None:
        self.x = float(x)
        self.y = float(y)
        self.consumo_de_banda = float(consumo_de_banda)
        self.id = id

def ler_clientes_do_arquivo_csv() -> List[Cliente]:
    with open('clientes.csv', newline='') as csvfile:
        leitor_csv = csv.reader(csvfile)
        clientes = list()
        
        for [x, y, consumo_de_banda] in leitor_csv:
            clientes.append(Cliente(x, y, consumo_de_banda))
        
        clientes = sorted(clientes, key= lambda cliente: cliente.consumo_de_banda)
        
        index = 0
        for cliente in clientes:
            cliente.id = index
            index += 1
        
    return clientes

class PontoDeAcesso:
    x: int = 0
    y: int = 0
    id: int = 0
    numero_de_clientes_acessiveis: int = 0
    
    def __init__(self, x: int, y: int, index: int) -> None:
        self.x = x
        self.y = y
        self.id = index
        self.numero_de_clientes_acessiveis: int = 0
        pass

class Solucao:
    id: int = 0
    fitness: float = 0
    penalidade: float = 0
    fitness_penalizado: float = 0
    solution: Dict[int, int] = {}
    cliente_para_ponto_de_acesso: Dict[int, int] = {}
    clientes_atendidos_em_porcentagem = 0
    numero_de_clientes_por_ponto_de_acesso: Dict[int, int] = {}
    
    def __init__(self) -> None:
        self.id: int = 0
        self.fitness: float = 0
        self.penalidade: float = 0
        self.fitness_penalizado: float = 0
        self.solution: Dict[int, int] = {}
        self.cliente_para_ponto_de_acesso: Dict[int, int] = {}
        self.clientes_atendidos_em_porcentagem = 0
        self.numero_de_clientes_por_ponto_de_acesso: Dict[int, int] = {} 

class DadosDoProblema:
    pontos_de_acessos: List[PontoDeAcesso] = []
    clientes: List[Cliente] = []
    distancia_pa_por_cliente: Dict[int, Dict[int, float]] = np.empty((6400,256))
    taxa_minima_de_clientes: float = 0.97
    numero_de_clientes: int = 256

def definir_dados_do_problema() -> DadosDoProblema:
    
    dados = DadosDoProblema()
    index = 0
    for x in range (0, 400, 5):
        for y in range(0, 400, 5):
            dados.pontos_de_acessos.append(PontoDeAcesso(x, y, index))
            index += 1
            
    dados.clientes = ler_clientes_do_arquivo_csv()
    
    for ponto_de_acesso in dados.pontos_de_acessos:
        for cliente in dados.clientes:
            distancia_x = (ponto_de_acesso.x - cliente.x) ** 2
            distancia_y = (ponto_de_acesso.y - cliente.y) ** 2
            dados.distancia_pa_por_cliente[ponto_de_acesso.id][cliente.id] = (distancia_x + distancia_y) ** 0.5
            
            if dados.distancia_pa_por_cliente[ponto_de_acesso.id][cliente.id] <= 70:
                ponto_de_acesso.numero_de_clientes_acessiveis += 1
    
    return dados

def ativar_ponto_de_acesso_sol(ponto_de_acesso_id: int, clientes_sol: Dict[int, int], dados: DadosDoProblema):
    soma_do_consumo = 0

    for cliente in dados.clientes:
        # se o ponto ainda nao esta atendido
        cliente_nao_atendido = clientes_sol[cliente.id] == NAO_DEFINIDO
        # se o ponto nao fara com que o ponto de acesso fique sobrecarregado com a banda necessária
        banda_suportada = (soma_do_consumo + cliente.consumo_de_banda) <= 54
        # se o ponto esta dentro do alcance do PA
        cliente_dentro_do_range = dados.distancia_pa_por_cliente[ponto_de_acesso_id][cliente.id] <= 70
        
        if ( cliente_nao_atendido and banda_suportada and cliente_dentro_do_range ):
            cliente.ponto_de_acesso_id = ponto_de_acesso_id
            soma_do_consumo += cliente.consumo_de_banda
            clientes_sol[cliente.id] = ponto_de_acesso_id


def criar_solucao_inicial(dados: DadosDoProblema):
    x: Solucao = Solucao()
    
    for ponto_de_acesso_id in range(0, 6400, 256):
        x.solution[ponto_de_acesso_id] = ponto_de_acesso_id

    return x

def funcao_objetivo_distancias(x: Solucao, dados_do_problema: DadosDoProblema):
    
    # criando o dicionário que irá receber a relação de cliente para PA
    cliente_atendido_por_pa: Dict[int, int] = {}
    
    # criando a posição dos clientes no dicionário com um valor padrão
    for cliente in dados.clientes:
        cliente_atendido_por_pa[cliente.id] = NAO_DEFINIDO
    
    # ativando cada PA para a solução
    for ponto_de_acesso_id in x.solution:
        ativar_ponto_de_acesso_sol(ponto_de_acesso_id, cliente_atendido_por_pa, dados_do_problema)
    
    # relacionando o número de PA's em relação ao número de clientes que ele atende
    x.numero_de_clientes_por_ponto_de_acesso: Dict[int, int] = {}
    numero_de_clientes_atendidos: int = 0
    
    for cliente in dados.clientes:
        if cliente_atendido_por_pa[cliente.id] != NAO_DEFINIDO:
            numero_de_clientes_atendidos += 1
            if cliente_atendido_por_pa[cliente.id] in  x.numero_de_clientes_por_ponto_de_acesso:
                x.numero_de_clientes_por_ponto_de_acesso[cliente_atendido_por_pa[cliente.id]] += 1
            else:
                x.numero_de_clientes_por_ponto_de_acesso[cliente_atendido_por_pa[cliente.id]] = 1
    
    
    # calculando a restrição de número minímo de clientes disponíveis
    restricao_numero_de_clientes_minimo = -numero_de_clientes_atendidos + dados_do_problema.numero_de_clientes * dados_do_problema.taxa_minima_de_clientes
    restricao_numero_de_clientes_minimo = 0 if restricao_numero_de_clientes_minimo < 0 else restricao_numero_de_clientes_minimo

    # calculando o número minímo de PA's disponíveis
    restricao_numero_maximo_de_PAs = (len(x.solution) - 10)
    restricao_numero_maximo_de_PAs = 0 if restricao_numero_maximo_de_PAs < 0 else restricao_numero_maximo_de_PAs    
    
    distancias: float = 0
    
    for cliente_dist_id, ponto_de_acesso_dist_id in cliente_atendido_por_pa.items():
        distancias += dados.distancia_pa_por_cliente[ponto_de_acesso_dist_id][cliente_dist_id]
    
    restricao_distancia_maxima = distancias / numero_de_clientes_atendidos - 20
    restricao_distancia_maxima = 0 if restricao_distancia_maxima < 0 else restricao_distancia_maxima
    
    x.fitness = distancias  
    x.penalidade = 200*restricao_distancia_maxima**2 + 100* restricao_numero_de_clientes_minimo**2 + 10*restricao_numero_maximo_de_PAs**2
    x.fitness_penalizado = x.fitness + x.penalidade
    x.clientes_atendidos_em_porcentagem = numero_de_clientes_atendidos / 256
    x.cliente_para_ponto_de_acesso = cliente_atendido_por_pa
    
    return x

def shake(x: Solucao, k: int, dados: DadosDoProblema):
    y = copy.deepcopy(x)
    
    index = 0
    keys = list(x.solution.keys())
    generated_keys: Dict[int, int] = {}
    while index < k:
        key = random.choice(keys)
        while (key in generated_keys):
            key = random.choice(keys)
        
        generated_keys[key] = 0
    
        y.solution.pop(key)
        new_key = int(6400 * random.random())
        y.solution[new_key] = new_key
        index += 1
    
    if len(y.solution) <= 18:
        while len(y.solution) < 25:
            key = int(6400 * random.random())
            if not (key in y.solution):
                y.solution[key] = key
    
    return y

def neighborhoodChange(x: Solucao, y: Solucao, k: int):
    
    if  (y.fitness_penalizado < x.fitness_penalizado or y.fitness_penalizado == x.fitness_penalizado and y.clientes_atendidos_em_porcentagem > x.clientes_atendidos_em_porcentagem):
        x = copy.deepcopy(y)
        k  = 1
    else:
        k += 1
        
    return x, k   

def find_solution_using_RVNS(solucao_inicial: Solucao, dados: DadosDoProblema, historico: Struct) -> Solucao:
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 1

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = 2000

    # Número de estruturas de vizinhanças definidas
    kmax = 3

    # Gera solução inicial
    x = solucao_inicial

    # Armazena dados para plot
    historico.fit.append(x.fitness)
    historico.sol.append(x.solution)
    historico.pen.append(x.penalidade)
    historico.fit_pen.append(x.fitness_penalizado)


    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:
        
        k = 1
        while k <= kmax:
            
            # Gera uma solução candidata na k-ésima vizinhança de x          
            y = shake(x, k, dados)
            y = funcao_objetivo_distancias(y, dados)
            num_sol_avaliadas += 1
            y.id = num_sol_avaliadas
            
            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            x, k = neighborhoodChange(x, y, k)
            
            # Armazena dados para plot
            historico.fit.append(x.fitness)
            historico.sol.append(x.solution)
            historico.pen.append(x.penalidade)
            historico.fit_pen.append(x.fitness_penalizado)
    
    return x
        
dados = definir_dados_do_problema()
solucao_inicial = criar_solucao_inicial(dados)

def BVNS(solucao_inicial: Solucao, dados: DadosDoProblema) -> [Solucao, Struct]:

    x = funcao_objetivo_distancias(solucao_inicial, dados)
    y = copy.deepcopy(x)

    historico = Struct()
    historico.fit = []
    historico.sol = []
    historico.pen = []
    historico.fit_pen = []

    numero_de_tentativas = 0
    while numero_de_tentativas < 6:
        numero_de_iteracoes_RVNS = 0
        while x.fitness_penalizado >= y.fitness_penalizado and numero_de_iteracoes_RVNS < 5:
            numero_de_iteracoes_RVNS += 1
            x = find_solution_using_RVNS(y, dados, historico)
            
        y = copy.deepcopy(x)
        numero_de_tentativas += 1

    
    return x, historico

numero_de_execucoes = 0

melhores_solucoes: List[Solucao] = []
historicos_de_solucoes: List[Struct] = []

while numero_de_execucoes < 5:
    melhor_solucao, historico = BVNS(solucao_inicial, dados)
    print(f'execução = {numero_de_execucoes} PAs = {melhor_solucao.fitness} Porcentagem de Clientes Atendidos = {melhor_solucao.clientes_atendidos_em_porcentagem} penalização = {melhor_solucao.fitness_penalizado}')
    historicos_de_solucoes.append(historico)
    melhores_solucoes.append(melhor_solucao)
    numero_de_execucoes += 1


fig, (ax1, ax2) = plt.subplots(2, 1)
cores_historico = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(numero_de_execucoes)]
index_historico = 0
for historico in historicos_de_solucoes:
    s = len(historico.fit_pen)
    ax1.plot(np.linspace(0,s-1,s),historico.fit_pen,'k-', color=cores_historico[index_historico])
    ax2.plot(np.linspace(0,s-1,s),historico.pen,'b:', color=cores_historico[index_historico])
    index_historico += 1
    
fig.suptitle('Evolução da qualidade da solução candidata')
ax1.set_ylabel('fitness(x) penalizado')
ax2.set_ylabel('penalidade(x)')
ax2.set_xlabel('Número de avaliações')
plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
plt.show()



def plot_triangulo(x, y, size, color):
    plt.plot(x, y, marker='^', markersize=size, color=color, linestyle='None')

# Função para plotar um círculo
def plot_circulo(x, y, size, color):
    plt.plot(x, y, marker='o', markersize=size, color=color, linestyle='None')
    

for x in melhores_solucoes:
    # Gerar uma lista de cores aleatórias para os pontos de acesso
    cores_pontos_acesso = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(x.solution))]

    cores: Dict[int, str] = {}

    index = 0
    for ponto_de_acesso_id in x.numero_de_clientes_por_ponto_de_acesso:
        cores[ponto_de_acesso_id] = cores_pontos_acesso[index]
        index += 1

    pontos_de_acessos_ativos: Dict[int, float] = {}
        
    for cliente_id, ponto_de_acesso_id in x.cliente_para_ponto_de_acesso.items():
        if ponto_de_acesso_id != NAO_DEFINIDO:
            plot_triangulo(dados.pontos_de_acessos[ponto_de_acesso_id].x, dados.pontos_de_acessos[ponto_de_acesso_id].y, 10, color = cores[ponto_de_acesso_id])
            plot_circulo(dados.clientes[cliente_id].x, dados.clientes[cliente_id].y, 5, color = cores[ponto_de_acesso_id])

    # Configurar o gráfico
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    label = f'Número de PAs {int(len(x.solution))}\n Média das Distâncias {x.fitness / (x.clientes_atendidos_em_porcentagem * 256)}\n Clientes Atendidos {x.clientes_atendidos_em_porcentagem}'
    plt.title(f'Pontos de Acesso e Clientes Atendidos \n{label}')
    plt.grid(True)

    # Mostrar o gráfico
    plt.show()