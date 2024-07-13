# -*- coding: utf-8 -*-
"""
Autores: Rafael Maia & Jourdan Martins
"""

import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from mono_objetivo.ch3 import sol_inicial

# Será usada com atributos dinâmicos
class Struct:
    pass

# Importa os dados de usuário e os insere em um dataset
clients_df = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

'''
Mostra a distribuição dos clientes
'''
plt.figure(figsize=(8,8))
plt.scatter(clients_df['x'], clients_df['y'], alpha=0.8)
plt.title('Distribuição Espacial dos Clientes')
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.axis([0, 400, 0, 400])
plt.xticks(range(0, 401, 25))
plt.yticks(range(0, 401, 25))
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5) 
plt.show()
#plt.savefig('./images/distribuicao_clientes.png')
plt.close()

'''
Define os dados de uma instância arbitrária do problema
'''
def probdef():
    # Definições do problema
    max_pas = 30  # Número máximo de pontos de acesso
    pa_capacity = 54  # Capacidade de cada ponto de acesso em Mbps
    max_distance = 85  # Distância máxima de atendimento em metros
    lambda_exposure = 1  # Coeficiente de exposição (sinal nominal)
    gamma = 1  # Fator de decaimento
    
    # Criar uma instância de Struct para armazenar os dados do problema
    probdata = Struct()
    probdata.clients = clients_df
    probdata.max_pas = max_pas
    probdata.pa_capacity = pa_capacity
    probdata.max_distance = max_distance
    probdata.lambda_exposure = lambda_exposure
    probdata.gamma = gamma

    return probdata

'''
Implementa a função objetivo 1 do problema
'''
def fobj1(solution, probdata):
    # Inicializa o contador de PAs ativos
    active_pas = set()
    # Contabilizar o uso de cada PA e verificar as distâncias
    pa_bandwidth_usage = [0] * probdata.max_pas
    number_of_clients = len(probdata.clients)
    allowed_unserved = int(0.02 * number_of_clients)  # 2% dos clientes podem não ser servidos
    unserved_clients = 0
    total_penalty = 0  # Inicializa a penalidade total

    for client_index, pa_index in enumerate(solution.assignments):
        # Acesso as informações do cliente
        client = probdata.clients.iloc[client_index]
        pa = solution.pas[pa_index]

        # Verificar distância
        distance = np.sqrt((pa[0] - client['x'])**2 + (pa[1] - client['y'])**2)
        if distance > probdata.max_distance:
            unserved_clients += 1
            total_penalty += 1000  # Penalidade alta por não servir um cliente dentro da distância permitida

        # Adicionar ao uso de banda do PA
        pa_bandwidth_usage[pa_index] += client['bandwidth']
        if pa_bandwidth_usage[pa_index] > probdata.pa_capacity:
            # Penalidade proporcional ao excesso de capacidade
            excess = pa_bandwidth_usage[pa_index] - probdata.pa_capacity
            total_penalty += excess * 10

        # Marcar PA como ativo
        active_pas.add(pa_index)

    # Se mais de 2% dos clientes não foram servidos, a solução é considerada inválida
    if unserved_clients > allowed_unserved:
        total_penalty += 6000  # Penalidade adicional por exceder o limite de clientes não servidos

    # Atribui o número de PAs ativos à propriedade fitness, ajustado pela penalidade
    solution.fitness = len(active_pas) + total_penalty

    return solution

'''
Implementa a função objetivo 2 do problema
'''
def fobj2(solution, probdata):
    total_distance = 0
    unserved_clients = 0
    total_bandwidth_usage = [0] * len(solution.pas)
    pa_active = [False] * len(solution.pas)
    penalties = 0

    # Calcular distâncias e verificar restrições
    for client_index, pa_index in enumerate(solution.assignments):
        if pa_index is None:  # Cliente não servido
            unserved_clients += 1
            penalties += 5000  
            continue

        client = probdata.clients.iloc[client_index]
        pa = solution.pas[pa_index]
        distance = np.sqrt((pa[0] - client['x'])**2 + (pa[1] - client['y'])**2)

        # Verificar se o PA pode servir este cliente
        if distance > probdata.max_distance:
            unserved_clients += 1
            penalties += 10000  # Penalidade por distância excessiva
            continue

        # Acumular uso de banda e ativar PA
        total_bandwidth_usage[pa_index] += client['bandwidth']
        if total_bandwidth_usage[pa_index] > probdata.pa_capacity:
            unserved_clients += 1  # Excede a capacidade do PA
            penalties += total_bandwidth_usage[pa_index] - probdata.pa_capacity  # Penalidade pelo excesso de capacidade
            continue

        pa_active[pa_index] = True
        total_distance += distance  # Soma a distância se todas as condições forem atendidas

    # Verificar se o número de clientes não atendidos está dentro do limite permitido
    if unserved_clients > 0.02 * len(probdata.clients):
        penalties += 5000  # Penalidade adicional por alto número de clientes não atendidos

    # Verificar o número de PAs ativos
    if sum(pa_active) > probdata.max_pas:
        penalties += 10000  # Penalidade por excesso de PAs ativos

    # Cálculo do fitness
    solution.fitness = total_distance + penalties  # Minimizar a distância, maximizar a penalidade
    
    return solution

'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    if y.fitness < x.fitness or (y.fitness == x.fitness and np.random.rand() < 0.5):
        x = cp.deepcopy(y)
        k = 1
    else:
        k += 1
    return x, k

'''
Solução inicial
'''

'''
Implementa a função shake
'''
def shake(solution, k, probdata):
    np.random.seed()  # Garantir aleatoriedade

    new_solution = cp.deepcopy(solution)  # Cria uma cópia profunda da solução atual para modificar
    grid_spacing = 5  # Espaçamento do grid em metros
    area_width = 400  # Largura em metros
    area_height = 400  # Altura em metros

    if k == 1:  # Troca de Ativação de PA
        # Encontrar um PA ativo para desativar
        active_pas = [i for i, pa in enumerate(solution.pas) if any(pa)]
        if active_pas:
            deactivate_index = np.random.choice(active_pas)
            new_solution.pas[deactivate_index] = None  # Desativa o PA

            # Ativar um novo PA em uma posição livre do grid
            empty_positions = [(x, y) for x in range(0, area_width + 1, grid_spacing) 
                                        for y in range(0, area_height + 1, grid_spacing) 
                                        if (x, y) not in new_solution.pas]
            if empty_positions:
                # Escolhe um índice aleatório de posições vazias
                new_pa_position_index = np.random.randint(len(empty_positions))
                new_pa_position = empty_positions[new_pa_position_index]
                new_solution.pas[deactivate_index] = new_pa_position  # Ativa um novo PA

    elif k == 2:  # Realocação de PA's
        # Mover um PA para uma posição adjacente no grid
        pa_index = np.random.randint(len(new_solution.pas))
        directions = [(-grid_spacing, 0), (grid_spacing, 0), (0, -grid_spacing), (0, grid_spacing)]
        direction_index = np.random.randint(len(directions))  # Escolher um índice aleatório para a direção
        direction = directions[direction_index]
        new_x = max(0, min(area_width, new_solution.pas[pa_index][0] + direction[0]))
        new_y = max(0, min(area_height, new_solution.pas[pa_index][1] + direction[1]))
        new_solution.pas[pa_index] = (new_x, new_y)

    elif k == 3:  # Realocação aleatória de PA's
        # Mover um PA para uma posição completamente aleatória no grid
        pa_index = np.random.randint(len(new_solution.pas))
        new_x = np.random.choice(range(0, area_width + 1, grid_spacing))
        new_y = np.random.choice(range(0, area_height + 1, grid_spacing))
        new_solution.pas[pa_index] = (new_x, new_y)

    return new_solution

def bestImprovement(current_solution, kmax, probdata):
    best_solution = cp.deepcopy(current_solution)
    
    for i in range(1, kmax + 1):
        neighbor_solution = shake(best_solution, i, probdata) 
        if neighbor_solution.fitness < best_solution.fitness or (neighbor_solution.fitness == best_solution.fitness and np.random.rand() < 0.5):
            best_solution = neighbor_solution
    
    return best_solution

'''
Implementa meta-heurística BVNS
'''
times = 0
historico_fit_1 = []
historico_fit_2 = []
historico_fit_3 = []
historico_fit_4 = []
historico_fit_5 = []
len_historico_fit_1 = 0
len_historico_fit_2 = 0
len_historico_fit_3 = 0
len_historico_fit_4 = 0
len_historico_fit_5 = 0

func = int(input('Informe 1 para a função F1, e 2 para a função F2: '))

while times < 1:
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = 20

    # Número de estruturas de vizinhanças definidas
    kmax = 3

    probdata = probdef()

    # Gera uma solução inicial para o problema
    # x = sol_inicial(
    #     probdata, 
    #     use_constructive_heuristic=False, 
    #     qtd_pas_region_1 = 15,
    #     qtd_pas_region_2 = 5,
    #     qtd_pas_region_3 = 10)

    x = sol_inicial(probdata, clients_df=clients_df)

    # Avalia solução inicial
    if (func == 1):
        x = fobj1(x,probdata)
    elif (func == 2):
        x = fobj2(x,probdata)
        
    num_sol_avaliadas += 1

    # Armazena dados para plot
    historico = Struct()
    historico.sol = []
    historico.fit = []
    historico.sol.append(x.pas)
    historico.fit.append(x.fitness)

    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:
        k = 1
        while k <= kmax:
            
            # Gera uma solução candidata na k-ésima vizinhança de x        
            y = shake(x,k,probdata)
            y = fobj1(y,probdata)
            z = bestImprovement(y,3,probdata)
            num_sol_avaliadas += 1
            
            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            x,k = neighborhoodChange(x,z,k)
            
            # Armazena dados para plot
            historico.sol.append(x.pas)
            historico.fit.append(x.fitness)


    print('\n--- SOLUÇÃO INICIAL CONSTRUÍDA ---\n')
    print('Alocação dos PAs:\n')
    print('x = {}\n'.format(historico.sol[0]))
    print('fitness(x) = {:.1f}\n'.format(historico.fit[0]))

    print('\n--- MELHOR SOLUÇÃO ENCONTRADA ---\n')
    print('Alocação dos PAs:\n')
    print('x = {}\n'.format(x.pas))
    print('fitness(x) = {:.1f}\n'.format(x.fitness))
    print('Número de PAs ativos: ', len(list(set(x.assignments))))

    if times == 0:
        historico_fit_1 = historico.fit
        len_historico_fit_1 = len(historico.fit)
    elif times == 1:
        historico_fit_2 = historico.fit
        len_historico_fit_2 = len(historico.fit)
    elif times == 2:
        historico_fit_3 = historico.fit
        len_historico_fit_3 = len(historico.fit)
    elif times == 3:
        historico_fit_4 = historico.fit
        len_historico_fit_4 = len(historico.fit)
    else:
        historico_fit_5 = historico.fit
        len_historico_fit_5 = len(historico.fit)

    # Gráfico que mostre as ligações entre os clientes e os PAs
    plt.figure(figsize=(8,8))
    # Plotando os clientes
    plt.scatter(clients_df['x'], clients_df['y'], color='blue', label='Clientes', alpha=0.8)
    # Plotando os PA's
    x_pas, y_pas = zip(*x.pas)
    plt.scatter(x_pas, y_pas, color='red', marker='s', label='PA', alpha=0.8)
    # Desenhar as linhas conectando clientes e pontos de acesso
    clients_pas_location = [x.pas[x.assignments[i]] for i in x.assignments]
    #print('clients_pas: ', clients_pas)
    x_client = clients_df['x'].to_numpy()
    y_client = clients_df['y'].to_numpy()
    for client_index, pa_index in enumerate(x.assignments):
        client_x = x_client[client_index]
        client_y = y_client[client_index]
        pa_x = x.pas[pa_index][0]  # Acessando a coordenada x do PA
        pa_y = x.pas[pa_index][1]  # Acessando a coordenada y do PA

        plt.plot([client_x, pa_x], [client_y, pa_y], 'gray', linewidth=0.5, alpha=0.5)
    plt.title('Conexão de Clientes a Pontos de Acesso')
    plt.xlabel('Posição X')
    plt.ylabel('Posição Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    times += 1

plt.figure(figsize=(8,8))
plt.plot(np.linspace(0,len_historico_fit_1-1,len_historico_fit_1),historico_fit_1, color='red', label='Tentativa 1')
plt.plot(np.linspace(0,len_historico_fit_2-1,len_historico_fit_2),historico_fit_2, color='green', label='Tentativa 2')
plt.plot(np.linspace(0,len_historico_fit_3-1,len_historico_fit_3),historico_fit_3, color='blue', label='Tentativa 3')
plt.plot(np.linspace(0,len_historico_fit_4-1,len_historico_fit_4),historico_fit_4, color='black', label='Tentativa 4')
plt.plot(np.linspace(0,len_historico_fit_5-1,len_historico_fit_5),historico_fit_5, color='orange', label='Tentativa 5')
plt.title('Evolução da qualidade da solução');
plt.xlabel('Número de avaliações');
plt.ylabel('fitness(x)');
plt.legend()
plt.grid(True)
plt.show()













