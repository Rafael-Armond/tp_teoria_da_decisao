from typing import List, Dict, Tuple
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy as cp
from collections import defaultdict

clients_df = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

class Solution:
    pass

def probdef():
    pass

'''
Parâmetros de entrada:
probdata = Os dados do problema
clients_df = Dataframe com as coordenadas e consumo de banda de cada cliente

Retorno: Uma solução candidata (A primeira do problema)
'''
def sol_inicial(probdata, clients_df: pd.DataFrame = clients_df) -> Solution:
    start_time = time.time()

    solution = Solution()
    solution.pas = []
    solution.assignments = []
    grid_spacing = 5  # Espaçamento do grid em metros
    max_radius = 85
    num_clients = 495
    minimum_clients_assigned = 0.9
    minimum_bandwidth_consumption = 0.6 * 54
    solution.clients = getClientsListFromDF(clients_df) # Pode-se mover isso para o probdata (Que vai ser atribuído no probdef)
    possible_assignments = []

    # Inicia uma posição aleatória para o PA dentro do grid de 400x400
    pa_pos_x = int(np.random.choice(np.arange(0, 401, grid_spacing)))
    pa_pos_y = int(np.random.choice(np.arange(0, 401, grid_spacing)))

    for pa_index in range(30): # Mudar para receber o valor que vem do probdata
        pa_bandwidth_usage = 0
        num_iterations = 0
        minimum_bandwidth_consumption = 0.9 * 54

        if (len(solution.assignments) >= (minimum_clients_assigned * num_clients)): 
            break

        while(pa_bandwidth_usage <= minimum_bandwidth_consumption):
            pa_bandwidth_usage = 0
            num_iterations += 1
            if (num_iterations >= 500000):
                minimum_bandwidth_consumption -= 0.001

            # Obter as distâncias entre os clientes e o PA em questão
            distances = getDistancesClientsPA(pa_pos_x, pa_pos_y, solution.clients)
            for obj in distances:
                if (pa_bandwidth_usage <= (54 - obj['bandwidth']) 
                    and obj['distance'] <= max_radius):
                    pa_bandwidth_usage += obj['bandwidth']
                    obj['pa_index'] = pa_index
                    possible_assignments.append({'x_pa': pa_pos_x, 
                                                 'y_pa': pa_pos_y, 
                                                 'x_client': obj['x'], 
                                                 'y_client': obj['y'],
                                                 'pa_index': pa_index,
                                                 'client_index': obj['client_index'],
                                                 'distance_client_pa': obj['distance']}) 
                else:
                    break
            
            if (pa_bandwidth_usage <= minimum_bandwidth_consumption):
                possible_assignments = []
                pa_bandwidth_usage = 0
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)
            else:
                for possible_assignment in possible_assignments:
                    for client in solution.clients:
                        if client['client_index'] == possible_assignment['client_index']:
                            client['assigned'] = True
                            break

                solution.assignments.extend(possible_assignments)
                solution.pas.append((pa_pos_x, pa_pos_y))
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print("Tempo de execução: ", execution_time)

    return solution

'''
Isolar essa parte do código para ser reusada em outros pontos do problema.
Esse trecho constrói uma solução para o problema a partir da posição dos PAs e dos clientes.
Retorna uma solução para o problema.
'''
def rearrangeClientsAndPAs(solution, clients_list: List) -> List:
    solution.assignments = []

    for index, value in enumerate(solution.pas):
        pa_bandwidth_usage = 0
        distances = getDistancesClientsPA(value[0], value[1], clients_list)
        for obj in distances:
            if (pa_bandwidth_usage <= (54 - obj['bandwidth'])
                and obj['distance'] <= 85):
                solution.assignments.append({
                    'x_pa': value[0], 
                    'y_pa': value[1], 
                    'x_client': obj['x'], 
                    'y_client': obj['y'],
                    'pa_index': index,
                    'client_index': obj['client_index'],
                    'distance_client_pa': obj['distance']
                })
    
    return solution.assignments

'''
Parâmetros de entrada: Uma solução candidata
Retorno: Porcentagem de clientes não conectados
'''
def getPercentOfUnconnectedClients(assignments: List, clients_list: List) -> float:
    return (len(assignments)/len(clients_list)) * 100

'''
Parâmetros de entrada: Uma solução candidata
Retorno: Distância total entre clientes e os PAs aos quais estão atribuídos
'''
def getSumDistanceClientsAndPAs(assignments: List) -> int:
    return sum(assignment['distance_client_pa'] for assignment in assignments)

def getPenalityForUnservedClients(assignments: List, clients_list: List) -> int:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    porc_unserved_clients = getPercentOfUnconnectedClients(assignments, clients_list)
    if (porc_unserved_clients < 98 and porc_unserved_clients >= 90):
        penality += 5
    elif (porc_unserved_clients < 90 and porc_unserved_clients >= 80):
        penality += 8
    elif (porc_unserved_clients < 80 and porc_unserved_clients >= 70):
        penality += 13
    elif (porc_unserved_clients < 70 and porc_unserved_clients >= 60):
        penality += 21
    elif (porc_unserved_clients < 60 and porc_unserved_clients >= 50):
        penality += 34
    else:
        penality += 55

    return penality

'''
Função objetivo 1
Parâmetros de entrada: Uma solução candidata e os dados do problema
Retorno: Um número inteiro representando o fitnesse da solução em questão
'''
def fobj1(solution, probdata) -> int:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    penality += getPenalityForUnservedClients(solution.assignments, solution.clients)

    # Medir a quantidade de PAs que estão sendo usados (Relativo somente a função objetivo 1)
    qtd_pas = len(solution.pas)

    # Calcula o fitness da solução 
    fitness = qtd_pas + penality

    return fitness

'''
Função objetivo 2
Parâmetros de entrada: Uma solução candidata e os dados do problema
Retorno: Um número inteiro representando o fitnesse da solução em questão
'''
def fobj2(solution, probdata) -> float:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    penality += getPenalityForUnservedClients(solution.assignments, solution.clients)

    # Medir a distância total entre clientes e seus respectivos PAs
    total_distance = getSumDistanceClientsAndPAs(solution.assignments)

    # Calcula o fitness da solução
    fitness = total_distance + penality 

    return fitness

'''
Parâmetros de entrada: Coordenadas do PA e lista de clientes
Retorno: Vetor de objetos, que possuem informações dos clientes, ordenado pela propriedade 'distance'
'''
def getDistancesClientsPA(pa_x: float, pa_y: float, clients: List):
    distances = []
    unassigned_clients = [client for client in clients if not client['assigned']]

    for client in unassigned_clients:
        distances.append({'distance': getDistance(pa_x, pa_y, float(client['x']), float(client['y'])), 
                        'x': float(client['x']), 'y': float(client['y']),
                        'bandwidth': float(client['bandwidth']),
                        'client_index': client['client_index']})

    distances = sorted(distances, key=lambda k: k['distance'])

    return distances

def getDistance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def UpdatePAPosition(x, y, grid_spacing: int = 5) -> int:
    direction = np.random.randint(0, 8)
    if (direction == 0):
        y += grid_spacing
    elif (direction == 1):
        x += grid_spacing
    elif (direction == 2):
        y -= grid_spacing
    elif (direction == 3):
        x -= grid_spacing
    elif (direction == 4):
        x -= grid_spacing
        y += grid_spacing
    elif (direction == 5):
        x += grid_spacing
        y += grid_spacing
    elif (direction == 6):
        x += grid_spacing
        y -= grid_spacing
    elif (direction == 7):
        x -= grid_spacing
        y -= grid_spacing

    # Limitar as coordenadas entre 0 e 400
    x = max(0, min(x, 400))
    y = max(0, min(y, 400))

    return x, y

def getClientsListFromDF(clients_df: pd.DataFrame) -> List:
    clients = []

    for index, client in clients_df.iterrows():
        clients.append({'x': float(client['x']), 
                        'y': float(client['y']),
                        'bandwidth': float(client['bandwidth']),
                        'client_index': index,
                        'assigned': False})
    
    return clients

'''
Função neighborhood change
Compara o fitnesse de duas soluções e faz a troca baseado nisso.
Retorna a nova ou a antiga solução e também a nova ou antiga estrutura de vizinhança
'''
def neighborhoodChange(x, y, k):
    if y.fitness < x.fitness or (y.fitness == x.fitness and np.random.rand() < 0.5):
        x = cp.deepcopy(y)
        k = 1
    else:
        k += 1
    return x, k

'''
Retorna a posição de um cliente aleatório que não está servido por um PA.
'''
def getRandomUnservedClient(assignments: List[Dict], clients: List[Dict]) -> Tuple[float, float]:
    served_clients = {client['client_index'] for client in assignments}
    unserved_clients = [(client['x'], client['y']) for client in clients if client['client_index'] not in served_clients]
    
    client_index = np.random.randint(0, len(unserved_clients))

    return unserved_clients[client_index]

'''
Retorna a posição do PA, na lista de PAs, que está sendo menos utilizado.
'''
def getLessUsedPA(assignments: List[Dict], clients: List[Dict]) -> int:
    pas = [{'pa_index': obj['pa_index'], 'bd_consumption': clients[obj['client_index']]['bandwidth']} for obj in assignments]
    
    # Usando defaultdict para somar bd_consumption por pa_index
    bd_consumption_by_pa = defaultdict(float)

    for item in pas:
        bd_consumption_by_pa[item['pa_index']] += item['bd_consumption']

    bd_consumption_by_pa = dict(bd_consumption_by_pa)
    sorted_bd_consumption = sorted(bd_consumption_by_pa.items(), key=lambda x: x[1], reverse=True)

    return sorted_bd_consumption[0][0]

'''
Retorna uma coordenada no grid de posibilidades do PA, 5x5, que seja mais próxima relativamente 
a coordenada do cliente informada.
'''
def getPAPositionFromClientPosition(x_client: float, y_client: float) -> Tuple[int, int]:
    return (int(round(x_client/5)*5), int(round(y_client/5)*5))

'''
Função shake
Parâmetros de entrada: Uma solução candidata, uma estrutura de vizinhança e os dados do problema.
Retorno: Uma nova solução candidata
'''
def shake(solution, k, probdata):
    np.random.seed() 
    new_solution = cp.deepcopy(solution) 

    '''
    Estruturas de vizinhança:
    1 - Ativar um PA próximo a um cliente que não está servido por um PA.
    2 - Desativar o PA que está sendo menos usado.
    3 - Mover um PA aleatório em uma direção aleatória em um raio de 5 metros.
    4 - Mover um PA aleatório para um ponto aleatório do mapa.
    '''
    if k == 1:
        '''
        1- Encontrar um cliente que não está servido por um PA.
        2- Calcular as coordenadas onde o PA será inserido.
        3- Inserir um PA na coordenada encontrada no passo 2.
        4- Rearranjar a atribuição de clientes a PAs.
        '''
        unserved_client = getRandomUnservedClient(new_solution.assignments, probdata.clients)
        pa_position = getPAPositionFromClientPosition(unserved_client[0], unserved_client[1])
        new_solution.pas.append(pa_position)
        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 2:
        '''
        1- Encontrar o PA que está sendo menos utilizado.
        2- Retirar o PA da lista de PAs.
        3- Rearranjar a solução.
        '''
        pa_index = getLessUsedPA(new_solution.assignments, probdata.clients)
        new_solution.pas.pop(pa_index)
        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 3:
        '''
        1- Selecionar um número aleatório de 0 até o tamanho da lista de PAs.
        2- Atualizar a posição do PA selecionado.
        3- Rearranjar a solução.
        '''
        pa_index = np.random.randint(0, len(new_solution.pas))
        pa_x = new_solution[pa_index][0]
        pa_y = new_solution[pa_index][1]
        new_pa_x, new_pa_y = UpdatePAPosition(pa_x, pa_y)
        new_solution.pas[pa_index][0] = new_pa_x
        new_solution.pas[pa_index][1] = new_pa_y
        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 4:
        multiples_of_5 = np.arange(0, 401, 5)
        pa_index = np.random.randint(0, len(new_solution.pas))
        new_pa_x = np.random.choice(multiples_of_5)
        new_pa_y = np.random.choice(multiples_of_5)
        new_solution.pas[pa_index][0] = new_pa_x
        new_solution.pas[pa_index][1] = new_pa_y

    return new_solution

'''
Heurística de busca local.
'''
def bestImprovement(current_solution: Solution, kmax: int, probdata) -> Solution:
    best_solution = cp.deepcopy(current_solution)
    
    for i in range(1, kmax + 1):
        neighbor_solution = shake(best_solution, i, probdata) 
        if neighbor_solution.fitness < best_solution.fitness or (neighbor_solution.fitness == best_solution.fitness and np.random.rand() < 0.5):
            best_solution = neighbor_solution
    
    return best_solution

def plotSolution(solution: Solution) -> None:
    plt.figure(figsize=(10, 10))
    
    # Plotar os clientes
    for client in clients_df.itertuples():
        plt.scatter(client.x, client.y, c='blue', label='Cliente' if client.Index == 0 else "")
    
    # Plotar os PAs
    for pa in solution.pas:
        plt.scatter(pa[0], pa[1], c='red', marker='X', label='PA' if solution.pas.index(pa) == 0 else "")
    
    # Desenhar linhas entre PAs e seus clientes
    for assignment in solution.assignments:
        plt.plot([assignment['x_pa'], assignment['x_client']], 
                 [assignment['y_pa'], assignment['y_client']], 'k-', alpha=0.2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clientes e Pontos de Acesso (PAs)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Gerar a solução
solution = sol_inicial(None, clients_df)
#solution = Solution()
# Testando a estrutura de vizinhança número 1 (Adicionar um PA, próximo a um cliente não conectado)
#solution.pas = [(145, 30), (85, 60), (255, 115), (390, 195), (180, 240), (100, 335), (275, 300), (110, 220), (340, 35)] 
#result = rearrangeClientsAndPAs(solution, getClientsListFromDF(clients_df))
print('Posição dos PAs: ', solution.pas)
print('Atribuições de clientes a PAs: ', solution.assignments)
# Plotar os resultados
plotSolution(solution)

'''
Implementa meta-heurística BVNS
'''
