import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

clients_df = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

class Solution:
    pass

def sol_inicial(probdata, clients_df = clients_df):
    solution = Solution()
    solution.pas = []
    solution.assignments = []
    grid_spacing = 5  # Espaçamento do grid em metros
    max_radius = 85
    num_clients = 495
    minimum_clients_assigned = 0.98
    minimum_bandwidth_consumption = 0.9 * 54
    clients = []
    possible_assignments = []

    for index, client in clients_df.iterrows():
        clients.append({'x': float(client['x']), 
                        'y': float(client['y']),
                        'bandwidth': float(client['bandwidth']),
                        'client_index': index,
                        'assigned': False})

    # Inicia uma posição aleatória para o PA dentro do grid de 400x400
    pa_pos_x = int(np.random.choice(np.arange(0, 401, grid_spacing)))
    pa_pos_y = int(np.random.choice(np.arange(0, 401, grid_spacing)))

    for _ in range(30):
        pa_bandwidth_usage = 0
        num_iterations = 0
        minimum_bandwidth_consumption = 0.9 * 54

        if (len(solution.assignments) >= (minimum_clients_assigned * num_clients)): # Informar aqui 98% do número máximo de clientes no centro de convenções
            break

        while(pa_bandwidth_usage <= minimum_bandwidth_consumption):
            pa_bandwidth_usage = 0
            num_iterations += 1
            if (num_iterations >= 500000):
                minimum_bandwidth_consumption -= 0.001

            # Obter as distâncias entre os clientes e o PA em questão
            distances = getDistancesClientsPA(pa_pos_x, pa_pos_y, clients)
            for obj in distances:
                if (pa_bandwidth_usage <= (54 - obj['bandwidth']) 
                    and obj['distance'] <= max_radius):
                    pa_bandwidth_usage += obj['bandwidth']
                    possible_assignments.append(obj)
                else:
                    break
            
            if (pa_bandwidth_usage <= minimum_bandwidth_consumption):       
                # Atualizar a posição
                possible_assignments = []
                pa_bandwidth_usage = 0
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)
            else:
                for possible_assignment in possible_assignments:
                    client = [client for client in clients if client['client_index'] == possible_assignment['client_index']]
                    client[0]['assigned'] = True
                solution.assignments.extend(possible_assignments)
                solution.pas.append((pa_pos_x, pa_pos_y))
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)

    return solution

def getDistancesClientsPA(pa_x, pa_y, clients):
    distances = []
    unassigned_clients = [client for client in clients if not client['assigned']]

    for client in unassigned_clients:
        distances.append({'distance': getDistance(pa_x, pa_y, float(client['x']), float(client['y'])), 
                          'x': float(client['x']), 'y': float(client['y']),
                          'bandwidth': float(client['bandwidth']),
                          'client_index': client['client_index']})

    distances = sorted(distances, key=lambda k: k['distance'])

    return distances

def getDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def UpdatePAPosition(x, y, grid_spacing = 5):
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

#print(getDistancesClientsPA(2,2,clients_df))
# clients_df['assigned'] = 0
# print(clients_df.head())

solution = sol_inicial(None, clients_df)
print('PAs: ', solution.pas)
print('Assignments: ', solution.assignments)