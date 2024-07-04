import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class Solution:
    pass

def sol_inicial_f1(probdata, clients):
    # Inserir um PA aleatoriamente dentro do grid 400x400.
    # Atribuir a ele os clientes mais proximos, respeitando as restricoes.
    # Verificar se esse PA usa pelo menos 90% da sua capacidade maxima.
    # Em caso positivo, inserir mais um PA, ate o limite maximo de 30, e repetir o processo anterior.
    # Em caso negativo, mudar a posicao do PA em uma casa, 5 metros, em um direcao aleatoria, e repetir o processo acima.

    solution = Solution()
    solution.pas = []
    solution.assignments = []
    grid_spacing = 5  # Espaçamento do grid em metros
    max_radius = 85
    num_clients = 495
    minimum_clients_assigned = 0.98

    for pa_index in range(30):
        pa_pos_x = int(np.random.choice(np.arange(0, 401, grid_spacing)))
        pa_pos_y = int(np.random.choice(np.arange(0, 401, grid_spacing)))
        pa_bandwidth_usage = 0
        minimum_bandwidth_consumption = 0.9 * 54
        possible_assignments = []
        iterations = 0
        minimum_clients_assigned = 0.98

        while (pa_bandwidth_usage <= minimum_bandwidth_consumption):
            if (iterations >= 500):
                minimum_clients_assigned -= 0.01

            for index, client in clients.iterrows():
                try: 
                    if (solution.assignments[index] is not None):
                        pass
                except IndexError:
                    distance_pa_client = getDistance(pa_pos_x, pa_pos_y, client['x'], client['y'])
                    if (distance_pa_client <= max_radius and pa_bandwidth_usage <= (54 - client['bandwidth'])):
                        pa_bandwidth_usage += client['bandwidth']
                        possible_assignments.append(pa_index)
                    elif (pa_bandwidth_usage >= (54 - client['bandwidth'])):
                        break

            if (pa_bandwidth_usage <= minimum_bandwidth_consumption): 
                iterations += 1           
                # Atualizar coordenadas do PA
                direction = np.random.randint(0, 8)
                possible_assignments = []
                pa_bandwidth_usage = 0
                if (direction == 0):
                    pa_pos_y += grid_spacing
                elif (direction == 1):
                    pa_pos_x += grid_spacing
                elif (direction == 2):
                    pa_pos_y -= grid_spacing
                elif (direction == 3):
                    pa_pos_x -= grid_spacing
                elif (direction == 4):
                    pa_pos_x -= grid_spacing
                    pa_pos_y += grid_spacing
                elif (direction == 5):
                    pa_pos_x += grid_spacing
                    pa_pos_y += grid_spacing
                elif (direction == 6):
                    pa_pos_x += grid_spacing
                    pa_pos_y -= grid_spacing
                elif (direction == 7):
                    pa_pos_x -= grid_spacing
                    pa_pos_y -= grid_spacing

                # Limitar as coordenadas entre 0 e 400
                pa_pos_x = max(0, min(pa_pos_x, 400))
                pa_pos_y = max(0, min(pa_pos_y, 400))
        
        solution.pas.append((int(pa_pos_x), int(pa_pos_y)))
        solution.assignments.extend(possible_assignments)

        if (len(solution.assignments) >= (minimum_clients_assigned * num_clients)): # Informar aqui 98% do número máximo de clientes no centro de convenções
            break

    return solution.pas, solution.assignments

    

def getDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

clientes_dataset = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

pas, assignments = sol_inicial_f1(None, clientes_dataset)
print('PAs: ', pas)
print('Atribuições: ', assignments)


