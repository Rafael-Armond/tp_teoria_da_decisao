import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy as cp
from collections import defaultdict
from typing import List, Dict, Tuple

# Valores mínimos e máximos esperados para normalização
min_f1 = 0
max_f1 = 100
min_f2 = 0
max_f2 = 100000

# Função para carregar dados de clientes
def carregar_dados_clientes(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, header=None, names=['x', 'y', 'bandwidth'])

# Normalização dos valores de fitness
def normalizar(valor, min_val, max_val):
    return (valor - min_val) / (max_val - min_val)

# Estrutura da solução
class Solucao:
    pass

class Estrutura:
    pass

# Definição do problema
def definir_problema(df_clientes: pd.DataFrame) -> Estrutura:
    problema = Estrutura()
    problema.clientes = converter_df_para_lista(df_clientes)
    problema.max_pas = 30
    problema.max_distancia = 85
    problema.capacidade_max_pa = 54
    problema.espacamento_grid = 5
    problema.largura_area = 400
    problema.altura_area = 400
    problema.num_clientes = 495
    return problema

# Função para converter DataFrame para lista de dicionários
def converter_df_para_lista(df: pd.DataFrame) -> List:
    lista_clientes = []
    for indice, cliente in df.iterrows():
        lista_clientes.append({'x': float(cliente['x']), 
                               'y': float(cliente['y']),
                               'bandwidth': float(cliente['bandwidth']),
                               'client_index': indice,
                               'assigned': False})
    return lista_clientes

# Função para inicializar a solução
def inicializar_solucao(problema, pesos) -> Solucao:
    inicio_tempo = time.time()
    solucao = Solucao()
    solucao.pas = []
    solucao.atribuicoes = []
    esp_grid = problema.espacamento_grid 
    max_raio = problema.max_distancia
    num_clientes = problema.num_clientes
    min_clientes_atribuidos = 0.7
    min_consumo_bandwidth = ((0.7 * pesos[0]) * 54)  
    solucao.clientes = problema.clientes
    possiveis_atribuicoes = []

    desatribuir_clientes(solucao.clientes)

    pos_pa_x = int(np.random.choice(np.arange(0, 401, esp_grid)))
    pos_pa_y = int(np.random.choice(np.arange(0, 401, esp_grid)))

    for indice_pa in range(30):
        uso_bandwidth_pa = 0
        num_iteracoes = 0

        if len(solucao.atribuicoes) >= (min_clientes_atribuidos * num_clientes): 
            break

        while uso_bandwidth_pa <= min_consumo_bandwidth:
            uso_bandwidth_pa = 0
            num_iteracoes += 1
            if num_iteracoes >= 50000:
                min_consumo_bandwidth -= 0.01

            distancias = calcular_distancias_pa_clientes(pos_pa_x, pos_pa_y, solucao.clientes)
            for obj in distancias:
                if uso_bandwidth_pa <= (54 - obj['bandwidth']) and obj['distancia'] <= max_raio:
                    uso_bandwidth_pa += obj['bandwidth']
                    obj['pa_index'] = indice_pa
                    possiveis_atribuicoes.append({
                        'x_pa': pos_pa_x, 
                        'y_pa': pos_pa_y, 
                        'x_cliente': obj['x'], 
                        'y_cliente': obj['y'],
                        'pa_index': indice_pa,
                        'client_index': obj['client_index'],
                        'distancia_pa_cliente': obj['distancia']
                    })
                else:
                    break
            
            if uso_bandwidth_pa <= min_consumo_bandwidth:
                possiveis_atribuicoes = []
                uso_bandwidth_pa = 0
                pos_pa_x, pos_pa_y = atualizar_posicao_pa(pos_pa_x, pos_pa_y)
            else:
                for possivel_atribuicao in possiveis_atribuicoes:
                    for cliente in solucao.clientes:
                        if cliente['client_index'] == possivel_atribuicao['client_index']:
                            cliente['assigned'] = True
                            break

                solucao.atribuicoes.extend(possiveis_atribuicoes)
                solucao.pas.append((float(pos_pa_x), float(pos_pa_y)))
                pos_pa_x, pos_pa_y = atualizar_posicao_pa(pos_pa_x, pos_pa_y)

    fim_tempo = time.time()
    tempo_execucao = (fim_tempo - inicio_tempo) / 60
    print("\nTempo de execução para solução inicial: ", tempo_execucao)

    return solucao

# Função para desatribuir clientes
def desatribuir_clientes(clientes: List[dict]) -> List[dict]:
    for cliente in clientes:
        cliente['assigned'] = False
    return clientes

# Função para calcular distâncias entre PA e clientes
def calcular_distancias_pa_clientes(pa_x: float, pa_y: float, clientes: List):
    distancias = []
    clientes_nao_atribuidos = [cliente for cliente in clientes if not cliente['assigned']]
    for cliente in clientes_nao_atribuidos:
        distancias.append({
            'distancia': calcular_distancia(pa_x, pa_y, float(cliente['x']), float(cliente['y'])), 
            'x': float(cliente['x']), 'y': float(cliente['y']),
            'bandwidth': float(cliente['bandwidth']),
            'client_index': cliente['client_index']
        })
    distancias = sorted(distancias, key=lambda k: k['distancia'])
    return distancias

# Função para calcular a distância euclidiana
def calcular_distancia(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Função para atualizar a posição do PA
def atualizar_posicao_pa(x, y, esp_grid: int = 5) -> int:
    direcao = np.random.randint(0, 8)
    if direcao == 0:
        y += esp_grid
    elif direcao == 1:
        x += esp_grid
    elif direcao == 2:
        y -= esp_grid
    elif direcao == 3:
        x -= esp_grid
    elif direcao == 4:
        x -= esp_grid
        y += esp_grid
    elif direcao == 5:
        x += esp_grid
        y += esp_grid
    elif direcao == 6:
        x += esp_grid
        y -= esp_grid
    elif direcao == 7:
        x -= esp_grid
        y -= esp_grid

    x = max(0, min(x, 400))
    y = max(0, min(y, 400))
    return float(x), float(y)

# Função objetivo 1
def objetivo1(solucao) -> int:
    penalidade = calcular_penalidade_clientes_nao_servidos(solucao.atribuicoes, solucao.clientes)
    qtd_pas = len(solucao.pas)
    fitness = qtd_pas + penalidade
    return fitness

# Função objetivo 2
def objetivo2(solucao) -> float:
    penalidade = calcular_penalidade_clientes_nao_servidos(solucao.atribuicoes, solucao.clientes) * 50000
    distancia_total = calcular_distancia_total(solucao.atribuicoes) * 0.2
    fitness = distancia_total + penalidade 
    return fitness

# Função para calcular a penalidade por clientes não servidos
def calcular_penalidade_clientes_nao_servidos(atribuicoes: List, lista_clientes: List) -> int:
    penalidade = 0
    porc_clientes_nao_servidos = calcular_percentual_clientes_servidos(atribuicoes, lista_clientes)
    if porc_clientes_nao_servidos < 98 and porc_clientes_nao_servidos >= 96:
        penalidade += 2
    elif porc_clientes_nao_servidos < 96 and porc_clientes_nao_servidos >= 94:
        penalidade += 3
    elif porc_clientes_nao_servidos < 94 and porc_clientes_nao_servidos >= 91:
        penalidade += 5
    elif porc_clientes_nao_servidos < 91 and porc_clientes_nao_servidos >= 88:
        penalidade += 8
    elif porc_clientes_nao_servidos < 88 and porc_clientes_nao_servidos >= 85:
        penalidade += 13
    elif porc_clientes_nao_servidos < 85 and porc_clientes_nao_servidos >= 82:
        penalidade += 21
    elif porc_clientes_nao_servidos < 79 and porc_clientes_nao_servidos >= 75:
        penalidade += 34
    else:
        penalidade += 55
    return penalidade

# Função para calcular o percentual de clientes servidos
def calcular_percentual_clientes_servidos(atribuicoes: List, lista_clientes: List) -> float:
    return (len(atribuicoes) / len(lista_clientes)) * 100

# Função para calcular a distância total entre PAs e clientes atribuídos
def calcular_distancia_total(atribuicoes: List) -> int:
    return sum(atribuicao['distancia_pa_cliente'] for atribuicao in atribuicoes)

# Função para calcular a função objetivo com ponderação
def calcular_objetivo_ponderado(x, pesos):
    fit_obj1 = objetivo1(x)
    fit_obj2 = objetivo2(x)
    fit_obj1_norm = normalizar(fit_obj1, min_f1, max_f1)
    fit_obj2_norm = normalizar(fit_obj2, min_f2, max_f2)
    x.fit_obj1 = fit_obj1_norm
    x.fit_obj2 = fit_obj2_norm
    x.distancia_total = calcular_distancia_total(x.atribuicoes)
    x.fitness = pesos[0]*fit_obj1_norm + pesos[1]*fit_obj2_norm
    return x

# Função para implementar a soma ponderada
def funcao_soma_ponderada(x, info_abordagem):
    x = calcular_objetivo_ponderado(x, info_abordagem.pesos)
    x.valor_objetivo_unico = np.dot(info_abordagem.pesos, np.transpose(np.array(x.fitness)))
    return x

# Função para encontrar soluções não dominadas
def encontrar_solucoes_nao_dominadas(solucoes):
    nao_dominadas = []
    for i, sol_a in enumerate(solucoes):
        dominada = False
        for j, sol_b in enumerate(solucoes):
            if i != j and solucao_e_dominada(sol_a, sol_b):
                dominada = True
                break
        if not dominada:
            nao_dominadas.append(sol_a)
    return nao_dominadas

# Função para verificar se uma solução é dominada por outra
def solucao_e_dominada(sol_a, sol_b):
    return all(x <= y for x, y in zip(sol_a, sol_b)) and any(x < y for x, y in zip(sol_a, sol_b))

# Função para plotar soluções
def plotar_solucoes(todas_solucoes, solucoes_nao_dominadas):
    fit_obj1_todas = [sol[0] for sol in todas_solucoes]
    fit_obj2_todas = [sol[1] for sol in todas_solucoes]
    fit_obj1_nd = [sol[0] for sol in solucoes_nao_dominadas]
    fit_obj2_nd = [sol[1] for sol in solucoes_nao_dominadas]
    plt.figure(figsize=(10, 6))
    plt.scatter(fit_obj1_todas, fit_obj2_todas, c='blue', label='Todas as soluções', alpha=0.5)
    plt.scatter(fit_obj1_nd, fit_obj2_nd, c='red', label='Soluções não dominadas', edgecolors='black', s=100)
    plt.title('Soluções e Soluções Não Dominadas')
    plt.xlabel('Fitness f1')
    plt.ylabel('Fitness f2')
    plt.legend()
    plt.show()

# Função BVNS para otimização multiobjetivo
def bvns(funcao_objetivo, x, problema, info_abordagem, max_avaliacoes=1000):
    num_avaliacoes = 0
    max_avaliacoes = max_avaliacoes
    max_vizinhança = 4
    x = funcao_objetivo(x, info_abordagem)
    num_avaliacoes += 1
    while num_avaliacoes < max_avaliacoes:
        k = 1
        while k <= max_vizinhança:
            y = aplicar_shake(x, k, problema)
            y = funcao_objetivo(y, info_abordagem)
            z = aplicar_best_improvement(y, 4, problema)
            num_avaliacoes += 1
            x, k = aplicar_neighborhood_change(x, z, k)
    return x

# Função shake para gerar nova solução na vizinhança
def aplicar_shake(solucao, k, problema):
    np.random.seed() 
    nova_solucao = cp.deepcopy(solucao)
    if k == 1 and len(solucao.pas) < 30:
        cliente_nao_servido = obter_cliente_nao_servido_aleatorio(solucao.atribuicoes, problema.clientes)
        if cliente_nao_servido == 0:
            return nova_solucao
        
        pos_pa = obter_posicao_pa(cliente_nao_servido[0], cliente_nao_servido[1])
        nova_solucao.pas.append(pos_pa)
        nova_solucao.atribuicoes = rearranjar_clientes_pas(nova_solucao, problema.clientes)
    elif k == 2 and len(solucao.pas) > 1:
        indice_pa = obter_menos_utilizado_pa(solucao.atribuicoes, problema.clientes)
        nova_solucao.pas.pop(indice_pa)
        nova_solucao.atribuicoes = rearranjar_clientes_pas(nova_solucao, problema.clientes)
    elif k == 3:
        indice_pa = np.random.randint(0, len(nova_solucao.pas))
        pa_x, pa_y = nova_solucao.pas[indice_pa]
        novo_pa_x, novo_pa_y = atualizar_posicao_pa(pa_x, pa_y)
        nova_posicao_pa = list(nova_solucao.pas[indice_pa])
        nova_posicao_pa[0] = novo_pa_x
        nova_posicao_pa[1] = novo_pa_y
        nova_solucao.pas[indice_pa] = tuple(nova_posicao_pa)
        nova_solucao.atribuicoes = rearranjar_clientes_pas(nova_solucao, problema.clientes)
    elif k == 4:
        multiplos_de_5 = np.arange(0, 401, 5)
        indice_pa = np.random.randint(0, len(nova_solucao.pas))
        novo_pa_x = np.random.choice(multiplos_de_5)
        novo_pa_y = np.random.choice(multiplos_de_5)
        nova_posicao_pa = list(nova_solucao.pas[indice_pa])
        nova_posicao_pa[0] = novo_pa_x
        nova_posicao_pa[1] = novo_pa_y
        nova_solucao.pas[indice_pa] = tuple(nova_posicao_pa)
        nova_solucao.atribuicoes = rearranjar_clientes_pas(nova_solucao, problema.clientes)
    return nova_solucao

# Função best improvement para busca local
def aplicar_best_improvement(solucao_atual, max_vizinhança, problema):
    melhor_solucao = cp.deepcopy(solucao_atual)
    for i in range(1, max_vizinhança + 1):
        solucao_vizinha = aplicar_shake(melhor_solucao, i, problema)
        if solucao_vizinha.fitness < melhor_solucao.fitness:
            melhor_solucao = solucao_vizinha
    return melhor_solucao

# Função para aplicar a mudança de vizinhança
def aplicar_neighborhood_change(solucao_atual, solucao_nova, k):
    if solucao_nova.fitness < solucao_atual.fitness:
        solucao_atual = cp.deepcopy(solucao_nova)
        k = 1
    else:
        k += 1
    return solucao_atual, k

# Função para obter um cliente não servido aleatório
def obter_cliente_nao_servido_aleatorio(atribuicoes: List[Dict], clientes: List[Dict]) -> Tuple[float, float]:
    clientes_servidos = {cliente['client_index'] for cliente in atribuicoes}
    if len(clientes_servidos) == 495:
        return 0
    
    clientes_nao_servidos = [(float(cliente['x']), float(cliente['y'])) for cliente in clientes if cliente['client_index'] not in clientes_servidos]
    indice_cliente = np.random.randint(0, len(clientes_nao_servidos))
    return clientes_nao_servidos[indice_cliente]

# Função para obter o PA menos utilizado
def obter_menos_utilizado_pa(atribuicoes: List[Dict], clientes: List[Dict]) -> int:
    pas = [{'pa_index': obj['pa_index'], 'bd_consumo': clientes[obj['client_index']]['bandwidth']} for obj in atribuicoes]
    consumo_bd_por_pa = defaultdict(float)
    for item in pas:
        consumo_bd_por_pa[item['pa_index']] += item['bd_consumo']
    consumo_bd_por_pa = dict(consumo_bd_por_pa)
    consumo_ordenado_bd = sorted(consumo_bd_por_pa.items(), key=lambda x: x[1], reverse=False)
    return consumo_ordenado_bd[0][0]

# Função para obter a posição do PA baseado na posição do cliente
def obter_posicao_pa(x_cliente: float, y_cliente: float) -> Tuple[int, int]:
    return int(round(x_cliente / 5) * 5), int(round(y_cliente / 5) * 5)

# Função para rearranjar clientes e PAs
def rearranjar_clientes_pas(solucao: Solucao, lista_clientes: List) -> List[dict]:
    solucao.atribuicoes = []
    desatribuir_clientes(lista_clientes)
    
    for indice, valor in enumerate(solucao.pas):
        uso_bandwidth_pa = 0
        distancias = calcular_distancias_pa_clientes(valor[0], valor[1], lista_clientes)
        for obj in distancias:
            if uso_bandwidth_pa <= (54 - obj['bandwidth']) and obj['distancia'] <= 85:
                solucao.atribuicoes.append({
                    'x_pa': float(valor[0]), 
                    'y_pa': float(valor[1]), 
                    'x_cliente': float(obj['x']), 
                    'y_cliente': float(obj['y']),
                    'pa_index': indice,
                    'client_index': obj['client_index'],
                    'distancia_pa_cliente': obj['distancia']
                })
                lista_clientes[obj['client_index']]['assigned'] = True
    
    return solucao.atribuicoes

# Função para plotar a solução
def plotar_solucao(solucao: Solucao, titulo: str = 'Clientes e Pontos de Acesso (PAs)', salvar_plot: bool = False, nome_arquivo: str = "plot") -> None:
    plt.figure(figsize=(10, 10))
    for cliente in clients_df.itertuples():
        plt.scatter(cliente.x, cliente.y, c='blue', label='Cliente' if cliente.Index == 0 else "")
    for pa in solucao.pas:
        plt.scatter(float(pa[0]), float(pa[1]), c='red', marker='X', label='PA' if solucao.pas.index(pa) == 0 else "")
    for atribuicao in solucao.atribuicoes:
        plt.plot([atribuicao['x_pa'], atribuicao['x_cliente']], [atribuicao['y_pa'], atribuicao['y_cliente']], 'k-', alpha=0.2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    if salvar_plot:
        plt.savefig(nome_arquivo)
    plt.show()

# Carregar dados dos clientes
clients_df = carregar_dados_clientes('clientes.csv')

# Definir o problema
probdata = definir_problema(clients_df)

# Armazenar dados para plotagem
arquivo = Estrutura()
arquivo.sol = []
arquivo.fitpen = []

# Armazenar dados da estratégia de otimização mono-objetivo
info_abordagem = Estrutura()

# Parâmetros da metaheurística BVNS
N = 15
for i in np.arange(0, N, 1):
    pesos = np.random.random(size=2)
    pesos = pesos / sum(pesos)
    x = inicializar_solucao(probdata, pesos)
    info_abordagem.pesos = pesos
    x = bvns(funcao_soma_ponderada, x, probdata, info_abordagem, max_avaliacoes=200)
    arquivo.fitpen.append((x.fit_obj1, x.fit_obj2))

# Encontrar soluções não dominadas
solucoes_nao_dominadas = encontrar_solucoes_nao_dominadas(arquivo.fitpen)

# Plotar soluções
plotar_solucoes(arquivo.fitpen, solucoes_nao_dominadas)

print("Soluções não dominadas:", solucoes_nao_dominadas)
print("Todas as soluções: ", arquivo.fitpen)
