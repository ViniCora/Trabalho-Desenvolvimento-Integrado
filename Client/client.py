import os
import requests
import time
import random
import numpy as np

PASTA_SINAIS = "sinais"
PASTA_PROCESSOS = "processos"
SERVER_URL = "http://localhost:5000"
TIPOS_ALGORITMOS = ["CGNR", "CGNE"]
TIPOS_SINAIS = ["Sinal-A-30x30-1.csv", "Sinal-G-30x30-1.csv", "Sinal-G-30x30-2.csv", "Sinal-A-60x60-1.csv",
                "Sinal-G-60x60-1.csv", "Sinal-G-60x60-2.csv"]

##_______________________________________________________ Funções CSV

def lerArquivoCSV(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            data.append([float(x) for x in line.strip().split(',')])
        return np.array(data)

##_______________________________________________________ Funções Imagem

def capturar_imagem(id_processo):
    """Captura a imagem do processo."""
    url = f"{SERVER_URL}/processo/{id_processo}/imagem"
    response = requests.get(url)
    if response.status_code == 200:
        # Salva a imagem na pasta do processo
        image_path = f"{PASTA_PROCESSOS}/{id_processo}/imagem.png"
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"Imagem capturada e salva em {image_path}")
    elif response.status_code == 202:
        print("O processo ainda está em andamento.")
    else:
        print(f"Erro ao capturar imagem: {response.status_code}")

##_______________________________________________________ Funções Sinais

def calcular_sinal_ganho(g):
    n = 64
    s = 794 if len(g) > 50000 else 436
    for c in range(n):
        for l in range(s):
            y = 100 + (1 / 20) * l * np.sqrt(l)
            g[l + c * s] = g[l + c * s] * y

    return g

def ler_sinal(tipo_sinal):
    """Lê o sinal de um arquivo e retorna um array de números."""
    sinal = lerArquivoCSV(f"{PASTA_SINAIS}/{tipo_sinal}")[:, 0]
    return calcular_sinal_ganho(sinal)


def enviar_sinal(id_processo, sinal: np.ndarray, isLast=False):
    """Envia um sinal para o servidor."""
    url = f"{SERVER_URL}/processo/{id_processo}"
    data = {"sinal": sinal.tolist(), "isLast": isLast}
    response = requests.patch(url, json=data)
    if response.status_code == 200:
        print("Sinal enviado com sucesso.")
    else:
        print(f"Erro ao enviar sinal: {response.status_code}")

##_______________________________________________________ Função principal do cliente, onde é feita a escolha do processo
def requisicao_processo_sinal_imagem():
    nome_cliente = input("Digite o nome do cliente: ")
    while True:
        print("\nEscolha uma opção:")
        print("1: Enviar sinal")
        print("2: Capturar imagem")
        opcao = input("Opção: ")

        if opcao == "1":
            # Cria um processo
            tipo_algoritmo = random.choice(TIPOS_ALGORITMOS)
            url = f"{SERVER_URL}/processo"
            data = {"tipoAlgoritmo": tipo_algoritmo, "nomeUsuario": nome_cliente}
            response = requests.post(url, data=data)
            if response.status_code == 201:
                id_processo = response.json()["idProcesso"]
                print(f"Processo criado com ID: {id_processo}")
                os.makedirs(f"{PASTA_PROCESSOS}/{id_processo}", exist_ok=True)

                # Lê o sinal
                tipo_sinal = random.choice(TIPOS_SINAIS)
                sinal = ler_sinal(tipo_sinal)

                porcentagem = random.randint(10, 20)
                tamanho_chunk = int(len(sinal) * porcentagem / 100)
                pedacos_sinal = [sinal[i:i + tamanho_chunk] for i in range(0, len(sinal), tamanho_chunk)]

                # Envia os pedaços de sinal
                for i, pedaco in enumerate(pedacos_sinal):
                    print(f"Enviando pedaço {i + 1}/{len(pedacos_sinal)}")
                    enviar_sinal(id_processo, pedaco, isLast=(i == len(pedacos_sinal) - 1))
                    time.sleep(random.randint(1, 2))
            else:
                print(f"Erro ao criar processo: {response.status_code}")

        elif opcao == "2":
            # Mostra os processos sem imagem
            processos_sem_imagem = []
            for processo in os.listdir("processos"):
                imagem_path = f"{PASTA_PROCESSOS}/{processo}/imagem.png"
                if not os.path.exists(imagem_path):
                    processos_sem_imagem.append(processo)

            if processos_sem_imagem:
                print("\nProcessos sem imagem:")
                for i, processo in enumerate(processos_sem_imagem):
                    print(f"{i + 1}: {processo}")
                escolha = input("Escolha um processo: ")
                if escolha.isdigit() and 1 <= int(escolha) <= len(processos_sem_imagem):
                    id_processo = processos_sem_imagem[int(escolha) - 1]
                    capturar_imagem(id_processo)
                else:
                    print("Opção inválida.")
            else:
                print("Não há processos sem imagem.")

        else:
            print("Opção inválida.")

if __name__ == "__main__":
    requisicao_processo_sinal_imagem()