import os
import json
import time
import random
import threading
from datetime import datetime
from flask import Flask, request, send_from_directory, jsonify
import psutil
import numpy as np
import matplotlib.pyplot as plt

##__________________________________________________ Variáveis Globais

MAZE_PATH = '../Server/Matrizes/'
SERVER_STATUS_PATH = '../Server/relatorios/statusServidor/statusServidor.csv'
PROCESS_FOLDER = '../Server/relatorios/processos'
ALGORITMO_CGNR = "CGNR"
ALGORITMO_CGNE = "CGNE"
FILA_PATH = "../Server/fila.txt"

##__________________________________________________ Variáveis especificadas no problema

erro = 1e-4
LENGTH_SINAL_30X30 = 27904
LENGTH_SINAL_60X60 = 50816
MAX_THREADS = 2
THREADS_ATUAIS = 0

##__________________________________________________ Inicializações

app = Flask(__name__)
semaphore = threading.Semaphore()


##__________________________________________________ Funções matriz

def readCSV(path):
    with open(path, 'r') as file:
        fileLines = file.readlines()
        data = []
        for line in fileLines:
            data.append([float(x) for x in line.strip().split(',')])
        return np.array(data)


Maze_60x60 = readCSV(f"{MAZE_PATH}/H-1.csv")
Maze_30x30 = readCSV(f"{MAZE_PATH}/H-2.csv")


def retornar_matriz_H(tamanho):
    if tamanho == 60:
        return Maze_60x60
    elif tamanho == 30:
        return Maze_30x30


##_____________________________________________________ Gerar id

def gerar_id_aleatorio():
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz1234567890') for i in range(12))


print(gerar_id_aleatorio())


##_____________________________________________________ Funções das rotas

def criar_processo_em_json(id_processo, tipo_algoritmo, nome_usuario):
    data = {
        "idProcesso": id_processo,
        "tipoAlgoritmo": tipo_algoritmo,
        "nomeUsuario": nome_usuario,
        "sinal": [],
        "dataCriacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataInicioProcessamento": None,
        "dataFimProcessamento": None
    }
    return data


def atualizar_processo_em_json(id_processo, sinal, isLast):
    process_folder_path = f"{PROCESS_FOLDER}/{id_processo}"
    with open(f"{process_folder_path}/processo.json", 'r+') as f:
        data = json.load(f)
        data["sinal"].extend(sinal)
        if isLast:
            with open(f"{FILA_PATH}", "a") as file:
                file.write(id_processo + "\n")
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


##_____________________________________________________ Rotas

@app.route('/processo', methods=['POST'])
def post_processo():
    tipo_algoritmo = request.form.get('tipoAlgoritmo')
    nome_usuario = request.form.get('nomeUsuario')
    id_processo = gerar_id_aleatorio()
    process_folder_path = f"{PROCESS_FOLDER}/{id_processo}"

    os.makedirs(process_folder_path, exist_ok=True)
    data = criar_processo_em_json(id_processo, tipo_algoritmo, nome_usuario)
    process_file_path = f"{process_folder_path}/processo.json"
    with open(process_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    return jsonify({"idProcesso": id_processo}), 201


@app.route('/processo/<id_processo>', methods=['PATCH'])
def patch_processo(id_processo):
    sinal = request.json.get('sinal')
    isLast = request.json.get('isLast', False)
    atualizar_processo_em_json(id_processo, sinal, isLast)
    return jsonify({"message": "Sinal atualizado com sucesso"}), 200


@app.route('/processo/<id_processo>/imagem', methods=['GET'])
def get_imagem(id_processo):
    print("OLA")
    process_folder_path = f"{PROCESS_FOLDER}/{id_processo}"
    with open(f"{process_folder_path}/processo.json", 'r') as f:
        data = json.load(f)
        if data["dataFimProcessamento"]:
            return send_from_directory(f"{process_folder_path}", "imagem.png")
        else:
            return jsonify({"message": "O processo ainda está em andamento"}), 202


##__________________________________________________________ Algoritmos de gradiente

def cgnr(h, g, tam_image):
    f = np.zeros((tam_image ** 2))
    r = g - np.dot(h, f)
    z = np.dot(np.transpose(h), r)
    p = z
    i = 0
    while True:
        w = np.dot(h, p)
        a = np.linalg.norm(z, ord=2) ** 2 / np.linalg.norm(w, ord=2) ** 2
        f = f + np.dot(a, p)
        r_ant = r
        r = r - np.dot(a, w)
        z_ant = z
        z = np.dot(np.transpose(h), r)
        beta = np.linalg.norm(z, ord=2) ** 2 / np.linalg.norm(z_ant, ord=2) ** 2
        p = z + np.dot(beta, p)
        if (calcular_erro(r, r_ant) < erro):
            break
        i += 1
    return f.reshape(tam_image, tam_image), i


def cgne(h, g, tam_image):
    f = np.zeros((tam_image ** 2))
    r = g - np.dot(h, f)
    p = np.dot(np.transpose(h), r)
    i = 0
    while True:
        r_ant = r
        a = np.dot(np.transpose(r), r) / np.dot(np.transpose(p), p)
        f = f + a * p
        r = r - np.dot(a, np.dot(h, p))
        Beta = np.dot(np.transpose(r), r) / np.dot(np.transpose(r_ant), r_ant)
        p = np.dot(np.transpose(h), r) + np.dot(Beta, p)
        if (calcular_erro(r, r_ant) < erro):
            break
        i += 1
    return f.reshape(tam_image, tam_image), i


def calcular_erro(r, r_ant):
    return abs(np.linalg.norm(r, ord=2) - np.linalg.norm(r_ant, ord=2))


##__________________________________________________________________________ Monitorar servidor

def monitorar_servidor():
    with open(SERVER_STATUS_PATH, 'w') as f:
        f.seek(0)
        f.write("created_at;cpu_porcentagem;ram_porcentage;ram_gb\n")
        f.truncate()

    while True:
        with open(SERVER_STATUS_PATH, 'a') as f:
            cpu = psutil.cpu_percent()
            ram_porcentage = psutil.virtual_memory()[2]
            ram_gb = psutil.virtual_memory()[3] / 1000000000
            createdAt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{createdAt};{cpu:.2f};{ram_porcentage:.2f};{ram_gb:.2f}\n")

            time.sleep(1)


##__________________________________________________________ Funções para o monitoramento da fila e processamento

def incrementa_threads():
    semaphore.acquire()
    global THREADS_ATUAIS
    THREADS_ATUAIS += 1
    semaphore.release()


def decrementa_threads_atuais():
    semaphore.acquire()
    global THREADS_ATUAIS
    THREADS_ATUAIS -= 1
    semaphore.release()


def salvar_imagem(path, image, metadata):
    plt.imsave(path, image, cmap='gray', metadata=metadata)


def processar_imagem(id_processo):
    incrementa_threads()

    process_folder_path = f"{PROCESS_FOLDER}/{id_processo}"

    tipoAlgoritmo = None
    usuario = None
    sinal = None
    dataInicioProcessamento = None
    dataFimProcessamento = None

    with open(f"{process_folder_path}/processo.json", 'r+') as f:
        data = json.load(f)
        data["dataInicioProcessamento"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        tipoAlgoritmo = data["tipoAlgoritmo"]
        sinal = np.array(data["sinal"])
        usuario = data["nomeUsuario"]
        dataInicioProcessamento = data["dataInicioProcessamento"]

    tamanhoImagem = None
    if sinal.shape[0] == LENGTH_SINAL_30X30:
        tamanhoImagem = 30
    elif sinal.shape[0] == LENGTH_SINAL_60X60:
        tamanhoImagem = 60

    matriz_h = retornar_matriz_H(tamanhoImagem)

    imagem = None
    iteracoes = None

    if tipoAlgoritmo == ALGORITMO_CGNE:
        imagem, iteracoes = cgne(matriz_h, sinal, tamanhoImagem)
    elif tipoAlgoritmo == ALGORITMO_CGNR:
        imagem, iteracoes = cgnr(matriz_h, sinal, tamanhoImagem)

    with open(f"teste.txt", "w") as f:
        f.write(str(imagem))

    dataFimProcessamento = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{process_folder_path}/processo.json", 'r+') as f:
        data = json.load(f)
        data["dataFimProcessamento"] = dataFimProcessamento
        data["iteracoes"] = iteracoes
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    salvar_imagem(f"{process_folder_path}/imagem.png", imagem, {
        'Algoritmo': tipoAlgoritmo,
        'Usuario': usuario,
        'Iteracoes': str(iteracoes),
        'dataInicioProcessamento': dataInicioProcessamento,
        'dataFimProcessamento': dataFimProcessamento
    })

    decrementa_threads_atuais()


##______________________________ Monitorar fila

def monitorar_fila():
    while True:
        print("Threads atuais: ", THREADS_ATUAIS)
        if THREADS_ATUAIS < MAX_THREADS:
            with open(f"{FILA_PATH}", "r+") as f:
                lines = f.readlines()

                if (len(lines) > 0):
                    idProcesso = lines[0].strip()
                    print(f"VAI PROCESSAR -{idProcesso}-")
                    thread = threading.Thread(target=processar_imagem, args=(idProcesso,))

                    thread.start()
                    del lines[0]

                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
        else:
            print("Maximo de Threads!")
        time.sleep(1)


##________________________________________________ Run app
threading.Thread(target=monitorar_servidor).start()
threading.Thread(target=monitorar_fila).start()
app.run(debug=False)
