# criacao do chatbot

#importando bibliotecas

import numpy as np
import tensorflow as tf
import re
import time

linhas = open('movie_lines.txt',encoding = 'utf-8',errors = 'ignore').read().split('\n')
conversas = open('movie_conversations.txt',encoding = 'utf-8',errors = 'ignore').read().split('\n')

id_linha = {}

for linha in linhas:
    _linha = linha.split(' +++$+++ ')
    #print(_linha)
    if len (_linha)==5:
        #print(_linha[4])
        id_linha[_linha[0]] = _linha[4]

conversas_id = []
for conversa in conversas[:-1]:
    _conversa = conversa.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    #print(_conversa)
    conversas_id.append(_conversa.split(','))


perguntas = []
respostas = []

for conversa in conversas_id:
    #print(conversa)
    #print(********)
    for i in range (len(conversa) - 1):
        #print(i)
        perguntas.append(id_linha[conversa[i]])
        respostas.append(id_linha[conversa[i+1]])

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"i'm","i am",texto)
    texto = re.sub(r"he's", "he is", texto)
    texto = re.sub(r"she's", "she is", texto)
    texto = re.sub(r"that's", "that is", texto)
    texto = re.sub(r"what's", "what is", texto)
    texto = re.sub(r"where's", "Where is", texto)
    texto = re.sub(r"\'ll", " will", texto)
    texto = re.sub(r"\'ve", " have", texto)
    texto = re.sub(r"\'re", " are", texto)
    texto = re.sub(r"\'d", " would", texto)
    texto = re.sub(r"won't", "will not", texto)
    texto = re.sub(r"can't", "cannot", texto)
    texto = re.sub(r"[-()#/@;:<>{}~+=?.|,]", "", texto)
    
    return texto

limpar_texto("ExeMplo i'm #@")

perguntas_limpas = []
for pergunta in perguntas:
    perguntas_limpas.append(limpar_texto(pergunta))

respostas_limpas = []
for resposta in respostas:
    respostas_limpas.append(limpar_texto(resposta))

palavras_contagem = {}
for pergunta in perguntas_limpas:
    for palavra in pergunta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] +=1

for resposta in respostas_limpas:
    for palavra in resposta.split():
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] +=1

#Palavras nao frequentes
            
limite = 20
perguntas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        perguntas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1

respostas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        respostas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    perguntas_palavras_int[token] = len(perguntas_palavras_int) + 1
for token in tokens:
    respostas_palavras_int[token] = len(respostas_palavras_int) + 1
    
#invervsao da chave do dicionario
# p_i = palavras (variavel) utilizando for na sequencia
respostas_int_palavras = {p_i: p for p, p_i in respostas_palavras_int.items()}
#token
for i in range(len(respostas_limpas)):
    respostas_limpas[i] += ' <EOS>'
#----trad inteiros e palavras menos frequentes
perguntas_para_int = []
for pergunta in perguntas_limpas:
    ints = []
    for palavra in pergunta.split():
        if palavra not in perguntas_palavras_int:
            ints.append(perguntas_palavras_int['<OUT>'])
        else:
            ints.append(perguntas_palavras_int[palavra])
    perguntas_para_int.append(ints)

respostas_para_int = []
for resposta in respostas_limpas:
    ints = []
    for palavra in resposta.split():
        if palavra not in respostas_palavras_int:
            ints.append(respostas_palavras_int['<OUT>'])
        else:
            ints.append(respostas_palavras_int[palavra])
    respostas_para_int.append(ints)

#ordem
perguntas_limpas_ordenadas = []
respostas_limpas_ordenadas = []
for tamanho in range (1, 25 + 1):
    for i in enumerate(perguntas_para_int):
        if len(i[1]) == tamanho:
            perguntas_limpas_ordenadas.append(perguntas_para_int[i[0]])
            respostas_limpas_ordenadas.append(respostas_para_int[i[0]])
#Seq2Seq construção
def entradas_modelo():
    entradas = tf.placeholder(tf.int32,[None,None], name = 'entradas')
    saidas = tf.placeholder(tf.int32,[None,None], name = 'saidas')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return entradas, saidas, lr, keep_prob
