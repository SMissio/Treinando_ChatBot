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
