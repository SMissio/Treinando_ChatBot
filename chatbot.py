# criacao do chatot

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