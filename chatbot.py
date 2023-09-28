import random
import nltk # Importa a biblioteca nltk para processamento de linguagem natural
import spacy # Importa a biblioteca spaCy para processamento de linguagem natural
import numpy as np # Importa a biblioteca numpy para computação numérica
from goose3 import Goose # Importa a classe Goose da biblioteca goose3 para extrair informações de páginas da web
from sklearn.metrics.pairwise import cosine_similarity # Importa a função cosine_similarity de sklearn.metrics.pairwise para calcular a similaridade de cosseno
from sklearn.feature_extraction.text import TfidfVectorizer # Importa a classe TfidfVectorizer de sklearn.feature_extraction.text para criar vetores TF-IDF

nltk.download('punkt') # Baixa os dados necessários para a tokenização de texto do nltk
nlp = spacy.load('en_core_web_sm') # Carrega o modelo de processamento de linguagem natural do spaCy para inglês

article = Goose().extract('https://en.wikipedia.org/wiki/Bitcoin') # Extrai informações da página da web no URL fornecido usando a classe Goose
article_sentences = nltk.sent_tokenize(article.cleaned_text) # Divide o texto extraído em sentenças usando a função sent_tokenize do nltk

welcome_words_input = ('hey', 'hello', 'hi') # Lista de palavras esperadas para boas vindas

welcome_words_output = ('hey', 'hello', 'how are you?', 'welcome', 'how are you doing?') # Palavras que responderão as esperadas

exit_inputs = ('quit', 'close', 'exit', 'q') # Palavras utilizadas para finalizar a execução

def welcome_message(text):
  words = text.split() # quebra o texto gerando uma lista com as palavras contidas nele

  for word in words: # validar se uma palavra se encontra na lista de palavras aguardadas
    if word.lower() in welcome_words_input: # converte palavras para minusculo
      return random.choice(welcome_words_output)

def preprocessing(sentence): # Função de pré processamento das sentenças
  sentence_lower = sentence.lower()

  tokens = [token.text for token in nlp(sentence_lower) if not (token.is_stop
                                                          or token.like_num
                                                          or token.is_punct
                                                          or token.is_space
                                                          or len(token) == 1)]
  return ' '.join(tokens)

def answer(user_text, threshold = 0.25): # Função para buscar resposta de acordo com texto inserido pelo usuário
    preprocessed_sentences = [preprocessing(sentence) for sentence in article_sentences]
    preprocessed_user_text = preprocessing(user_text)

    preprocessed_sentences.append(preprocessed_user_text) # Acrescenta o texto do usuário que foi processado no final das sentenças

    vectorizer = TfidfVectorizer() # Nova instância para o vetorizador TF-IDF

    vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences) # Vetoriza e transforma as sentenças preprocessadas

    similarity = cosine_similarity(vectorized_sentences[-1], vectorized_sentences) # Calcula a similaridade de conseno entre a ultima posição (pergunta usuário) e as demais sentenças
    similarity_index = similarity.argsort()[0][-2] # Obtém o indice da penúltima posição (maior corresopndencia)

    similarity_score = similarity[0][similarity_index] # Obtém o valor de similaridade

    if similarity_score < threshold:
      return 'sorry, no answear was found'

    return article_sentences[similarity_index] # Acessa a lista de sentenças originais

while True:
  print('\nEnter an sentence!')
  user_text = input()

  if user_text.lower() in exit_inputs:
    print('Chatbot: Bye! See you soon...')
    break

  if (msg := welcome_message(user_text)) != None:
    print(f'Chatbot: {msg}')
  else:
    print('In processing, please wait for a moment.')
    print(f'Chatbot: {answer(user_text)}')
