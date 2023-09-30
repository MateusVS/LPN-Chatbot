import random
import threading
import nltk # Importa a biblioteca nltk para processamento de linguagem natural
import spacy # Importa a biblioteca spaCy para processamento de linguagem natural
import numpy as np # Importa a biblioteca numpy para computação numérica
from goose3 import Goose # Importa a classe Goose da biblioteca goose3 para extrair informações de páginas da web
from sklearn.metrics.pairwise import cosine_similarity # Importa a função cosine_similarity de sklearn.metrics.pairwise para calcular a similaridade de cosseno
from sklearn.feature_extraction.text import TfidfVectorizer # Importa a classe TfidfVectorizer de sklearn.feature_extraction.text para criar vetores TF-IDF
import tkinter as tk # Biblioteca tkinter para interfaces gráficas
from tkinter import scrolledtext

nltk.download('punkt') # Baixa os dados necessários para a tokenização de texto do nltk
nlp = spacy.load('en_core_web_sm') # Carrega o modelo de processamento de linguagem natural do spaCy para inglês

article = Goose().extract('https://en.wikipedia.org/wiki/Bitcoin') # Extrai informações da página da web no URL fornecido usando a classe Goose
article_sentences = nltk.sent_tokenize(article.cleaned_text) # Divide o texto extraído em sentenças usando a função sent_tokenize do nltk

welcome_words_input = ('hey', 'hello', 'hi') # Lista de palavras esperadas para boas vindas
welcome_words_output = ('hey', 'hello', 'how are you?', 'welcome', 'how are you doing?') # Palavras que responderão as esperadas
exit_inputs = ('quit', 'close', 'exit', 'q') # Palavras utilizadas para finalizar a execução

def on_send_click():
    display_text.config(state='normal') # Habilita a área de exibição para edição
    answer_input_text = answer_input.get() # Obtém o texto inserido pelo usuário
    display_text.insert('end', f'You: {answer_input_text}\n') # Adiciona o texto à área de exibição
    print(f'You: {answer_input_text}') # Imprime o texto do usuário

    if answer_input_text.lower() in exit_inputs:
        print('Chatbot: Bye! See you soon...')
        display_text.insert('end', 'Chatbot: Bye! See you soon...\n') # Adiciona o texto à área de exibição
        window.quit()
    if (msg := welcome_message(answer_input_text)) != None:
        print(f'Chatbot: {msg}')
        display_text.insert('end', f'Chatbot: {msg}\n') # Adiciona o texto à área de exibição
    else:
        print('In processing, please wait for a moment.')
        print(f'Chatbot: {answer(answer_input_text)}')
        answer_thread = threading.Thread(target=process_answer, args=(answer_input_text,))
        answer_thread.start()

    display_text.config(state='disabled') # Desabilita a área de exibição novamente
    answer_input.delete(0, 'end') # Limpa o campo de entrada

def process_answer(user_text): # Adiciona o texto à área de exibição em uma thread separada
    response = answer(user_text)
    window.after(100, update_display, response)  # Agenda a atualização da interface gráfica

def update_display(response):
    display_text.config(state='normal')
    display_text.insert('end', f'Chatbot: {response}\n')
    display_text.config(state='disabled')
    
def on_enter(event):
   on_send_click()

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

def fill_input_with_hello():
    answer_input.delete(0, 'end')
    answer_input.insert(0, 'Hello')

def fill_input_with_last_bitcoin():
    answer_input.delete(0, 'end')
    answer_input.insert(0, 'When will the last bitcoin be generated?')

def fill_input_with_exit():
    answer_input.delete(0, 'end')
    answer_input.insert(0, 'exit')

window = tk.Tk() # Cria uma nova janela na interface gráfica
window.title('Bitcoin Chatbot')

hello_button = tk.Button(window, text='Hello', command=fill_input_with_hello)
hello_button.pack(side='left', padx=5)

last_bitcoin_button = tk.Button(window, text='When will the last bitcoin be generated?', command=fill_input_with_last_bitcoin)
last_bitcoin_button.pack(side='left', padx=5)

exit_button = tk.Button(window, text='Exit', command=fill_input_with_exit)
exit_button.pack(side='left', padx=5)

answer_input = tk.Entry(window, width=50) # Cria um campo de entrada de texto
answer_input.pack()
answer_input.bind('<Return>', on_enter) # Associa a função on_enter ao evento "Return" (tecla Enter)

send_button = tk.Button(window, text='Enter an sentence!', command=on_send_click) # Cria um botão para enviar a pergunta
send_button.pack()

display_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20, state='disabled') # Cria uma área de exibição de texto para os prints
display_text.pack()

window.mainloop()
