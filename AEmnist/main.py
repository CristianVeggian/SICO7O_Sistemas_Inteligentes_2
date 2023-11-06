# import de todos os módulos necessários para o experimento 

# manipulação de números de ponto flutuante
import numpy as np

# manipulação/definição de DL e NNs
import keras

# carregando dataset MNIST (via keras)
from keras.datasets import mnist, fashion_mnist
from keras.utils import set_random_seed

# plots via matplotlib
import matplotlib.pyplot as plt

# definir um seed para reprodução dos experimentos
set_random_seed(42)

# lendo os conjuntos de treinamento (x_train) e teste (x_test)
# nesse exemplo iremos descartar os labels, pois estamos interessados apenas
# nos processos de codificação/decodificação das imagens
#(x_train, _), (x_test, _) = mnist.load_data()
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Vamos normalizar todos os valores (pixels) entre 0 e 1
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

# aqui achatamos o sinal, cada imagem 28 x 28 vira um vetor de 784 valores
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# checando a dimensao dos dados
print('* Dimensões do conj treinamento: ', x_train.shape)
print('* Dimensões do conj teste:       ',x_test.shape)

# Definir o tamanho das nossas representações codificadas
# (codings, ou representações latentes)
# 32 features extraídas de 784 valores
encoding_dim = 10

# numero de epocas para os experimentos
exp_epochs = 15

# definir um encoder, que recebe como entrada o sinal já "achatado", 
# e possui apenas uma camada densa com 32 neurônios (codings),
# A função de ativação de cada neurônio é do tipo ReLU
encoder = keras.models.Sequential(name = "encoder",
    layers = [keras.layers.GaussianNoise(0.5, input_shape=[784]),
        keras.layers.Dense(encoding_dim, activation='relu', input_shape = [784]),
])

# imprimindo o modelo criado
encoder.summary()

# definir um decoder, que recebe como entrada um vetor com 32 valores 
# numéricos (codings), e recria as inputs (saida = 784 valores)
# a função de ativação dos neurônios é sigmoidal
decoder = keras.models.Sequential(name = "decoder", 
    layers = [keras.layers.Dense(784, activation= 'sigmoid', input_shape = [encoding_dim])
])

# imprimindo o modelo criado
decoder.summary()

# Agora podemos criar o autoencoder combinando enconder + decoder
autoencoder = keras.models.Sequential(
    name = "autoencoder", 
    layers = [encoder, decoder]
)

# imprimindo o modelo
autoencoder.summary()

# Vamos configurar as opções de treinamento do nosso modelo (autoencoder)
# usaremos o algoritmo Adam, e como função de custo (loss function)
# a entropia binária - medida recomendada para classificação binária
# (é a mesma que temos descrita nos slides de aula)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Configurado o aprendizado, precisamos treinar o modelo
# vamos armazenar todas as informações de treinamento na variável 
# 'autoencoder_history'
autoencoder_history = autoencoder.fit(
    # x = dados de treinamento (input data)
    x = x_train,
    # y = target / labels, aqui será x_train também, pois queremos reconstruir a entrada
    y = x_train,
    # épocas de treinamento
    epochs = exp_epochs,
    # número de amostras computadas para ter uma atualização do gradiente
    batch_size = 256,
    # embaralhar os exemplos de treinamento antes de cada época
    shuffle = True,
    # usamos o conjunto de teste como validação interna do treinamento
    validation_data =(x_test, x_test)
)

# realizar a codificação do conjunto de teste
encoded_imgs = encoder.predict(x_test)

# reconstruir as entradas por meio dos codings
decoded_imgs = decoder.predict(encoded_imgs)

# Verificar as reconstruções (outputs)
# Vamos visualizar as reconstruções com base nas imagens originais
# linha 1 - imagens originais
# linha 2 - imagens reconstruidas pelo AE simples

n = 30  #numero de imagens que queremos ver

plt.figure(figsize=(20, 4))
for i in range(n):
    
    # imagens originais
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # imagens reconstruidas
    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()