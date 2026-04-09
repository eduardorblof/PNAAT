from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dividindo o dataset de treino em treino e validação de forma balanceada
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25)
 
# Checando quantidade de imagens do dataset
print('Quantidade de imagens de treino:', x_train.shape[0])
print('Quantidade de imagens de validação:', x_val.shape[0])
print('Quantidade de imagens de test:', x_test.shape[0])
 
# Contando quantidade de imagens por dígito
import collections
counterTrain=collections.Counter(y_train)
counterVal=collections.Counter(y_val)
counterTest=collections.Counter(y_test)
 
# Plotando quantidade de imagens de cada dígito
from matplotlib import pyplot
 
fig, ax = pyplot.subplots()
rects1 = ax.bar(counterTrain.keys(), counterTrain.values(), label='Treino')
rects2 = ax.bar(counterVal.keys(), counterVal.values(), label='Validação')
rects3 = ax.bar(counterTest.keys(), counterTest.values(), label='Teste')
 
ax.set_title('Imagens por dígito')
ax.set_ylabel('Quantidade de imagens')
ax.set_xlabel('Dígito')
ax.legend()
pyplot.show()

# Formatando o dataset para funcionar como entrada do Keras 
# As imagens de entradas precisam estar em um array de 4 dimensões
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cada imagem precisa ter dimensão x, y e z

input_shape = (28, 28, 1)

# Convertento valores dos pixels para float (garantindo precisão em operações de divisão por exemplo)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
 
# Normalizando os valores dos pixels (valores entre 0 e 1).
x_train /= 255
x_val /= 255
x_test /= 255

# Importando Keras e suas operações
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Inicializando a CNN
model = Sequential()

# Operação de convolução com filtro 3 x 3 seguida da função de ativação ReLU
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

# Operação de Max Pooling 2 x 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Operação de convolução com filtro 3 x 3 seguida da função de ativação ReLU
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))

# Operação de flatten (convertento o mapa de características em um vetor)
model.add(Flatten())

# Camada densa com 128 nerônios seguida da função de ativação ReLU
model.add(Dense(128, activation='relu'))

# Dropout de 50% dos neurônios
model.add(Dropout(0.5))

# Camada densa de saída com 10 (um para cada dígito) seguida de função SoftMax
model.add(Dense(10,activation='softmax'))

# Resumo do modelo
model.summary()

# Avaliando a CNN treinada
score = model.evaluate(x_test, y_test)

print( '\nPerda:{:.3f}\nAcurácia:{}'.format( score[0], score[1] ) )

# Imprimindo uma imagem de exemplo
image_index = 2222
pyplot.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

# Predizendo o dígito dessa imagem
pred = model.predict( x_test[image_index].reshape(1, 28, 28, 1) )
print( '\nO valor predito é:', pred.argmax() )

# Definindo otimizador, função de perda e métrica de eficiência.
from keras.optimizers import Adam

adamOptimizer = Adam(learning_rate=0.001)

model.compile( optimizer=adamOptimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

# Efetuando o treinamento de 10 épocas com o dataset de treino e validando no dataset de validação
history = model.fit( x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=10, batch_size=16, shuffle=False )

# Plotando o histórico de treino

# Histórico de acurácia
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Acurácia do modelo no treino e validação')
pyplot.ylabel('Acurácia')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()

# Histórico da função de perda
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Perda do modelo no treino e validação')
pyplot.ylabel('Perda')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()
  
