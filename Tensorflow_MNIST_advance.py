import tensorflow as tf # Importamos la libreria 
"""
Importamos la Base de datos de numeros MNIST
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
====================================
    DEFINIENDO ENTRADAS Y MODELO
====================================
Creamos un modelo de red multicapa convolucional (3D).

Primero debemos definir un rectificador en una red neuronal,
como, una funcion de activacion o desicion:
	f(x) = max(0,x)
	x : es la entrada de la neurona

Una neurona ReLU es aquella que posee una 
unidad lineal rectificada como, como funcion de activacion:
	f(x) = ln(1 + exp(x))
	f'(x) = exp(x) / (exp(x) + 1)
	      = 1 / (1 + exp(-x))

weight_variable : funcion que crea un tensor variable
 con la funcion,
  tf.truncated_normal : 
  	creamos una variable "initial" con la
  	forma de una distribucion normal, con
  	media 0.0 y desviacion estandar 1.0.

  	el parametro shape define la forma de salida
  	de la variable
"""

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
"""
bias_variable : funcion que crea un tensor constante
 con la funcion,
  tf.constant: 
  	Agrega en los espacios del tensor 0.1 y lo crea 
  	de acuerdo al tamano del parametro shape.
Source:
https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op.html#truncated_normal
"""
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
conv2d : funcion que crea la convolucion entre 2 funciones
- es el corrimiento de una funcion con respecto a la otra,
  y la multiplicacion entre ellas genera el resultado, en
  1d.
Images:
https://en.wikipedia.org/wiki/Convolution#/media/File:Comparison_convolution_correlation.svg
Source:
https://en.wikipedia.org/wiki/Convolution
Ejem, 1d:
x = [0, 10, 10, 10, 10, 10, 10, 10, 10, 0]
W = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

R1= [0, 10, 10, 10, 10, 10, 10, 10, 10, 0]
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

R2= [0, 10, 10, 10, 10, 10, 10, 10, 10, 0]
    [8, 9, 0, 0, 0, 0, 0, 0, 0, 0]
  = [0, 90, 0, 0, 0, 0, 0, 0, 0, 0]

R3= [0, 10, 10, 10, 10, 10, 10, 10, 10, 0]
    [7, 8,  9, 0, 0, 0, 0, 0, 0, 0]
  = [0, 80, 90, 0, 0, 0, 0, 0, 0, 0]  
...

Rt= R1+R2+R3+...

en 2d:
http://www.songho.ca/dsp/convolution/convolution2d_example.html

La salida la llenan de ceros para que la input
 tenga el mismo tamano de la salida.
"""
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"""
max_pool_2x2:
- Divide la variable x en sub matrices o piscina de 2x2 (pool),
  y obtiene de cada pool el valor maximo para crear una nueva 
  matriz. Ejem:
  Matrix x = 
  1,  2,  3,  4;
  4,  3,  2,  1;
  8,  7,  6,  5;
  9, 10, 11, 14;

Pool Matrix x = 
  1,  2,        3,  12;
  4,  3,        2,  1;

  8,  7,        6,  5;
  9, 10,        11, 14;

Result Max Pool Matrix x = 
                    12,
  4                 ;
  
                 
     10,            14;

Result Max Pool Matrix x = 
  4, 12;
  10, 14;

Tambien existen tecnicas de sub muestreo que
 usan mean, y min. 

La salida la llenan de ceros para que la input
 tenga el mismo tamano de la salida.

tutorial:
http://deeplearning.net/tutorial/lenet.html
Imagen:
http://cdn-ak.f.st-hatena.com/images/fotolife/v/vaaaaaanquish/20150126/20150126055504.png
"""

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


"""
====================================
    PRIMERA CAPA CONVOLUCIONAL
====================================
recordemos que una red neuronal simple (preceptron),
  se puede definir como:
  y = w(pesos)*x + bias
Entonces, definimos la matriz de pesos de 5x5 y como
las imagenes tienen un solo color (gris normalizado)
la siguiente dimension de la matriz de pesos es 1,
hemos definido sin ningun criterio usar 32 caracteristicas,
32 variables de salidas o 32 neuronas.

Cada neurona debe tener un valor de compensacion, o 
  bias. Por eso creamos un arreglo de 32.

  Note que:
    n = 5
    2^n = 32

    La imagen se reduce de 28x28, a un espacio de 
      (n+1)x(n+1)xfeatures
      6x6x32
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

"""
Reorganizamos la imagen de entrada en un vector de 4d,
  como el de pesos.

  El primer valor del nuevo vector aplana la imagen a 1d.
  Fuente:
  https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#reshape
    Buscar en la pagina: tf.reshape(tensor, shape, name=None)
      "If one component of shape is the special value -1, the size of that dimension 
      is computed so that the total size remains constant. In particular, 
      a shape of [-1] flattens into 1-D. At most one component of shape can be -1."
  El segundo valor corresponde al Ancho de la imagen
  El tercer valor corresponde al Alto de la imagen
  El cuarto valor corresponde a los canales de la imagen
"""
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

"""
Implementamos la Red neuronal con funcion de activacion,
 ReLu, y luego hacemos un max pool de los 32 resultados.
"""
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


"""
====================================
    SEGUNDA CAPA CONVOLUCIONAL
====================================
La red neuronal profunda usa muchas capas convolucionales,
en cascada. Aqui creamos 64 caracteristicas, respuestas.

En esta se recive el max pool de la anterior capa.

  Note que:
    n = 6
    2^n = 64

    La imagen se reduce de 6x6, a un espacio de 
      (n+1)x(n+1)xfeatures
      7x7x64
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
====================================
    CAPA DENSAMENTE CONECTADA
====================================
Si deseamos que la imgen se calcule pixel a pixel, 
  recordemos que el tamano de la imagen es de 28x28 = 784
  y la potencia de 2 mas grande sera 1024.

  2^10 = 1024

  Note que:
    n = 10
    2^n = 1024

    La imagen se reduce de 7x7, a un espacio de 
      (n+1)x(n+1)
      11x11

Eso dice necesitamos 1024 neuronas para procesar estas
 imagenes.
Reorganizamos el vector de pesos para que pueda multiplicarse
 con el resultado de la anterior capa de 7x7x64.

El resultado anterior (h_pool2) se debe aplanar.
 y se crea la red densamente conectada.
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""
Para evitar el sobre ajuste (que la red se acostumbre a los
  datos de la base de datos). Se hace una eliminacion probabilistica
  de los resultados de la capa anterior.

Ademas, manejamos un parametro que SOLO habilitamos cuando 
  entrenamos. (keep_prob)

tf.nn.dropout: La funcion calcula el tamano de la red.
"""
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


"""
La ultima capa usa una funcion de activacion softmax,
y entrega un vector de tamano 10.
"""
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



"""
====================================
    EVALUACION DEL MODELO
====================================
Para evaluar el modelo calculamos:

A). LA ENTROPIA CRUZADA.
Se define en el modelo que esta bien y 
 que esta mal clasificado. Se usa la 
 "cross-entropy" (funcion de costo).

En comunicaciones "cross-entropy": mide la
 media de bits necesarios para identificar 
 un evento de un conjunto de probabilidades.

"Medira que tan eficiente son las predicciones
  con respecto a la verdad dada por los labels"

 "dependencia entre dos variables, 
  0=independientes"
  H_{y'} (y) = - sum_i [y'_i * Log(y_i)]

 y : es la distribucion de probabilidad
 y' : es el vector one-shot de los labels

 Las intrucciones multiplican cada resultado del
  Log(y) : (cada Log(label) one-shot) con la 
  dist de probabilidad hace la sumatoria y 
  reduce el resultado a un valor, la media de la
  entropia.
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

"""
B) Cambiamos la funcion del anterior ejemplo:
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  Por el optimizador de Adam: Gradiante de primer orden adaptativo.
  Fuente:
  http://arxiv.org/pdf/1412.6980.pdf

La variable feed_dict controla la eliminacion
probabilistica en cada paso de la evaluacion.
"""
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#iniciamos una sesion de entrenamiento
# con nuestras variables inicializadas
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

"""
Se haran 20 000 iteraciones de evaluacion, con lotes de 50 imagenes.
Para mantener la aletoriedad, cada 100 iteraciones SOLO se hara la
evaluacion.
"""
for i in range(20000):
  batch = mnist.train.next_batch(50)

  if i%100 == 0:
    """
    Se compara la presicion del modelo
    """        
    train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    """
    Se evalua el modelo
    """    
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
