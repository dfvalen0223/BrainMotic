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
Creamos el arreglo de variables simbolicas x que recibe las
entradas de:

28x28 px = 784, [y axis]
None =  porque pueden ser N imagenes (Any), [x axis]
			^ 784 [px]
			|
			|---> # of images
SON DATOS NO MODIFICABLES
"""
x = tf.placeholder(tf.float32, [None, 784]) 
#							   [x axis, y axis]
"""
Creamos las variables para los pesos W:
W1,1 - W2,1 - W3,1 - W4,1 - W5,1 - W6,1 - W7,1 - W8,1 - W9,1 - W0,1
W1,2 - W2,2 - W3,2 - W4,2 - W5,2 - W6,2 - W7,2 - W8,2 - W9,2 - W0,2
...
W1,784 - W2,784 - W3,784 - W4,784 - W5,784 - W6,784 - W7,784 - W8,784 - W9,784 - W0,784
			^ 10 clases
			|
			|---> 784 [px]
SON DATOS MODIFICABLES, inician en cero.
En esta variable será donde se ajusta el algoritmo
	de aprendizaje
"""
W = tf.Variable(tf.zeros([784, 10]))

"""
Creamos las variables para los sesgos b o bias por cada clase
SON DATOS MODIFICABLES y unidimensionales, inician en cero
En esta variable será donde se ajusta el algoritmo
	de aprendizaje
"""
b = tf.Variable(tf.zeros([10]))

"""
Se define el modelo.
	x*W : tf.matmul(x,W)
	tf.nn.softmax : neural network 

Recordemos la multiplicación matriz x matriz:
	A_{mxn} X B_{nxp} = C_{mxp}
	x_{None, 784} X W_{784, 10} = C_{None, 10}

	C_{None, 10} + b_{10} = C1_{10}
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)
"""
====================================
    ENTRENANDO MODELO
====================================
Se define en el modelo que esta bien y 
 que esta mal clasificado. Se usa la 
 "cross-entropy" (función de costo).

En comunicaciones "cross-entropy": mide la
 media de bits necesarios para identificar 
 un evento de un conjunto de probabilidades.

"Medirá que tan eficiente son las predicciones
 	con respecto a la verdad dada por los labels"

 "dependencia entre dos variables, 
 	0=independientes"
 	H_{y'} (y) = - sum_i [y'_i * Log(y_i)]

 y : es la distribución de probabilidad
 y' : es el vector one-shot de los labels

 Las intrucciones multiplican cada resultado del
  Log(y) : (cada Log(label) one-shot) con la 
  dist de probabilidad hace la sumatoria y 
  reduce el resultado a un valor, la media de la
  entropia.
"""
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""
TF usa una red neural en backpropagation:
	1. Ajusta los pesos (W) y valores de bias (b)
	   desde las entradas hasta las salidas
	2. Se devuelve ajustando nuevos valores
	   de pesos (W) y bias (b)
TF usa un "gradiente descendiente":
	busca el valor minimo de la función de probabilidad
	usando la entropia cruzada para enseñar al algoritmo
	los patrones. (gradiente descendiente)

	usa 0.5 como un porcentaje de aprendizaje.
	que es cuanto aprende el algoritmo de la
	base de datos, para que cuando llegue un nuevo
	dato lo reconozca, si es uno será muy restringido
	si es muy bajo no aprende nada y no reconocerá nada.
	
Usa ademas un gradiente descendiente.
"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# iniciamos todas las variables
init = tf.initialize_all_variables()
#iniciamos una sesión de entrenamiento
# con nuestras variables inicializadas
sess = tf.Session()
sess.run(init)

"""
Realizamos el entramiento 1000 veces.
El entrenamiento lo hacemos con batchs o lotes de datos
 de 100 datos en cada pasadas.
Esto entrena la red neuronal 1000 veces con 100 datos cada
 vez.
 xs : son las imagenes
 ys : son las etiques o labels
"""
for i in range(1000):
	"""
	Tomo lotes de 100 datos aleatoriamente en cada iteración o pasada
	 xs : son las imagenes
 	 ys : son las etiques o labels

 	 La toma aleatoria, permite que el algorimo no espere
 	 siempre el mismo patrón.
 	 "como en una asignatura se espera encontrar que los 
 	 examenes NO sean estocasticos, si no repetitivos y
 	 poder encontrar un patrón en una base de datos de
 	 examenes"
 	 "las asignaturas NO tienen examenes estocasticos"
 	 estocastico = aleatorio
	"""
  batch_xs, batch_ys = mnist.train.next_batch(100)
	"""
	Alimentamos la red neuronal con la una variable de 
	entrada train_step, que será luego los placeholder "x"
	"""  
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""
====================================
    EVALUANDO EL MODELO
====================================
Vamos a averiguar si el algoritmo clasifica bien.

tf.argmax : asigna un indice a la entrada con probabilidad mas
			alta.
Por ejemplo: el algoritmo entrega tf.argmax(y,1) creyendo que
	el dato que entra es 1, y el verdadero es tf.argmax(y_,1)
	(porque lo conozco).

Para comparar los resultados usamos:
 tf.equal : que entrega un vector booleano [True, False, True, True]
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

"""
tf.reduce_mean :  promedia el peso de la etiquetas.
Ej. 4 verdaderas 1 falsa entonces (1 - 1falsa/4verdaderas = 0.75)
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Al final verficamos la exactitud de la predicción
"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
