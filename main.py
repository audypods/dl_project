import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random as ran
#from PIL import Image, ImageFilter
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

#im=Image.open('three.jpg')
#wpercent=(28/float(im.size[0]))
#hsize=int((float(im.size[1])*float(wpercent)))
#im2=im.resize((28,hsize), Image.ANTIALIAS)

x=tf.placeholder(tf.float32,[None, 784])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)

def train_size(num):
    training_images=mnist.train.images[:num,:]
    training_labels=mnist.train.labels[:num,:]
    return (training_images, training_labels)

def test_size(num):
    testing_images=mnist.test.images[:num,:]
    testing_labels=mnist.test.labels[:num,:]
    return (testing_images, testing_labels)

def display_digits(num):
    label=training_labels[num].argmax(axis=0)
    image=training_images[num].reshape([28,28])
    plt.title('Example: %d Label: %d' %(num,label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images=training_images[start].reshape([1,784])
    for i in range(start+1,stop):
        images=np.concatenate((images, training_images[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


training_images, training_labels=train_size(55000)

#regression
x=tf.placeholder(tf.float32, shape=[None,784])


W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)


#training
sess=tf.InteractiveSession()
y_=tf.placeholder(tf.float32, shape=[None,10])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

training_images, training_labels=train_size(5500)
testing_images, testing_labels=test_size(10000)
learning_rate=0.05
train_steps=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init=tf.global_variables_initializer().run()

#evaluation
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testing_images, y_: testing_labels}))


        
