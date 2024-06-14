import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trainingData import *

tf.compat.v1.disable_eager_execution()  # just letting tf 1.x features to work on tf 2.x

iterations = 10001 # how many training iterations you want
deltaX = 10 # just writes in the csv file every 10 times

# matrix, Z, of random values from -1 to 1
def sampleZ(m, n):
    return np.random.uniform(-1., 1., size = [m, n])

# generator network which takes Z, random values, outputs data, trying to be as close to real as possible
def generator(Z, hsize = [16, 16], reuse = False):
    with tf.compat.v1.variable_scope("GAN/Generator", reuse = reuse):
        h1 = tf.compat.v1.layers.dense(Z, hsize[0], activation = tf.nn.leaky_relu)
        h2 = tf.compat.v1.layers.dense(h1, hsize[1], activation = tf.nn.leaky_relu)
        output = tf.compat.v1.layers.dense(h2, 2)
    return output

# discriminator network which takes X, generated data and outputs whether generator data real or fake
def discriminator(X, hsize = [16, 16], reuse = False):
    with tf.compat.v1.variable_scope("GAN/Discriminator", reuse = reuse):
        h1 = tf.compat.v1.layers.dense(X, hsize[0], activation = tf.nn.leaky_relu)
        h2 = tf.compat.v1.layers.dense(h1, hsize[1], activation = tf.nn.leaky_relu)
        h3 = tf.compat.v1.layers.dense(h2, 2)
        output = tf.compat.v1.layers.dense(h3, 1)
    return output, h3

# calculates how accurate discrminator is
def calculate_accuracy(real_logits, fake_logits):
    realCorrect = tf.reduce_mean(tf.cast(tf.greater(real_logits, 0.5), tf.float32))
    fakeCorrect = tf.reduce_mean(tf.cast(tf.less(fake_logits, 0.5), tf.float32))
    return (realCorrect + fakeCorrect) / 2

# X = real data
# Y = placeholder for random input to generator
X = tf.compat.v1.placeholder(tf.float32, [None, 2])
Z = tf.compat.v1.placeholder(tf.float32, [None, 2])

graphSample = generator(Z) # gets data from the generator
rLogits, rRep = discriminator(X) # discriminators output for real data
fLogits, gRep = discriminator(graphSample, reuse = True) # discriminators output for generated data

# gets the loss for discriminator by combining real and generated data losses
discLoss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = rLogits, labels = tf.ones_like(rLogits)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits = fLogits, labels = tf.zeros_like(fLogits)))

# loss for generator, making it help produce data that the discriminator classifies true 
genLoss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = fLogits, labels = tf.ones_like(fLogits)))

# accuracy of discrim.
accuracy = calculate_accuracy(rLogits, fLogits)

genVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator") # variables in generator
discVars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator") # variables in discrim

genStep = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.001).minimize(genLoss, var_list = genVars)  # optimizer step for generator
discStep = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.001).minimize(discLoss, var_list = discVars)  # optimizer step for discrim

# jus creating tf sesh =)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batchSize = 256 
ndSteps = 10 # no. of training steps per iteration for discrim
ngSteps = 10 # no. of training steps per iteration for generator

# jus plotting the trained data
xPlot = sampleData(n = batchSize) 

# jus opening CSV file to log stuff
yes = open('logs.csv', 'w')
yes.write('Iteration, Discriminator Loss, Generator Loss, Accuracy\n')

# training loop - trains discrim and generator, calculates losses for both & accuracy, logs to csv, plots w/ matplotlib
for i in range(iterations):  
    # data pipeline 
    XBatch = sampleData(n = batchSize)
    ZBatch = sampleZ(batchSize, 2)

    for _ in range(ndSteps):
        _, dLoss = sess.run([discStep, discLoss], feed_dict = {X: XBatch, Z: ZBatch})

    rrep_dstep, grep_dstep = sess.run([rRep, gRep], feed_dict = {X: XBatch, Z: ZBatch})

    for _ in range(ngSteps):
        _, gLoss = sess.run([genStep, genLoss], feed_dict = {Z: ZBatch})

    rrep_gstep, grep_gstep = sess.run([rRep, gRep], feed_dict = {X: XBatch, Z: ZBatch})
    
    acc = sess.run(accuracy, feed_dict = {X: XBatch, Z: ZBatch})


    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f\t Accuracy: %.4f" % (i, dLoss, gLoss, acc))

    # just writes in the csv file every X times
    if i % deltaX == 0: 
        yes.write("%d, %f, %f, %f\n" % (i, dLoss, gLoss, acc))

    # just splits up every 1000 iterations to see the comparisons
    if i % 1000 == 0:
        plt.figure()
        gPlot = sess.run(graphSample, feed_dict = {Z: ZBatch})
        xax = plt.scatter(xPlot[:, 0], xPlot[:, 1])
        gax = plt.scatter(gPlot[:, 0], gPlot[:, 1])
        
        # comparison between real trained data and generated matplot
        plt.legend((xax, gax), ("Real Data", "Generated Data"))
        plt.title('Iteration %d' % i)
        plt.tight_layout()
        plt.show()
        # plt.close()

        # transformed features at iteration X matplot
        plt.figure()
        rrg = plt.scatter(rrep_gstep[:, 0], rrep_gstep[:, 1], alpha = 0.5)
        grd = plt.scatter(grep_dstep[:, 0], grep_dstep[:, 1], alpha = 0.5)
        grg = plt.scatter(grep_gstep[:, 0], grep_gstep[:, 1], alpha = 0.5)

        plt.legend((rrg, grd, grg), ("Real Data After G step", "Generated Data Before G step", "Generated Data After G step"))
        plt.title('Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        # plt.show()
        plt.close()

yes.close()