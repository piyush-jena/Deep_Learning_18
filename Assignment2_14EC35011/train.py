import numpy as np
from zipfile import ZipFile
import data_loader
import module
import matplotlib.pyplot as plt

f = open('accuracy.txt', 'w')

losses = []
batch_size = 1000
DL = data_loader.DataLoader()
images,labels = DL.load_data()
input_batches,output_batches = DL.create_batches(images,labels,batch_size)
ANN = module.NN(input_batches[0].shape[1],100,10)
ANN.batch_size = batch_size
num_batches = images.shape[0]/batch_size

for i in xrange(2500):
    N = i%num_batches
    images_ = input_batches[N]
    labels_ = output_batches[N]
    ANN.batch_size = input_batches[N].shape[0]
    hidden,delta,output = ANN.forward(images_,labels_)
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, ANN.loss)
    losses.append(ANN.loss)
    ANN.backward(images_,hidden,delta)

ANN.batch_size = images.shape[0]
hidden_layer = np.maximum(0, np.dot(images, ANN.W1) + ANN.B1)
scores = np.dot(hidden_layer, ANN.W2) + ANN.B2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == labels))
s = 'training accuracy' + str((np.mean(predicted_class == labels))) + '\n'
f.write(s)
images,labels = DL.load_data('test')
ANN.batch_size = images.shape[0]
hidden_layer = np.maximum(0, np.dot(images, ANN.W1) + ANN.B1)
scores = np.dot(hidden_layer, ANN.W2) + ANN.B2
predicted_class = np.argmax(scores, axis=1)
print 'testing accuracy: %.2f' % (np.mean(predicted_class == labels))
s = 'testing accuracy' + str((np.mean(predicted_class == labels)))
f.write(s)
f.close()
########### This completes the assignment ###############
### Below is the code used to do a comparative study between SGD and gradient-descent
### It was found that gradient-descent took significantly larger time to converge
### The difference in error was marginal

plt.plot(range(len(losses)),losses)
images,labels = DL.load_data()
losses=[]
ANN = module.NN(input_batches[0].shape[1],100,10)
ANN.batch_size = 60000

for i in xrange(2500):
    hidden,delta,output = ANN.forward(images,labels)
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, ANN.loss)
    losses.append(ANN.loss)
    ANN.backward(images,hidden,delta)

plt.plot(range(len(losses)),losses,color='r')
plt.show()
