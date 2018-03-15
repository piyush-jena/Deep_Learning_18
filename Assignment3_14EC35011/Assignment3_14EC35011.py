import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_loader
import sys
import argparse
from sklearn.linear_model import LogisticRegression as LR

tf.set_random_seed(123)  # reproducibility

parser = argparse.ArgumentParser()

parser.add_argument('--train', action="store_true",dest='train',default=False)
parser.add_argument('--test', action="store_true",dest='test',default=False)
parser.add_argument('--layer=1', action="store_true",dest='layer1',default=False)
parser.add_argument('--layer=2', action="store_true",dest='layer2',default=False)
parser.add_argument('--layer=3', action="store_true",dest='layer3',default=False)

# Link to weights: https://1drv.ms/u/s!AuWnQxJtgcQUiAB9trtrJbLA2L7k


def relu(x):
    return tf.maximum(x, 0)

def softmax(x):
	return tf.divide(tf.exp(tf.subtract(x,tf.reduce_max(x))), tf.reduce_sum(tf.subtract(x,tf.reduce_max(x))))

DL = data_loader.DataLoader()
train_data,train_labels_nhv = DL.load_data()
train_labels = np.eye(10)[np.asarray(train_labels_nhv, dtype=np.int32)]

test_data,test_labels_nhv = DL.load_data('test')
test_labels = np.eye(10)[np.asarray(test_labels_nhv, dtype=np.int32)]

##Normalization
train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)
train_data = train_data/255
test_data = test_data/255

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=123)

training_epochs = 100
batch_size = 100

class NN(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.mode = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32)
            self.h1 = tf.layers.dense(inputs=self.X, units=500)
            self.h1 = relu(self.h1)
            self.h2 = tf.layers.dense(inputs=self.h1, units=500)
            self.h2 = relu(self.h2)
            self.h3 = tf.layers.dense(inputs=self.h2, units=500)
            self.h3 = relu(self.h3)
            self.logits = tf.layers.dense(inputs=self.h3, units=10)
            self.pred = tf.nn.softmax(self.logits)
            self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            #tf.summary.scalar('pred',self.pred)
            self.learning_rate = 0.001
            self.cost = -tf.reduce_sum(self.Y*tf.log(self.pred))
            #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('mean_loss', self.cost)
            self.merged = tf.summary.merge_all()

if (parser.parse_args().train):
    nn = NN()
    best_validation_accuracy = 0.0
    last_improvement = 0
    patience = 10

    sv = tf.train.Supervisor(graph=nn.graph, logdir='weights/', summary_op=None, save_model_secs=0)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(train_data) / batch_size)
            if sv.should_stop():
                break
            for i in range(total_batch):
                batch_xs, batch_ys = train_data[(i)*batch_size:(i+1)*batch_size], train_labels[(i)*batch_size:(i+1)*batch_size]
                feed_dict = {nn.X: batch_xs, nn.Y: batch_ys, nn.mode:True, nn.keep_prob:0.8}
                c, _ = sess.run([nn.cost, nn.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                if i%50:
                    sv.summary_computed(sess, sess.run(nn.merged, feed_dict))
                    gs = sess.run(nn.global_step, feed_dict)
            if np.isnan(avg_cost):
                print("Early stopping ...")
                break
            print 'Epoch : ' + str(epoch) + ' Training Loss: ' + str(avg_cost)
            acc = sess.run(nn.accuracy, feed_dict={nn.X: val_data, nn.Y: val_labels, nn.mode:False, nn.keep_prob:1.0})
            print 'Validation Accuracy: ' + str(acc)
            if acc > best_validation_accuracy:
                last_improvement = epoch
                best_validation_accuracy = acc
                sv.saver.save(sess, 'weights' + '/model_gs', global_step=gs)
            if epoch - last_improvement > patience:
                print("Early stopping ...")
                break

if (parser.parse_args().test):
    nn = NN()
    print("Graph loaded")
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
            print("Restored!")
            acc = sess.run(nn.accuracy, feed_dict={nn.X: test_data, nn.Y: test_labels, nn.mode:False, nn.keep_prob:1.0})
            print('Accuracy:', acc)

if (parser.parse_args().layer1):
    nn = NN()
    print("Graph loaded")
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
            var = [v for v in tf.trainable_variables() if v.name == "dense/kernel:0"][0]
            w1 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense/bias:0"][0]
            b1 = sess.run(var)
            H1 = np.dot(train_data,w1)+b1
            H1 = H1*(H1>0)
            model = LR()
            model.fit(H1,train_labels_nhv[:54000])
            h1 = np.dot(test_data,w1)+b1
            h1 = h1*(h1>0)
            pred_label = model.predict(h1)
            score = np.eye(10)[np.asarray(pred_label, dtype=np.int32)]
            print 'Layer1 - Logistic Regression Accuracy: %.5f' % (np.mean(score == test_labels))


if (parser.parse_args().layer2):
    nn = NN()
    print("Graph loaded")
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
            var = [v for v in tf.trainable_variables() if v.name == "dense/kernel:0"][0]
            w1 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense/bias:0"][0]
            b1 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_1/kernel:0"][0]
            w2 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_1/bias:0"][0]
            b2 = sess.run(var)
            H1 = np.dot(train_data,w1)+b1
            H1 = H1*(H1>0)
            H2 = np.dot(H1,w2)+b2
            H2 = H2*(H2>0)
            model = LR()
            model.fit(H2,train_labels_nhv[:54000])
            h1 = np.dot(test_data,w1)+b1
            h1 = h1*(h1>0)
            h2 = np.dot(h1,w2)+b2
            h2 = h2*(h2>0)
            pred_label = model.predict(h2)
            score = np.eye(10)[np.asarray(pred_label, dtype=np.int32)]
            print 'Layer2 - Logistic Regression Accuracy: %.5f' % (np.mean(score == test_labels))


if (parser.parse_args().layer3):
    nn = NN()
    print("Graph loaded")
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
            var = [v for v in tf.trainable_variables() if v.name == "dense/kernel:0"][0]
            w1 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense/bias:0"][0]
            b1 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_1/kernel:0"][0]
            w2 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_1/bias:0"][0]
            b2 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_2/kernel:0"][0]
            w3 = sess.run(var)
            var = [v for v in tf.trainable_variables() if v.name == "dense_2/bias:0"][0]
            b3 = sess.run(var)
            H1 = np.dot(train_data,w1)+b1
            H1 = H1*(H1>0)
            H2 = np.dot(H1,w2)+b2
            H2 = H2*(H2>0)
            H3 = np.dot(H2,w3)+b3
            H3 = H3*(H3>0)
            model = LR()
            model.fit(H3,train_labels_nhv[:54000])
            H1 = np.dot(test_data,w1)+b1
            H1 = H1*(H1>0)
            H2 = np.dot(H1,w2)+b2
            H2 = H2*(H2>0)
            H3 = np.dot(H2,w3)+b3
            H3 = H3*(H3>0)
            pred_label = model.predict(H3)
            score = np.eye(10)[np.asarray(pred_label, dtype=np.int32)]
            print 'Layer3 - Logistic Regression Accuracy: %.5f' % (np.mean(score == test_labels))
