import tensorflow as tf
import numpy as np
import os
import csv
import time

from tensorflow.contrib import rnn

result_size = 256
condition_size = result_size*4
hidden_size = 256
hidden_layer_num = 3

mb_size = 64
Z_dim = 100
X_dim = result_size
y_dim = condition_size
h_dim = 128
time_steps = 1500
batch_length = 100
batch_interval = 10
batch_size = int((time_steps - batch_length) / batch_interval) + 1

""" Discriminator Net model """

X = tf.placeholder(tf.float32, shape=[None, batch_length, X_dim])
y = tf.placeholder(tf.float32, shape=[None, batch_length, y_dim])


def lstm_cell(layer_size):
    cell = rnn.BasicLSTMCell(layer_size, state_is_tuple=True)
    return cell


D_multi_cells = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(hidden_layer_num)])


def discriminator(x, y):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as vs:
        inputs = tf.concat(axis=2, values=[x, y])  # what axis?
        D_outputs, _ = tf.nn.dynamic_rnn(D_multi_cells, inputs, dtype=tf.float32)
        D_last_output = D_outputs[:, -1, :]
        D_logit = tf.contrib.layers.fully_connected(D_last_output, 1, activation_fn=None)
        D_prob = tf.nn.sigmoid(D_outputs)
        return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, batch_length, Z_dim])

G_multi_cells = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(hidden_layer_num)])


def generator(z, y):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as vs:
        inputs = tf.concat(axis=2, values=[z, y])  # what axis?
        G_output, _ = tf.nn.dynamic_rnn(G_multi_cells, inputs, dtype=tf.float32)
        G_prob = tf.round(tf.nn.sigmoid(G_output))
        return G_prob


def sample_Z(l, m, n):
    return np.random.uniform(-1., 1., size=[l, m, n])


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss)
G_solver = tf.train.AdamOptimizer().minimize(G_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

total_epoch = 1000
prev_time = time.time()

log = open('./temp.log', 'w')


def get_batch(data, batch_length, batch_interval):
    idx = 0
    result = []
    while idx + batch_length <= data.shape[0]:
        result.append(data[idx:idx+batch_length, :])
        idx += batch_interval
    result = np.array(result)
    return result


x_batches = []
y_batches = []

for data_idx in range(20):
    with open('./output{}.csv'.format(data_idx)) as f:
        rdr = csv.reader(f)
        data = []
        for line in rdr:
            if len(data) == time_steps+1:
                break
            data.append(line)
        # data = np.array([data], dtype=np.float)
        # x_batch = data[:, :, 256*3:256*4]  # bass
        # y_batch = np.concatenate((data[:, :, 0:256*3], data[:, :, 256*4:]), axis=2)  # melody, guitar, piano, drum
        data = np.array(data, dtype=np.float)
        x_batch = get_batch(data[:time_steps, 256*3:256*4], batch_length, batch_interval)
        y_batch = get_batch(np.concatenate((data[1:, :256*3], data[1:, 256*4:]), axis=1), batch_length, batch_interval)
        x_batches.append(x_batch)
        y_batches.append(y_batch)

for i in range(total_epoch):
    for data_idx in range(20):
        Z_sample = np.array(sample_Z(batch_size, batch_length, Z_dim))
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x_batches[data_idx], Z: Z_sample,
                                                                 y: y_batches[data_idx]})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_batches[data_idx]})

    if i % 20 == 0:
        log.write('Iter: {}\n'.format(i))
        log.write('D_loss: {:.4}\n'.format(D_loss_curr))
        log.write('G_loss: {:.4}\n'.format(G_loss_curr))
        log.write('time: {}\n'.format(time.time() - prev_time))
        log.write('\n')
        log.flush()
        prev_time = time.time()

    if i % 100 == 0:
        for data_idx in range(20):
            with open('./result{}.csv'.format(data_idx), 'w') as w:
                with open('./output{}.csv'.format(data_idx)) as f:
                    rdr = csv.reader(f)
                    data = []
                    for line in rdr:
                        if len(data) == time_steps+1:
                            break
                        data.append(line)
                    data = np.array(data, dtype=np.float)
                    y_batch = get_batch(np.concatenate((data[1:, :256*3], data[1:, 256*4:]), axis=1),
                                        batch_length, batch_interval)

                    wr = csv.writer(w)
                    wr.writerow([0.0 for _ in range(X_dim)])
                    temp = sess.run(generator(Z, y), feed_dict={Z: Z_sample, y: y_batch})
                    for idx in range(0, temp.shape[0], int(batch_length / batch_interval)):
                        val = temp[idx]
                        for line in val:
                            wr.writerow(line)


log.close()
