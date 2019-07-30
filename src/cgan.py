import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import os
import cv2

# import session image
piano = plt.imread('./../result_images_gray/piano.png')[:, :, 0]
bass = plt.imread('./../result_images_gray/bass.png')[:, :, 0]
drum = plt.imread('./../result_images_gray/drum.png')[:, :, 0]
guitar = plt.imread('./../result_images_gray/guitar.png')[:, :, 0]
melody = plt.imread('./../result_images_gray/melody.png')[:, :, 0]

# reshape session images to 1d array
piano = piano.reshape(-1)
bass = bass.reshape(-1)
drum = drum.reshape(-1)
guitar = guitar.reshape(-1)
melody = guitar.reshape(-1)

###################################################################
# - Object session : bass (X)                                     #
# - Other sessions : piano, drum, guitar, melody (Y)              #
#         order    : melody - piano - guitar - drum               #
###################################################################

obj_session = bass
other_sessions = np.concatenate((melody, piano, guitar, drum))

X_dim = obj_session.shape[0]
y_dim = other_sessions.shape[0]
Z_dim = 100  # noise dimension
h_dim = 128  # hidden node dimension
batch_size = 1


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Implement Discriminator
X = tf.placeholder(tf.float32, shape=[None, None])
y = tf.placeholder(tf.float32, shape=[None, None])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
# print(D_W1.shape)
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
# print(D_b1.shape)
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    d_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    d_logit = tf.matmul(d_h1, D_W2) + D_b2
    d_prob = tf.nn.sigmoid(d_logit)

    return d_prob, d_logit


# Implement Generator
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
# print(Z.shape)
G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    # print(y.shape)
    # print(inputs.shape)
    g_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    print(g_h1.shape)
    g_log_prob = tf.matmul(g_h1, G_W2) + G_b2
    g_prob = tf.nn.sigmoid(g_log_prob)

    return g_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # print(sample.shape)
        sample = sample.reshape(128, -1)
        # print(np.array([[[sample[i][j], sample[i][j], sample[i][j], 1.0] for j in range(sample.shape[1])] for i in range(sample.shape[0])]))
        plt.imsave('../gan_result/{}.png'.format(str(i).zfill(3)), np.array([[[sample[i][j], sample[i][j], sample[i][j], 1.0] for j in range(sample.shape[1])] for i in range(sample.shape[0])]), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('./../gan_result'):
    os.makedirs('./../gan_result')

i = 0

for it in range(100000):
    if it % 1000 == 0:
        n_sample = 1

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 7] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        # print(fig.shape)
        # plt.savefig('./../gan_result/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')

        i += 1
        # plt.close(fig)

    X_mb = obj_session.reshape(1, obj_session.shape[0])
    y_mb = other_sessions.reshape(1, other_sessions.shape[0])

    Z_sample = sample_Z(batch_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 10 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
    # if it == 0:
    #     n_sample = 1
    #
    #     Z_sample = sample_Z(n_sample, Z_dim)
    #     y_sample = np.zeros(shape=[n_sample, y_dim])
    #     y_sample[:, 7] = 1
    #
    #     samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
    #
    #     fig = plot(samples)
    #     # print(fig.shape)
    #     # plt.savefig('./../gan_result/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    #
    #     i += 1
    #     # plt.close(fig)
    #     break