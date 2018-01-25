#encoding:utf8

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_data(noise=0.1):
    from sklearn.datasets import make_moons
    m = 2000
    X_moons, y_moons = make_moons(m, noise=noise, random_state=42)
    return X_moons, y_moons

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def fc_layers(input_tensor,regularizer):
    HINDENN1 = 6
    HINDENN2 = 4
    with tf.name_scope("full-connect-layer"):
        fc1 = tf.layers.dense(input_tensor, HINDENN1, activation=tf.nn.elu,\
            kernel_regularizer=regularizer, name="fc1")
        fc2 = tf.layers.dense(fc1, HINDENN2, activation=tf.nn.elu,\
            kernel_regularizer=regularizer, name="fc2")
    return fc2

def train(data, label,learning_rate,lambd,n_epochs,batch_size):
    test_ratio = 0.2
    test_size = int(len(data) * test_ratio)
    X_train = data[:-test_size]
    X_test = data[-test_size:]
    y_train = label[:-test_size]
    y_test = label[-test_size:]

    n_inputs = X_train.shape[1]
    n_outputs = len(set(y_train))
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

    regularizer = tf.contrib.layers.l2_regularizer(lambd)
    fc2 = fc_layers(X,regularizer)
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc2, n_outputs, kernel_regularizer=regularizer,name="output")

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits= logits)
        loss = tf.reduce_mean(xentropy, name = 'loss')
        loss_summary = tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable = False)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.name_scope('eval'):
        predictions = tf.argmax(logits, 1)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        acc_summary = tf.summary.scalar('acc', accuracy)

    summary_op = tf.summary.merge([loss_summary, acc_summary])

    checkpoint_path = "./chickpoints/model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./chickpoints/model"

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = './logs/'+ now
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()

    n_epochs = n_epochs
    batch_size = batch_size
    n_batches = int(np.ceil(len(data) / batch_size))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            loss_val, summary_str,test_pred, test_acc = sess.run(
                                            [loss, summary_op,predictions, accuracy],\
                                            feed_dict={X: X_test, y: y_test})

            file_writer.add_summary(summary_str, epoch)
            if epoch % 50 == 0:
                print("Epoch:", epoch, "\tLoss:", loss_val,"\tAcc:",test_acc)
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))

        saver.save(sess, final_model_path)
        y_pred = predictions.eval(feed_dict={X: X_test, y: y_test})
        print('precision_score',precision_score(y_test, y_pred))
        print('recall_score',recall_score(y_test, y_pred))

        sess.close()


if __name__ == '__main__':
    X_moons, y_moons = load_data(noise=0.1)

    learning_rate = 0.001
    lambd = 0.01
    n_epochs = 5000
    batch_size = 64
    train(X_moons, y_moons,learning_rate,lambd,n_epochs,batch_size)