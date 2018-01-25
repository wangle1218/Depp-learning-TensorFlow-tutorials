#encoding:utf8

import tensorflow as tf 
import os 
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import flowers_cnn
import data_helps
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_EPOCH = 100
REGULARAZTION_RATE = 0.01

MODEL_SAVE_PACH = './model/'
MODEL_NAME = 'flowers_model.ckpt'

if not os.path.exists(MODEL_SAVE_PACH):
    os.mkdir(MODEL_SAVE_PACH)

def train(flowers_train,flowers_dev):
    X = tf.placeholder(tf.float32,[
                        BATCH_SIZE,
                        flowers_cnn.IMAGE_SIZE,
                        flowers_cnn.IMAGE_SIZE,
                        flowers_cnn.NUM_CHANNELS],
                        name = 'x-input')
    y_ = tf.placeholder(tf.int32,[BATCH_SIZE],name = 'y-input')
    training = tf.placeholder_with_default(False, shape=[], name='training')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    logits = flowers_cnn.Flowers_Cnn(X, train = True, regularizer = regularizer)

    global_step = tf.Variable(0, trainable = False)

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits= logits)
        loss = tf.reduce_mean(xentropy, name = 'loss') + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.name_scope('eval'):
        predictions = tf.argmax(logits, 1)
        correct = tf.nn.in_top_k(logits, y_, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_EPOCH):
            label_list = []
            prediction_list = []
            for xs, ys in data_helps.batch_iter(flowers_train, BATCH_SIZE):
                _, loss_value, step ,pred, acc = sess.run(
                                                [train_op, loss, global_step, predictions, accuracy],\
                                                feed_dict = {X: xs, y_:ys})
                print('After %d training steps, loss on training batch is %g, accuracy is %g.' % \
                        (step, loss_value, acc))

            # 批量在验证集上验证
            for x_dev, y_dev in data_helps.batch_iter(train=False, flowers_dev, BATCH_SIZE, shuffle=False):
                pre_dev = predictions.eval(feed_dict = {X: x_dev, y_: y_dev})
                label_list.extend(y_dev)
                prediction_list.extend(pre_dev)

            print("Epoch %d Precision, Recall and F1-Score..." % i)
            report = classification_report(label_list, prediction_list,\
                                    target_names=['daisy','dandelion','roses','sunflowers','tulips'])
            print(report)
            print("Epoch %d Confusion Matrix..." % i)
            cm = confusion_matrix(label_list, prediction_list)
            print(cm)

            saver.save(sess, os.path.join(MODEL_SAVE_PACH, MODEL_NAME), global_step = global_step)

def main(argv = None):
    img_dir = './flower_photos'
    flower_paths_and_classes_train,flower_paths_and_classes_test = data_helps.load_data(img_dir)
    train(flower_paths_and_classes_train,flower_paths_and_classes_test)

if __name__ == '__main__':
    tf.app.run()