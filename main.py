import time
import numpy as np
import tensorflow as tf
import tensornets as nets
import os
from tensornets.datasets import voc
val_length = 71
trains_length= 213
data_dir = os.getcwd()+'\data\VOC2012'
trains = voc.load_train([data_dir],
                        'train',
                        batch_size=2,classes=4)

# Define a model
inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
outputs = tf.placeholder(tf.float32,[None,4])
is_training = tf.placeholder(tf.bool)
#model = nets.YOLOv2(inputs, nets.Darknet19, is_training=is_training,classes=4)
model = nets.YOLOv2VOC(inputs,is_training=is_training,classes=4)
train_list = model.get_weights()[-2:]
print(len(train_list))
print(train_list)
# Define an optimizer
step = tf.Variable(0, trainable=False)
lr = tf.train.piecewise_constant(
    step, [100, 180, 320, 570, 1000, 40000, 60000],
    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-4, 1e-5])


update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.MomentumOptimizer(lr, 0.9).minimize(model.loss,
                                                         var_list=train_list)
with tf.Session() as sess:
    # Load Darknet19
    #print(sess.run(train_list[-1].shape))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(model.pretrained())
    saver = tf.train.Saver()
    losses = []
    for i in range(233):

        # Iterate on VOC0_12 trainval once
        _t = time.time()
        for (imgs, metas) in trains:

            if imgs is None:
                break
            metas.insert(0, model.preprocess(imgs))  # for `inputs`
            metas.append(True)

            # for `is_training`
            print('training '+str(i))
            outs = sess.run([train, model.loss],
                            dict(zip(model.inputs, metas)))
            losses.append(outs[1])
            saver.save(sess, str(os.getcwd())+r'\model' + r"\{}.ckpt".format(i))

        # Report step, learning rate, loss, weight decay, runtime
        print('***** %d %.5f %.5f %.5f %.5f *****' %
              (sess.run(step), sess.run(lr),
               losses[-1], sess.run(tf.losses.get_regularization_loss()),
               time.time() - _t))

        results = []
        tests = voc.load(data_dir, 'val', total_num=val_length)
        for (img, scale) in tests:
            outs = sess.run(model, {inputs: model.preprocess(img),
                                    is_training: False})
            results.append(model.get_boxes(outs, img.shape[1:3]))
        print(voc.evaluate(results, data_dir, 'val'))

