import os
import datetime
import time
import sys
import numpy as np
import tensorflow as tf
from mlp.data_providers import WRSDataProvider
from mlp.layer_helpers import (
    conv_layer,
    pool_layer,
    norm_layer,
    reshape_layer,
    fully_connected_layer,
    inception2d,
)

# load train data
train_data = WRSDataProvider('train', batch_size=20, convolution=True)
valid_data = WRSDataProvider('valid', batch_size=20, convolution=True)
# load valid data
valid_inputs = valid_data.inputs
valid_targets = valid_data.to_one_of_k(valid_data.targets)

#-----------------------------------------------------------------------------------------------------
#---------------------- define model graph------------------------------------------------------------

output_dim=train_data.num_classes

tf.reset_default_graph()
batch_size = tf.placeholder(tf.int32)
inputs = tf.placeholder(tf.float32, [None,
                                     train_data.inputs.shape[1],
                                     train_data.inputs.shape[2],
                                     train_data.inputs.shape[3]
                                    ], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')

# ----------------------------- NETWORK DEFINITION --------------------------------------------------
# put conv layers on cpu because too big for gtx1050
with tf.name_scope('inception-layer-1'):
    # with tf.device('/cpu:0'):
    inception1 = inception2d(inputs,
                             input_channel_dim=train_data.inputs.shape[3],
                             output_channel_dim=20,
                             kernel_sizes=[3, 5, 10, 15],
                             name="1")
with tf.name_scope('pool-layer-1'):
    pool1 = pool_layer(inception1, name="1")
with tf.name_scope('norm-layer-1'):
    norm1 = norm_layer(pool1)
with tf.name_scope('conv-layer-2'):
    conv2 = conv_layer(norm1,
                       input_channel_dim=norm1.get_shape().as_list()[3],
                       output_channel_dim=32,
                       kernel_size=5,
                       name="2")
with tf.name_scope('pool-layer-2'):
    pool2 = pool_layer(conv2, name="2")
with tf.name_scope('norm-layer-2'):
    norm2 = norm_layer(pool2)
with tf.name_scope('conv-layer-3'):
    conv3 = conv_layer(norm2,
                       input_channel_dim=norm2.get_shape().as_list()[3],
                       output_channel_dim=32,
                       kernel_size=5,
                       bias_init=0.1,
                       name="3")
with tf.name_scope('pool-layer-3'):
    pool3 = pool_layer(conv3, name="3")
with tf.name_scope('norm-layer-3'):
    norm3 = norm_layer(pool3)
# with tf.name_scope('conv-layer-4'):
#     conv4 = conv_layer(norm3,
#                        input_channel_dim=norm3.get_shape().as_list()[3],
#                        output_channel_dim=128,
#                        kernel_size=3,
#                        bias_init=0.1,
#                        name="4")
with tf.name_scope('pool-layer-4'):
    pool4 = pool_layer(pool3, name="4")
# put fcl on cpu because too big for gtx1050
# with tf.device('/cpu:0'):
with tf.name_scope('fully_connected_layer1'):
    with tf.device('/cpu:0'):
        fully_connected_layer1 = reshape_layer(pool4, 200, batch_size)
with tf.name_scope('fully_connected_layer2'):
    fully_connected_layer2 = fully_connected_layer(fully_connected_layer1, 200, 100)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fully_connected_layer2, 100, train_data.num_classes, nonlinearity=tf.identity)

with tf.name_scope('predictions'):
    predictions = tf.nn.softmax(outputs, 1)

# ------------ define error computation -------------
with tf.name_scope('error'):
    vars = tf.trainable_variables()
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets) +
                           tf.add_n([ tf.nn.l2_loss(v) for v in vars
                                     if 'bias' not in v.name]) * 0.005)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
# --- define training rule ---
with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(error)
    # train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# add summary operations
tf.summary.scalar('error', error)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# create objects for writing summaries and checkpoints during training
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = "out-deepcnn"
checkpoint_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver(max_to_keep=10)

# create arrays to store run train / valid set stats
num_epoch = 100
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)


# run on cpu because too big for gtx1050
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )

# create session and run training loop
sess = tf.Session()#(config=config)
sess.run(tf.global_variables_initializer())
train_step_no = 0
valid_step_no = 0
max_valid_accuracy = (None, 0)
min_valid_error = (None, 1000)
for e in range(num_epoch):
    current_time = time.time()
    for input_batch, target_batch in train_data:
        # do train step with current batch
        _, summary, batch_error, batch_acc = sess.run(
            [train_step, summary_op, error, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch, batch_size: target_batch.shape[0]})
        # add symmary and accumulate stats
        train_writer.add_summary(summary, train_step_no)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        train_step_no += 1
    # normalise running means by number of batches
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance

    for valid_input_batch, valid_target_batch in valid_data:
        # do valid test with current batch
        valid_summary, valid_batch_error, valid_batch_acc = sess.run(
            [summary_op, error, accuracy],
            feed_dict={inputs: valid_input_batch, targets: valid_target_batch, batch_size: valid_target_batch.shape[0]})
        # add symmary and accumulate stats
        valid_writer.add_summary(valid_summary, valid_step_no)
        valid_error[e] += valid_batch_error
        valid_accuracy[e] += valid_batch_acc
        valid_step_no += 1
    valid_error[e] /= valid_data.num_batches
    valid_accuracy[e] /= valid_data.num_batches

    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), train_step_no)
    # write stats summary to stdout
    print('-------- time elapsed: {}sec -------'.format(time.time()-current_time))
    print('Epoch {0:02d}: err(train)={1:.6f} acc(train)={2:.6f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.6f} acc(valid)={1:.6f}'
          .format(valid_error[e], valid_accuracy[e]))
    sys.stdout.flush()

    # early stopping

    if min_valid_error[1] > valid_error[e]:
        min_valid_error = (e, valid_error[e])
    elif e > min_valid_error[0] + 10:
        print('Early stopping! \nMinimum error: {}, Maximum accuracy {}'.format(min_valid_error[1], max_valid_accuracy[1]))
        break

    if max_valid_accuracy[1] < valid_accuracy[e]:
        max_valid_accuracy = (e, valid_accuracy[e])
    # elif e > max_valid_accuracy[0] + 10:
    #     break

# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()

# save run stats to a .npz file
np.savez_compressed(
    os.path.join(exp_dir, 'run.npz'),
    train_error=train_error,
    train_accuracy=train_accuracy,
    valid_error=valid_error,
    valid_accuracy=valid_accuracy
)
