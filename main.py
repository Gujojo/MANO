import os
import numpy as np
import json

from network import *
from wrappers import *
from params import *
import config
from hand_mesh import HandMesh

hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)




# def evaluate(img):
#     with tf.Graph().as_default() as g:
#         img = tf.compat.v1.placeholder(tf.uint8, [batch_size, 128, 128, 3])
#         label = tf.compat.v1.placeholder(tf.float32, [batch_size, 21, 3])
#         input = tf.cast(img, tf.float32) / 255
#         hmaps, dmaps, lmaps = detnet(input, 1, True)
#         hmap = hmaps[-1]
#         dmap = dmaps[-1]
#         lmap = lmaps[-1]
#
#         uv = tf_hmap_to_uv(hmap)
#         xyz = tf.gather_nd(
#             tf.transpose(lmap, [0, 3, 1, 2, 4]), uv, batch_dims=2
#         )[0]
#
#         global_step = tf.Variable(0, trainable=False)
#
#         variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
#         variables_to_restore = variable_averages.variables_to_restore()
#         saver = tf.train.Saver(variables_to_restore)
#
#         with tf.Session() as sess:
#             ckpt = tf.train.get_checkpoint_state(model_path)
#             saver.restore(sess, ckpt.model_checkpoint_path)

