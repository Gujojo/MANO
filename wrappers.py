import os

import numpy as np
import tensorflow as tf

from config import *
from kinematics import *
from network import *
from utils import *
from dataset import *


class ModelDet:
    """
    DetNet: estimating 3D keypoint positions from input color image.
    """
    def __init__(self, model_path):
        """
        Parameters
        ----------
        model_path : str
            Path to the trained model.
        """
        print(model_path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.compat.v1.variable_scope('prior_based_hand'):
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(config=config)
                self.input_ph = tf.compat.v1.placeholder(tf.uint8, [batch_size, 128, 128, 3])
                self.feed_img = tf.cast(self.input_ph, tf.float32) / 255
                if model_path != '':
                    self.hmaps, self.dmaps, self.lmaps = \
                        detnet(self.feed_img, 1, False)
                else:
                    self.hmaps, self.dmaps, self.lmaps = \
                        detnet(self.feed_img, 1, True)

                self.hmap = self.hmaps[-1]
                self.dmap = self.dmaps[-1]
                self.lmap = self.lmaps[-1]

                self.uv = tf_hmap_to_uv(self.hmap)
                self.delta = tf.gather_nd(
                    tf.transpose(self.dmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
                )[0]
                self.xyz = tf.gather_nd(
                    tf.transpose(self.lmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
                )[0]

                self.uv = self.uv[0]
            if model_path != '':
                tf.train.Saver().restore(self.sess, model_path)

    def process(self, img):
        """
        Process a color image.

        Parameters
        ----------
        img : np.ndarray
            A 128x128 RGB image of **left hand** with dtype uint8.

        Returns
        -------
        np.ndarray, shape [21, 3]
            Normalized keypoint locations. The coordinates are relative to the M0
            joint and normalized by the length of the bone from wrist to M0. The
            order of keypoints is as `kinematics.MPIIHandJoints`.
        np.ndarray, shape [21, 2]
            The uv coordinates of the keypoints on the heat map, whose resolution is
            32x32.
        """
        results = self.sess.run([self.xyz, self.uv], {self.input_ph: img})
        return results

    def train(self):
        dataset_init()

        with self.graph.as_default():
            with tf.compat.v1.variable_scope('prior_based_hand'):
                img_batch, label_batch = get_batch()
                label = tf.compat.v1.placeholder(tf.float32, [batch_size, 21, 3])

                global_step = tf.Variable(0, trainable=False)

                reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                # loss = tf.norm(xyz-label) + reg_term
                loss = tf.norm(self.xyz - label)

                # 定义滑动平均类
                variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                variable_average_op = variable_average.apply(tf.trainable_variables())

                # 定义指数衰减学习率
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           global_step,
                                                           image_number / batch_size,
                                                           LEARNING_RATE_DECAY,
                                                           staircase=True)

                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
                with tf.control_dependencies([train_step, variable_average_op]):
                    train_op = tf.no_op(name='train')

                saver = tf.train.Saver(max_to_keep=1)

                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.sess.run(tf.compat.v1.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
                try:
                    num_batches = int(image_number / batch_size)
                    for i in range(num_epochs):
                        loss_value = 0
                        for j in range(num_batches):
                            if coord.should_stop():
                                break
                            img_tmp, label_tmp = self.sess.run([img_batch, label_batch])
                            xyz, _, loss_tmp, step = self.sess.run([self.xyz, train_op, loss, global_step], {self.input_ph: img_tmp, label: label_tmp})
                            if i >= 99:
                                print(xyz)
                            loss_value += loss_tmp
                        loss_value /= num_batches
                        print("After %d training step(s),loss is %g." % (i + 1, loss_value))

                    saver.save(self.sess, model_path, global_step=global_step)
                    print(model_path)

                except tf.errors.OutOfRangeError:
                    print('done!')

                finally:
                    coord.request_stop()
                coord.join(threads)

        # with open(json_path, 'r') as f:
        #     labels = json.loads(f.read())
        # tmp = np.array(labels)
        # tmp = tmp.astype('float32')
        # labels = tmp
        # labels -= np.expand_dims(labels[:, 9, :], axis=1)
        #
        # i = 0
        # loss = 0
        # det_model = ModelDet(DETECTION_MODEL_PATH)
        # for image_path in os.listdir(dataset_path):
        #     img = Image.open(dataset_path + '\\' + image_path)
        #     img = img.resize(image_size[0: 2])
        #     img = np.array(img)
        #
        #     label = labels[i, :]
        #
        #     refer_length = np.sqrt(np.sum(np.power(label[0, :], 2)))
        #     label /= refer_length
        #
        #     xyz = self.sess.run([self.xyz], {self.input_ph: img[np.newaxis, :]})
        #     xyz_, _ = det_model.process(img[np.newaxis, :])
        #     loss += np.linalg.norm(xyz - xyz_)
        #
        #     i += 1
        #     if i >= image_number:
        #         break
        #
        # print(loss / image_number)


class ModelIK:
    """
    IKnet: estimating joint rotations from locations.
    """
    def __init__(self, input_size, network_fn, model_path, net_depth, net_width):
        """
        Parameters
        ----------
        input_size : int
            Number of joints to be used, e.g. 21, 42.
        network_fn : function
            Network function from `network.py`.
        model_path : str
            Path to the trained model.
        net_depth : int
            Number of layers.
        net_width : int
            Number of neurons in each layer.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ph = tf.compat.v1.placeholder(tf.float32, [1, input_size, 3])
            with tf.name_scope('network'):
                self.theta = \
                    network_fn(self.input_ph, net_depth, net_width, training=False)[0]
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
            if model_path != '':
                tf.train.Saver().restore(self.sess, model_path)
            else:
                self.sess.run(tf.compat.v1.global_variables_initializer())

    def process(self, joints):
        """
        Estimate joint rotations from locations.

        Parameters
        ----------
        joints : np.ndarray, shape [N, 3]
            Input joint locations (and other information e.g. bone orientation).

        Returns
        -------
        np.ndarray, shape [21, 4]
            Estimated global joint rotations in quaternions.
        """
        theta = \
            self.sess.run(self.theta, {self.input_ph: np.expand_dims(joints, 0)})
        if len(theta.shape) == 3:
            theta = theta[0]
        return theta


class ModelPipeline:
    """
    A wrapper that puts DetNet and IKNet together.
    """
    def __init__(self):
        # load reference MANO hand pose
        mano_ref_xyz = load_pkl(HAND_MESH_MODEL_PATH)['joints']
        # convert the kinematic definition to MPII style, and normalize it
        mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
        mpii_ref_xyz -= mpii_ref_xyz[9:10]
        # get bone orientations in the reference pose
        mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
        mpii_ref_delta = mpii_ref_delta * mpii_ref_length

        self.mpii_ref_xyz = mpii_ref_xyz
        self.mpii_ref_delta = mpii_ref_delta

        self.det_model = ModelDet(DETECTION_MODEL_PATH)
        # self.det_model = ModelDet('')
        # 84 = 21 joint coordinates
        #        + 21 bone orientations
        #        + 21 joint coordinates in reference pose
        #        + 21 bone orientations in reference pose

        # self.ik_model = ModelIK(84, iknet, IK_MODEL_PATH, 6, 1024)
        self.ik_model = ModelIK(84, iknet, '', 6, 1024)

    def process(self, frame):
        """
        Process a single frame.

        Parameters
        ----------
        frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
            Frame to be processed.

        Returns
        -------
        np.ndarray, shape [21, 3]
            Joint locations.
        np.ndarray, shape [21, 4]
            Joint rotations.
        """
        xyz, _ = self.det_model.process(frame)

        return xyz

        # delta, length = xyz_to_delta(xyz, MPIIHandJoints)
        # delta *= length
        # pack = np.concatenate(
        #     [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0
        # )
        # theta = self.ik_model.process(pack)

        # return xyz, theta
