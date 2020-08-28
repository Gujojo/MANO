import tensorflow as tf
import os

from params import *
from network import *
from dataset import *
from wrappers import *


def train():
    det_model = ModelDet('')

    img_batch, label_batch = get_batch()
    img = tf.compat.v1.placeholder(tf.uint8, [batch_size, 128, 128, 3])
    label = tf.compat.v1.placeholder(tf.float32, [batch_size, 21, 3])
    # xyz, _ = det_model.process(img)

    feed_img = tf.cast(img, tf.float32) / 255
    hmaps, dmaps, lmaps = \
        detnet(feed_img, 1, True)

    hmap = hmaps[-1]
    dmap = dmaps[-1]
    lmap = lmaps[-1]

    uv = tf_hmap_to_uv(hmap)
    delta = tf.gather_nd(
        tf.transpose(dmap, [0, 3, 1, 2, 4]), uv, batch_dims=2
    )[0]
    xyz = tf.gather_nd(
        tf.transpose(lmap, [0, 3, 1, 2, 4]), uv, batch_dims=2
    )[0]

    global_step = tf.Variable(0, trainable=False)

    reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # loss = tf.norm(xyz-label) + reg_term
    loss = tf.norm(xyz - label)

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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            batch_idxs = int(image_number / batch_size)
            for i in range(num_epochs):
                for j in range(batch_idxs):
                    if coord.should_stop():
                        break
                    img_tmp, label_tmp = sess.run([img_batch, label_batch])

                    _, loss_value, step = sess.run([train_op, loss, global_step], {img: img_tmp, label: label_tmp})
                    print("After %d training step(s),loss is %g." % (step, loss_value))
            saver.save(sess, model_path, global_step=global_step)
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


def test():
    with open(json_path, 'r') as f:
        label_list = json.loads(f.read())
    with open(k_path, 'r') as f:
        k_list = json.loads(f.read())
    labels = np.array(label_list)
    k_array = np.array(k_list)
    labels = labels.astype('float32')
    labels -= np.expand_dims(labels[:, 9, :], axis=1)

    det_model = ModelDet(DETECTION_MODEL_PATH)
    i = 0
    loss = 0
    for image_path in os.listdir(dataset_path):
        img = Image.open(dataset_path + '\\' + image_path)
        img = img.resize(image_size[0: 2])
        img = np.array(img)

        tmp = labels[i, :]
        k = k_array[i, :]
        # label = np.matmul(k, tmp.T).T
        label = tmp
        refer_length = np.sqrt(np.sum(np.power(label[0, :], 2)))
        label /= refer_length

        if i == 23:
            print(np.sqrt(np.sum(np.power(label, 2), axis=1)))
            print('')
        xyz, _ = det_model.process(img[np.newaxis, :])
        res = xyz - label
        loss += np.linalg.norm(xyz - label)

        i += 1
        if i >= image_number:
            break

    print(loss / image_number)


def main():
    # det_model = ModelDet('')
    # det_model.train()
    test()


if __name__ == '__main__':
    main()
