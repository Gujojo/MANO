from PIL import Image
import json

from params import *
from network import *
from transforms3d import quaternions

# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成浮点数组型的属性（value不带中括号）
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dataset_init():
    with open(json_path, 'r') as f:
        label_list = json.loads(f.read())
    with open(k_path, 'r') as f:
        k_list = json.loads(f.read())
    labels = np.array(label_list)
    k_array = np.array(k_list)

    labels = labels.astype('float32')
    labels -= np.expand_dims(labels[:, 9, :], axis=1)

    i = 0
    writer = tf.python_io.TFRecordWriter(record_path)
    for image_path in os.listdir(dataset_path):
        img = Image.open(dataset_path + '\\' + image_path)
        img = img.resize(image_size[0: 2])
        img_raw = img.tobytes()

        tmp = labels[i, :]
        k = k_array[i, :]
        # label = np.matmul(k, tmp.T).T
        label = tmp
        ref_pose = label.copy()
        quats = []
        for n in range(21):
            if parents[n] == None:
                quats.append(quaternions.axangle2quat([1, 0, 0], 0))
            elif parents[n] == 0:
                ref_pose[n] -= label[parents[n]]
                a = -ref_pose[0]
                b = ref_pose[n]
                quats.append(rot2quat(a, b))
            else:
                ref_pose[n] -= label[parents[n]]
                a = ref_pose[parents[n]]
                b = ref_pose[n]
                quats.append(rot2quat(a, b))
        quats = np.array(quats)
        print(quats)
        refer_length = np.sqrt(np.sum(np.power(label[0, :], 2)))
        label /= refer_length
        label = label.flatten()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': _floats_feature(label),
                    'img_raw': _bytes_feature(img_raw)
                }
            )
        )
        writer.write(example.SerializeToString())

        i += 1
        if i >= image_number:
            break

    writer.close()
    return


def get_batch():
    tf_queue = tf.train.string_input_producer([record_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.VarLenFeature(tf.float32),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    label = tf.sparse_tensor_to_dense(features['label'], default_value=0)
    label_decode = tf.reshape(label, [21, 3])

    img_raw = features['img_raw']
    img_decode = tf.decode_raw(img_raw, tf.uint8)
    img_decode = tf.reshape(img_decode, image_size)

    img_batch, label_batch = tf.train.shuffle_batch([img_decode, label_decode],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue,
                                                    allow_smaller_final_batch=False
                                                    )

    return img_batch, label_batch


def rot2quat(a, b):
    theta = np.arccos(min(1.0, a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    vector = np.cross(a, b + 2 ** -8)

    quat = quaternions.axangle2quat(vector, theta)
    return quat


def main():
    dataset_init()
    img_batch, label_batch = get_batch()
    print(label_batch)


if __name__ == '__main__':
    main()
