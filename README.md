#### network.py

利用tensorflow进行基础层的搭建，包括conv_bn_relu，resnet50等；最终提供了两个接口：

```python
detnet(img, n_stack, training)
iknet(xyz, depth, width, training)
```

DetNet：

* 首先将图片数据输入到resnet50当中得到features

* 参数n_stack表示需要几次堆叠，一般取1即可；
* 每次堆叠中，将features先后输入到2D和3D检测器当中，依次得到hmap（输出21个热力图代表了2D图片中对21个关节点位置的置信度），dmap（中间特征，输出21个delta maps），lmap（输出21*3个热力图，分别代表了图片坐标系下uv坐标对应的相机坐标系下的xyz坐标），并且每次将其与已有的features进行拼接；

可能存在的问题：

* dmap本质上是一个中间特征，其中的数值没有具体的物理含义，可以考虑参考resnet50的方式，直接将其嵌入到3D探测器当中；
* lmap中存在大量的信息冗余，论文中原话为“This redundancy helps to the robustness”，可以提高网络的鲁棒性，具体效果待定；

IKNet：

* 将三维xyz坐标与归一化后的相对坐标等特征拼接为新特征，随后84\*3的特征整体输入到IKNet中（6层全连接），最终得到的是21\*4的四元数向量，输入到MANO模型中用于绘图；

可能存在的问题：

* Inverse Kinematics问题存在一对多的问题，可能可以利用父子连接关系约束解决，减少神经网络的训练；

#### capture.py

利用opencv进行照片捕获；

```python
OpenCVCapture.read()
```

#### config.py

各种model的path以及基础的参数；

#### kinematics.py

定义了MANO模型和MPII两个模型的参数以及相互的转换函数```mpii_to_mano(mpii)```和```mano_to_mpii(mano)```，以及从节点坐标转化为子节点指向父节点的单位向量与长度的```xyz_to_delta(xyz, joints_def)```；

```python
MANOHandJoints.n_joints
MANOHandJoints.labels
MANOHandJoints.mesh_mapping
MANOHandJoints.parents
```

```python
MPIIHandJoints.n_joints
MPIIHandJoints.labels
MPIIHandJoints.parents
```

#### utils.py

一些基本处理函数还有滤波器的定义；

```python
imresize(img, size)
load_pkl(path)
class LowPassFilter
class OneEuroFilter
```

#### hand_mesh.py

整合了MANO模型，四元组转矩阵表述；

```python
class HandMesh()
```

#### wrappers.py

DetNet将图片转成0~1的float32，输入到神经网络，通过去最值得到各个节点的xyz坐标和uv坐标；

IKNet将输入数据n\*3数组扩展维度之后输入到全连接层中，得到节点的旋转角度21\*4（四元组表示）；

将DetNet和IKNet封装整合到session中，输入128\*128\*3的图片，返回21\*3的节点xyz坐标以及21\*4的四元组节点方向；

```python
class ModelPipeline
```

#### app.py

利用open3d和pygame对输入输出图像进行可视化处理；

#### prepare_mano.py

主要用于转换mano模型；

```python
prepare_mano()
```

#### plot.py

绘图函数；



### 添加部分：

#### dataset.py

定义了dataset_init()函数，利用TFRecord模块搭建数据集，主要包括：

* 128\*128\*3的图片信息，用_bytes_feature存储，为基础输入；
* 21*3的相机坐标系下的xyz标注，用_floats_feature存储，采用MPII的关节顺序，转换为相对于中指指根M0的三维坐标，并以手腕W到M0的距离作为reference_bone，对长度进行归一化；主要用于DetNet的训练；
* 3*3的相机内参，用_floats_feature存储，将相机坐标系下的xyz坐标降维转换为图像坐标系下的uv坐标，可用于重建DetNet中损失函数所需的hmaps等；
* 21*4的四元数标注的各个关节相对旋转信息，用_floats_feature存储，主要用于IKNet的训练；

定义了get_batch()函数，用于训练时从中抽取数据集，再利用```tf.parse_single_example```从中解析得到各个特征；

#### train.py

最初进行测试用的神经网络训练函数，也可用于训练，但在后续跑demo时载入会出现不匹配问题；

