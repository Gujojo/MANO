dataset_path = r'..\dataset\FreiHAND\training\rgb'
record_path = r'.\train.tfrecords'
model_path = r'.\model\model.ckpt'
json_path = r'..\dataset\FreiHAND\training_xyz.json'
k_path = r'..\dataset\FreiHAND\training_K.json'
image_size = (128, 128, 3)
# image_number = 32560
image_number = 100

batch_size = 1
capacity = 2000
min_after_dequeue = 10

num_epochs = 5

MOVING_AVERAGE_DECAY = 0.9
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]
