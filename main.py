import tensorflow as tf
print('\nNum GPUs Available:', len(tf.config.list_physical_devices('GPU')))