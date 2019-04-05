import tensorflow as tf


base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

i = 0
for layers in base_model.layers:
    i += 1
    print(i)
    print(layers)
