# Developped by buuvo95
# 
import tensorflow as tf

def AlexNet(input_shape, num_classes):
    # Define Initializer
    initializer = tf.keras.initializers.GlorotNormal() 

    # Define the model
    model = tf.keras.layer.Sequential()

    # First Convolution layer
    model.add(tf.keras.layers.Conv2D(filters = 96, kernel_size = (11,11), strides = 4, padding = 'valid', activation = 'relu', kernel_initializer=initializer, input_shape = input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=2, padding='valid'))

    # Second Convolution layer
    model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size=(5,5), padding='valid', strides=2, activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=2, padding='valid'))
    
    # Third Convolution layer
    model.add(tf.keras.layers.Conv2D(filters = 384, kernel_size=(3,3), padding='valid', strides=1, activation='relu', kernel_initializer=initializer))

    # Fourth Convolution layer
    model.add(tf.keras.layers.Conv3D(filters = 384, kernel_size=(3, 3), padding='valid', strides=1 , activation='relu', kernel_initializer=initializer))

    # Fifth Convolution layer
    model.add(tf.keras.layers.Conv2D(filter = 256, kernel_size=(3,3,192), padding='valid', strides=1, activation='relu', kernel_initializer=initializer))
    model.add(tf.keras.layers.MaxPooling2D((3,3), padding='valid', strides=2))

    model.add(tf.keras.layers.Flatten())
    # First Dense layer
    model.add(tf.keras.layers.Dense(units = 4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    # Second Dense layer
    model.add(tf.keras.layers.Dense(units = 4096, activation='relu'))
    mmode.add(tf.keras.layers.Dropout(0.5))
    # Third Dense layer - Output
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

# InceptionNet - Google LeNet
    # Define Inception Module
def inception_module(x,
                    filters_1x1,
                    filters_3x3_reduce,
                    filters_3x3,
                    filters_5x5_reduce,
                    filters_5x5,
                    filters_pool_proj,
                    kernel_init,
                    bias_init,
                    name=None):
    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1,1), padding='same',
                        activation='relu', kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1,1), padding='same',
                        activation='relu', kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3,3), padding='same',
                        activation='relu', kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1,1), padding='same',
                         activation='relu', kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5,5), padding='same',
                         activation='relu', kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(conv_5x5)
    
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1,1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1,1), padding = 'same',
                        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
    
    output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output

def InceptionNet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64,(7,7), padding='same', strides=(2,2), activation='relu',
            name = 'conv_1_7x7/2', kernel_initializer=tf.keras.initializers.glorot_uniform(), 
                                    bias_initializer=tf.keras.initializers.Constant(value=0.2))(inputs)
    x = tf.keras.layers.MaxPooling2D((3,3), padding='same', strides=(2,2),
                    name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init =tf.keras.initializers.Constant(value=0.2),
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_3b')

    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init =tf.keras.initializers.Constant(value=0.2),
                        name='inception_4a')


    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(num_classes, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_4d')


    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(num_classes, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_4e')

    x = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        kernel_init = tf.keras.initializers.glorot_uniform(),
                        bias_init = tf.keras.initializers.Constant(value=0.2),
                        name='inception_5b')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs, [outputs, x1, x2], name = 'inception_v1')
    return model

# XCeption Model
# Creating the Conv-BatchNorm block:
def conv_bn(x, filter, kernel_size, strides = 1):
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same',
                use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x
# Creating the SeparableConv - BatchNorm block:
def sep_bn(x, kernel_size, strides = 1):
    x = tf.keras.layers.SeparableConv2D(
        filters = filters,
        kernel_size = kernel_size,
        padding = 'same',
        use_bias = False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
# Entry flow
def entry_flow(x):
    x = conv_bn(x, filters = 32, kernel_size = 3, strides = 2)
    x = tf.keras.layers.ReLU()(x)
    x = conv_bn(x, filters = 64, kernel_size = 3, strides = 1)
    tensor = tf.keras.layers.ReLU()(x)

    x = sep_bn(tensor, filters = 128, kernel_size = 3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size = 3)
    x = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)

    tensor = conv_bn(tensor, filters = 128, kernel_size = 1, strides = 2)
    x = tf.keras.layers.Add()([tensor, x])
    
    x = tf.keras.layers.ReLU(x)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = tf.keras.layers.ReLU(x)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = tf.keras.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)

    tensor = conv_bn(tensor, filters = 728, kernel_size = 1, strides = 2)
    x = tf.keras.layers.Add()([tensor, x])
    return x
# Middle flow
def middle_flow(tensor):
    for _ in range(8):
        x = tf.keras.layers.ReLU(tensor)
        x  = sep_bn(x, filters = 728, kernel_size = 3)
        x = tf.keras.layers.ReLU()(x)
        x  = sep_bn(x, filters = 728, kernel_size = 3)
        x = tf.keras.layers.ReLU()(x)
        x  = sep_bn(x, filters = 728, kernel_size = 3)
        x = tf.keras.layers.ReLU()(x)
        tensor = tf.keras.layers.Add()([tensor, x])
    return tensor

# Exit flow
def exit_flow(tensor, num_classes):
    x = tf.keras.layers.ReLU(tensor)
    x = sep_bn(x, filters = 728, kernel_size = 3)
    x = tf.keras.layers.ReLU(tensor)
    x = sep_bn(x, filters = 1024, kernel_size = 3)
    x = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)

    tensor = conv_bn(tensor, filters = 1024, kernel_size = 1, strides = 2)
    x = tf.keras.layers.Add()([tensor, x])

    x = sep_bn(x, filters = 1536, kernel_size = 3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters = 2048, kernel_size = 3)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(units = num_classes, activations = 'softmax')(x)

    return x
def XceptionNet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape = (299,299,3))
    x = entry_flow(inputs)
    x = middle_flow(x)
    outputs = exit_flow(x, num_classes)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

# VGG model
def VGGNet(input_shape, num_classes):
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        padding = 'same'
    ))
    # 2 
    model.add(tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        padding = 'same'
    ))
    # 3
    model.add(tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        padding = 'same'
    ))
    # 4
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        padding = 'same'
    ))
    # 5
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size = (2,2),
        strides = 2,
        padding = 'same'
    ))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 4096, activations = 'relu'))
    model.add(tf.keras.layers.Dense(units = 4096, activations = 'relu'))
    model.add(tf.keras.layers.Dense(units = num_classes, activations = 'softmax'))
    return model

# Resnet code block
    # Define resnet module:
def res_net_block(input_data, filters, conv_size):
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = conv_size, activation='relu', padding='same')(input_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = conv_size, activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.Add()[x, input_data]
    x = tf.keras.layers.Activation('relu')(x)
    return x 

def ResNet(input_shape, num_res_block, num_classes):
    inputs = tf.keras.Input(shape= input_shape)
    x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation='relu')(inputs)
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((3,3))(x)

    # Retnet Block
    for i in range(num_res_block):
        x = res_net_block(input_data = x, filters = 64, conv_size = (3,3))
    
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units = 256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model