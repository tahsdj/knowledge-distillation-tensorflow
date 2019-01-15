import tensorflow as tf
import numpy as np


class DeepConvNet():
    def __init__(self, 
                    x, 
                    image_size, 
                    n_chennel, 
                    kernel_sizes, 
                    n_feature_maps, 
                    index_of_pool_layers, 
                    n_classes,
                    temperature = 5,
                    alpha = .9,
                    learning_rate = 1e-3
                ):
        self.n_conv_layers = n_conv_layers
        self.image_size = image_size
        self.n_chennel = n_chennel
        self.n_classes = n_classes
        self.kernel_sizes = kernel_sizes
        self.n_feature_maps = n_feature_maps
        self.attention_maps = []
        self.index_of_pool_layers = index_of_pool_layers
        self.training = True
        self.output_logits = None
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.loss = 0

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



    def max_pool_2x2(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def max_pool_4x4(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    def max_pool_8x8(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

    def conv_layer(self, inputs, kernel_size, n_chennel, n_feature_map, scope):
        with tf.name_scope(scope):
            x = inputs
            w = self.weight_variable([kernel_size, kernel_size, n_chennel, n_feature_map])
            b = self.bias_variable([n_feature_map])
            x = self.conv2d(x,w) + b

            #batch normalization
            x = tf.layers.batch_normalization(x, axis=3, training=self.training)
            x = tf.nn.relu(x)
            return x

    def fc_layer(self, inputs, output_size, use_relu=True):
        with tf.name_scope(scope):
            x = inputs
            w = self.weight_variable([(inputs.shape)[1], output_size])
            b = self.bias_variable([output_size])
            x = tf.matmul(x,w) + b
            
            x = tf.layers.batch_normalization(x, axis=1, training=self.training)
            if use_relu:
                x = tf.nn.relu(x)
            # x = tf.nn.relu(x)
            return x
    
    def build_model(self, inputs):
        
        #build first layer
        x = None
        for index, kernel_size, n_feature_map in zip(range(self.kernel_sizes), self.kernel_sizes, self.n_feature_maps):
            if index == 0:
                x = self.conv_layer(inputs, kernel_size, self.n_chennel, n_feature_map, scope="conv_layer"+str(index+1))
            else:
                if index in set(self.index_of_pool_layers):
                    x = self.max_pool_2x2(x)
                x = self.conv_layer(x, kernel_size, kernel_size, self.n_feature_maps[index-1], n_feature_map, scope="conv_layer"+str(index+1))

        ## last max pool
        x = self.max_pool_8x8(x)

        ## flatten layer
        width = self.image_size[0]
        height = self.image_size[1]

        # after first 2x2 pooling
        width = width//2 + width%2
        height = height//2 + height//2
        
        # after second 2x2 pooling
        width = width//2 + width%2
        height = height//2 + height//2

        # ater last 8x8 pooling
        width = width//8 + width%8
        height = height//8 + height//8

        flatten_shape = width*height*self.n_feature_maps[-1]
        x = tf.reshape(x, [-1, flatten_shape])
        x = self.fc_layer(x, 128)
        x = self.fc_layer(x, self.n_classes, use_relu=False) # output without activative function
        
        self.output_logits = x

        return x

    def calc_training_loss(self, sotfen_teacher_outputs = None, ground_truth):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits, labels=ground_truth))
        
        if not(sotfen_teacher_outputs):
            loss += self.alpha*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits/self.temperature, labels=sotfen_teacher_outputs))
        
        self.loss = loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        