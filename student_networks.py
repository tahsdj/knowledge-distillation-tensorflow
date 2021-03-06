import tensorflow as tf
import numpy as np


import tensorflow as tf
import numpy as np


class NiN():
    def __init__(
        self,
        image_size,
        n_chennel,
        n_classes,
        hyperparameters,
        ######### hyperparameters example #################
        # [
        #     {
        #         "kernel_size": 5,
        #         "n_filters": 192,
        #         "mlp": [
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 160
        #             },
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 96
        #             }
        #         ]
        #     },
        #     {
        #         "kernel_size": 5,
        #         "n_filters": 192,
        #         "mlp": [
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 192
        #             },
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 192
        #             }
        #         ]
        #     },
        #     {
        #         "kernel_size": 3,
        #         "n_filters": 192,
        #         "mlp": [
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 192
        #             },
        #             {
        #                 "kernel_size": 1, # should be fixed to 1
        #                 "n_filters": 10
        #             }
        #         ]
        #     }
        # ]
        ################################
        temperature = 7,
        alpha = .9,
        learning_rate = 1e-3
    ):
        self.image_size = image_size
        self.n_classes = n_classes
        self.n_chennel = n_chennel
        self.training = True
        self.output_logits = None
        self.alpha = alpha
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.loss = 0
        self.hyperparameters = hyperparameters
        self.dropout = tf.placeholder(tf.float32)
        self.optimizer = None
        
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

    def max_pool_2x2(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def max_pool_4x4(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

    def avg_pool_8x8(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.avg_pool(x, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')

    def conv_layer(self, inputs, kernel_size, n_chennel, n_feature_map, scope):
        with tf.name_scope(scope):
            x = inputs
            print('scope: {:} weights: {:}'.format(scope,[kernel_size, kernel_size, n_chennel, n_feature_map]))
            w = self.weight_variable([kernel_size, kernel_size, n_chennel, n_feature_map])
            b = self.bias_variable([n_feature_map])
            x = self.conv2d(x,w) + b
            return x

    def build_model(self, inputs):
        
        x = None
        n_chennel = 0
        for index, obj in enumerate(self.hyperparameters):
            scope = "conv_layer{:}".format(index+1)
            kernel_size = obj["kernel_size"]
            n_filter = obj["n_filters"]
            if index == 0:
                n_chennel = self.n_chennel
                x = self.conv_layer(inputs, kernel_size, n_chennel, n_filter, scope=scope)
                n_chennel = n_filter
            else:
                x = self.conv_layer(x, kernel_size, n_chennel, n_filter,scope=scope)
                n_chennel = n_filter
            
            x = tf.nn.relu(x)

            # for mlp part
            for i, o in enumerate(obj["mlp"]):
                kernel_size = o["kernel_size"]
                n_filter = o["n_filters"]
                x = self.conv_layer(x,kernel_size,n_chennel,n_filter,scope=scope)
                x = tf.nn.relu(x)
                n_chennel = n_filter

            if index < len(self.hyperparameters) - 1:
                x = self.max_pool_2x2(x)
                x = tf.nn.dropout(x,self.dropout)

        # output from last conv layer
        x = self.avg_pool_8x8(x)
        x = tf.layers.flatten(x)
        
        self.output_logits = x
        
        return self.output_logits
      
    def calc_training_loss(self, ground_truth, sotfen_teacher_outputs = None):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits, labels=ground_truth))
        
        if sotfen_teacher_outputs:
            loss += self.alpha*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits/self.temperature, labels=sotfen_teacher_outputs))
        
        self.loss = loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


     def train(self,sess,dataset,iteration=30000,batch_size=256,batch_size_of_test='all',display_step=100):
        
        if dataset:
            dataset.reset()
        if batch_size_of_test == 'all':
            test_samples = dataset.test_samples
            test_labels = dataset.test_labels
        else:
            test_samples = dataset.test_samples[:batch_size_of_test]
            test_labels = dataset.test_labels[:batch_size_of_test]
            
        for i in range(iteration):
            batch_xs, batch_ys = dataset.next_batch(batch_size)
            feed_dict = {
                xs: batch_xs,
                ys: batch_ys,
                self.dropout: 0.5
            }
            
            _, training_loss, outputs = sess.run([self.optimizer, self.loss, self.output_logits], feed_dict=feed_dict)

            if i % display_step == 0:
                predicts = sess.run([self.output_logits], feed_dict={xs: test_samples, self.dropout: 1})
                correct = np.equal(np.argmax(predicts[0],1),np.argmax(test_labels,1))
                test_acc = np.mean(correct*1)
                correct = np.equal(np.argmax(outputs,1),np.argmax(batch_ys,1))
                training_acc = np.mean(correct*1)
                print('iteration: {:}  epoch: {:}  batch loss: {:.6}   training acc: {:.4}   test acc: {:.4}'.format(i, fmnist_data.epoch, training_loss, training_acc, test_acc))
   
    def test(self,sess,dataset):
        predicts = sess.run([self.output_logits], feed_dict={xs: dataset.test_samples, self.dropout: 1})
        correct = np.equal(np.argmax(predicts[0],1),np.argmax(dataset.test_labels,1))
        test_acc = np.mean(correct*1)
        print('test accuracy: ', test_acc)



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
        self.optimizer = None

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
    
    def build_model(self, inputs):
        
        #build cnn layer
        x = None
        for index, kernel_size, n_feature_map in zip(range(len(self.kernel_sizes)), self.kernel_sizes, self.n_feature_maps):
            scope = "conv_layer{:}".format(index+1)
            if index == 0:
                x = self.conv_layer(inputs, kernel_size, self.n_chennel, n_feature_map, scope=scope)
            else:
                if index in set(self.index_of_pool_layers):
                    x = self.max_pool_2x2(x)
                x = self.conv_layer(x, kernel_size, self.n_feature_maps[index-1], n_feature_map,scope=scope)

        ## add 8x8 max pool after last cnn layer
        x = self.max_pool_8x8(x)

        ## flatten layer
        x = tf.layers.flatten(x)

        ### 2 fully connected layer ####
#         x = self.fc_layer(x, 128, 'fc_layer1')
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, self.n_classes) # output without activative function
        
        self.output_logits = x

        return self.output_logits

    def calc_training_loss(self, ground_truth, sotfen_teacher_outputs = None):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits, labels=ground_truth))
        
        if sotfen_teacher_outputs:
            loss += self.alpha*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_logits/self.temperature, labels=sotfen_teacher_outputs))
        
        self.loss = loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self,sess,dataset,iteration=30000,batch_size=256,display_step=100):
        
        if dataset:
            dataset.reset()

        for i in range(iteration):
            batch_xs, batch_ys = dataset.next_batch(batch_size)
            feed_dict = {
                xs: batch_xs,
                ys: batch_ys
            }
            
            opt, training_loss, outputs = sess.run([self.optimizer, self.loss, self.output_logits], feed_dict=feed_dict)

            if i % display_step == 0:
                predicts = sess.run([self.output_logits], feed_dict={xs: dataset.test_samples})
                correct = np.equal(np.argmax(predicts[0],1),np.argmax(dataset.test_labels,1))
                test_acc = np.mean(correct*1)
                correct = np.equal(np.argmax(outputs,1),np.argmax(batch_ys,1))
                training_acc = np.mean(correct*1)
                print('iteration: {:}  epoch: {:}  batch loss: {:.6}   training acc: {:.4}   test acc: {:.4}'.format(i, fmnist_data.epoch, training_loss, training_acc, test_acc))
                

    def test(self,sess,dataset):
        predicts = sess.run([self.output_logits], feed_dict={xs: dataset.test_samples})
        correct = np.equal(np.argmax(predicts[0],1),np.argmax(dataset.test_labels,1))
        test_acc = np.mean(correct*1)
        print('test accuracy: ', test_acc)