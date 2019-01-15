
import student_networks
import data_handler


def main():

    # define the dataset
    fmnist_data = data_handler.Data(dataset='fashion-mnist')

    xs = tf.placeholder(tf.float32, [None, 784])   # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    
    # 9 conv layers network ( total = 9 + 2 fc layers = 11 layers )
    cnn9_params = [
        [3 for _ in range(9) ], # kernal size
        [16,16,16,32,32,32,64,64,64], # feature maps
        [2,5] # inner max pool layer, number starts from 0, 2 means 3rd conv layer
    ]

    model_selected = cnn9_params
    model = student_networks.DeepConvNet(
        xs, # inputs
        [28,28], # image size
        1, # chennel
        model_selected[0],
        model_selected[1],
        model_selected[2],
        10, # num of classes
        5, # temperature
        .9, # weight of student-teacher loss
        1e-3, # learning rate
    )

    model.calc_training_loss(None,ys)
    sess = tf.Session()

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    model_saver = tf.train.Saver()

    # train cnn model

    sess.run(init)

    #####################################################################
    # training
    #####################################################################
    iterations = 50000
    batch_size = 256
    display_step = 100

    for i in range(iterations):
        batch_xs, batch_ys = fmnist_data.next_batch(batch_size)

        feed_dict = {
            xs: batch_xs,
            ys: batch_ys
        }
        opt, training_loss, model_outputs = sess.run([model.optimizer, model.loss, model.output_logits], feed_dict=feed_dict)

        if i % display_step == 0:
            predicts = sess.run([model.output_logits], feed_dict={xs: fmnist_data.test_smaples})
            correct = np.equal(np.argmax(predicts,1),np.argmax(fmnist_data.test_labels,1))
            test_acc = np.mean(correct*1)
            correct = np.equal(np.argmax(model_outputs,1),np.argmax(batch_ys,1))
            training_acc = 
            print('iteration: {:} epoch: {:} batch loss: {:.5} training accuracy: {:.2} test accuracy: {:.2}'.format(i, fmnist_data.epoch, training_loss, training_acc, test_acc))
            


if __name__ == '__main__':
    main()