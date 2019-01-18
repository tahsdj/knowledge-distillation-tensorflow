
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

    # 6 conv layers network ( total = 6 + 2 fc layers = 8 layers )

    cnn6_params = [
        [3 for _ in range(6)], # kernal size
        [16,16,32,32,64,64], # feature maps
        [1,3] # inner max pool layer, number starts from 0, 2 means 3rd conv layer
    ]

    cnn12_params = [
        [3 for _ in range(12) ], # kernal size
        [16,16,16,16,32,32,32,32,64,64,64,64], # feature maps
        [1,3] # inner max pool layer, number starts from 0, 2 means 3rd conv layer
    ]
    
    NIN_light_params =  [
            {
                "kernel_size": 3,
                "n_filters": 192,
                "mlp": [
                    {
                        "kernel_size": 1, # should be fixed to 1
                        "n_filters": 160
                    }
                    
                ]
            },
            {
                "kernel_size": 3,
                "n_filters": 192,
                "mlp": [
                    {
                        "kernel_size": 1, # should be fixed to 1
                        "n_filters": 192
                    }
                ]
            },
            {
                "kernel_size": 3,
                "n_filters": 192,
                "mlp": [
                    {
                        "kernel_size": 1, # should be fixed to 1
                        "n_filters": 10
                    }
                ]
            }
        ]


    model_selected = cnn6_params
    # model = student_networks.DeepConvNet(
    #     xs, # inputs
    #     [28,28], # image size
    #     1, # chennel
    #     model_selected[0],
    #     model_selected[1],
    #     model_selected[2],
    #     10, # num of classes
    #     5, # temperature
    #     .9, # weight of student-teacher loss
    #     1e-3, # learning rate
    # )

    model = NiN(
            32,
            3,
            10,
            NIN_light_params
        )


    model = NiN(
        32, # image_size
        3, #channel
        10, # n_classes
        NIN_params # network in network parameters
        )

    model.build_model(xs_image)
    model.calc_training_loss(ys,None)


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

    # train
    model.train(sess,fmnist_data)

    # test
    model.test(sess,fmnist_data)

if __name__ == '__main__':
    main()