import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import os

import tensorflow._api.v2.compat.v1 as tf
from tensorflow.keras.layers import Flatten

from sklearn.utils import shuffle


""" Load and Extend Data """


def print_stats(x_tr, y_tr, x_va, y_va, x_te, y_te):
    # Number of training examples
    assert len(y_tr) == x_tr.shape[0]
    n_train = len(y_tr)

    # Number of validation examples
    assert len(y_va) == x_va.shape[0]
    n_validation = len(y_va)

    # Number of testing examples.
    assert len(y_te) == x_te.shape[0]
    n_test = len(y_te)

    # What's the shape of an traffic sign image?
    image_shape = x_tr[0].shape

    # How many unique classes/labels there are in the dataset.
    class_set = np.unique([*y_tr, *y_va, *y_te])

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", len(class_set))

    return class_set


def load_data():
    with open('./traffic-signs-data/train.p', mode='rb') as f:
        train = pickle.load(f)
    with open('./traffic-signs-data/valid.p', mode='rb') as f:
        valid = pickle.load(f)
    with open('./traffic-signs-data/test.p', mode='rb') as f:
        test = pickle.load(f)

    x_tr, y_tr = train['features'], train['labels']
    x_val, y_val = valid['features'], valid['labels']

    x_test, y_test = test['features'], test['labels']

    return x_tr, y_tr, x_val, y_val, x_test, y_test


def blur_img(img):
    side = np.random.randint(1, 4)
    if side % 2 != 1:
        side = side + 1
    return cv2.GaussianBlur(img, (side, side), cv2.BORDER_DEFAULT)


def generate_data(data, labels, file_name):
    small_classes = [0, 6, *range(14, 17), *range(19, 25), *range(26, 35), 36, 37, *range(39, 43)]
    gen_data = list()
    gen_labels = list()
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            generated = pickle.load(f)
            gen_data = generated['features']
            gen_labels = generated['labels']
    else:
        for i in labels:
            temp = cv2.rotate(data[i, :, :, :], cv2.ROTATE_90_CLOCKWISE)
            gen_data.append(temp)
            gen_labels.append(i)
            if i in small_classes:
                temp = cv2.rotate(data[i, :, :, :], cv2.ROTATE_180)
                gen_data.append(temp)
                gen_labels.append(i)
                temp = cv2.rotate(data[i, :, :, :], cv2.ROTATE_90_COUNTERCLOCKWISE)
                gen_data.append(temp)
                gen_labels.append(i)

        gen_data, gen_labels = shuffle(np.array(gen_data), np.array(gen_labels))
        generated = {
            'features': gen_data,
            'labels': gen_labels
        }
        with open(file_name, 'wb') as f:
            pickle.dump(generated, f)

    return gen_data, gen_labels


def extend_dataset(x_tr, y_tr, x_val, y_val):
    train_gen_data_file = './traffic-signs-data/train_gen.p'
    x_train_gen, y_train_gen = generate_data(x_tr, y_tr, train_gen_data_file)

    x_train_extended = np.vstack((x_tr, x_train_gen))
    y_train_extended = np.concatenate((y_tr, y_train_gen))
    x_train_extended, y_train_extended = shuffle(x_train_extended, y_train_extended)

    validation_gen_data_file = './traffic-signs-data/valid_gen.p'
    x_valid_gen, y_valid_gen = generate_data(x_val, y_val, validation_gen_data_file)

    x_valid_extended = np.vstack((x_val, x_valid_gen))
    y_valid_extended = np.concatenate((y_val, y_valid_gen))
    x_valid_extended, y_valid_extended = shuffle(x_valid_extended, y_valid_extended)

    return x_train_extended, y_train_extended, x_valid_extended, y_valid_extended


print('\nInitial Dataset:')
x_train, y_train, x_valid, y_valid, test_data, test_labels = load_data()
classes_set = print_stats(x_train, y_train, x_valid, y_valid, test_data, test_labels)
out_size = len(classes_set)

# print('\nExtended Dataset:')
# x_train_ext, y_train_ext, x_valid_ext, y_valid_ext = extend_dataset(x_train, y_train, x_valid, y_valid)
# print_stats(x_train_ext, y_train_ext, x_valid_ext, y_valid_ext, test_data, test_labels)


""" Visualize Data """


def visualize_dataset(tr_labels, te_labels, va_labels, classes):
    # Visualizations will be shown in the notebook.
    output_size = len(classes)
    with open('signnames.csv') as names_file:
        data = dict(enumerate(csv.DictReader(names_file)))
        names_list = [data[i]['SignName'] for i in classes]

    plt.style.use('seaborn-deep')
    plt.hist([tr_labels, va_labels, te_labels], bins=range(output_size + 1), density=True, rwidth=0.8,
             align='left', label=['train', 'validation', 'test'])
    plt.legend(loc='upper right')
    axes = plt.gca()
    plt.xticks(range(output_size), names_list, fontsize=6, rotation=45, ha='right')
    plt.yticks(ticks=axes.get_yticks(), labels=np.round(axes.get_yticks()*100))
    plt.ylabel('%', rotation=0)
    plt.tight_layout()
    plt.show()


visualize_dataset(y_train, test_labels, y_valid, classes_set)
# visualize_dataset(y_train_ext, test_labels, y_valid_ext, classes_set)

""" Preprocess Data """


def preprocess(filename=None, x_extended=None, y_extended=None, color=True):
    """
    Convert a dataset to zero mean and equal variance, and produce color or gray images.
    For color images the output image is 3D, while for gray images 2D
    :param x_extended:
    :param y_extended:
    :param color:
    :param filename:
    :return:
    """
    x_ext_norm = list()
    normalized_data = None
    if filename:
        normalized_data = './traffic-signs-data/' + filename  # x_train_norm.p; x_train_norm_gray.p
    if normalized_data and os.path.isfile(normalized_data):
        with open(normalized_data, 'rb') as file:
            x_ext_norm, y_extended = pickle.load(file)
    else:
        if color:
            for image in x_extended:
                temp = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
                x_ext_norm.append(temp)
            x_ext_norm = np.array(x_ext_norm)
            l, h, w, d = x_ext_norm.shape
            temp = x_ext_norm.ravel()
            temp = (temp - 128.) / 128
            x_ext_norm = temp.reshape(l, h, w, d)
                # mean, std = cv2.meanStdDev(image)
                # temp = np.zeros_like(image, dtype=np.float)
                # temp[:, :, 0] = (image[:, :, 0] - mean[0])/std[0]
                # temp[:, :, 1] = (image[:, :, 1] - mean[1])/std[1]
                # temp[:, :, 2] = (image[:, :, 2] - mean[2])/std[2]
                # x_ext_norm.append(temp)  # Zero mean and equal variance
            x_ext_norm = np.array(x_ext_norm)
            if normalized_data:
                with open(normalized_data, 'wb') as file:
                    x_ext_norm, y_extended = shuffle(x_ext_norm, y_extended)
                    pickle.dump([x_ext_norm, y_extended], file)
        else:
            x_train_ext_g = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in x_extended])
            # mean, std = cv2.meanStdDev(x_train_ext_g)
            # for image in x_train_ext_g:
            #     x_ext_norm.append((image[:, :] - mean) / std)
            l, h, w = x_train_ext_g.shape
            temp = x_train_ext_g.ravel()
            temp = (temp - 128.) / 128
            x_ext_norm = temp.reshape(l, h, w)
            x_ext_norm = np.array(x_ext_norm)
            x_ext_norm = x_ext_norm[..., np.newaxis]
            if normalized_data:
                with open(normalized_data, 'wb') as f:
                    x_ext_norm, y_extended = shuffle(x_ext_norm, y_extended)
                    pickle.dump([x_ext_norm, y_extended], f)

    return shuffle(x_ext_norm, y_extended)


""" Hyper-parameters """

EPOCHS = 6
BATCH_SIZE = 128  # 178

start_learning_rate = 0.0002
lambda_reg = 0.01

""" Net implementation """


def my_net(x_in, keep):
    initializer = tf.initializers.he_normal()

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28
    filter1 = 5
    out_depth1 = 64
    conv1_w = tf.Variable(initializer(shape=(filter1, filter1, 1, out_depth1)))
    conv1_b = tf.Variable(tf.zeros(out_depth1))  # bias
    conv1 = tf.nn.conv2d(x_in, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)  # Activation.
    # mean1, var1 = tf.nn.moments(conv1, axes=[0, 1, 2], keep_dims=False)
    # conv1 = tf.nn.batch_normalization(conv1, mean1, var1, None, None, var_epsilon)

    # Layer 2: Convolutional. Output = 24
    filter2 = 5
    out_depth2 = 64
    conv2_w = tf.Variable(initializer(shape=(filter2, filter2, out_depth1, out_depth2)))
    conv2_b = tf.Variable(tf.zeros(out_depth2))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)  # Activation.
    conv2 = tf.nn.dropout(conv2, keep_prob=keep+0.4)

    # Layer 2: Convolutional. Output = 22
    filter3 = 3
    out_depth3 = 64
    conv3_w = tf.Variable(initializer(shape=(filter3, filter3, out_depth2, out_depth3)))
    conv3_b = tf.Variable(tf.zeros(out_depth3))
    conv3 = tf.nn.conv2d(conv2, conv3_w, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)  # Activation.
    conv3 = tf.nn.dropout(conv3, keep_prob=keep+0.2)

    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = Flatten()(conv3)

    # Layer 4: Fully Connected. Input = 7744
    out_fc1 = 1290
    fc1_w = tf.Variable(initializer(shape=(7744, out_fc1)))
    fc1_b = tf.Variable(tf.zeros(out_fc1))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)  # Activation.
    fc1 = tf.nn.dropout(fc1, keep_prob=keep+0.1)

    # Layer 5: Fully Connected. Input = 1290
    out_fc2 = 258   #258
    fc2_w = tf.Variable(initializer(shape=(out_fc1, out_fc2)))
    fc2_b = tf.Variable(tf.zeros(out_fc2))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)  # Activation.
    fc2 = tf.nn.dropout(fc2, keep_prob=keep+0.1)

    # Layer 6: Fully Connected. Input = 258
    fc3_w = tf.Variable(initializer(shape=(out_fc2, out_size)))
    fc3_b = tf.Variable(tf.zeros(out_size))
    logits_loc = tf.matmul(fc2, fc3_w) + fc3_b

    l2_regular = (tf.nn.l2_loss(conv1_w) + tf.nn.l2_loss(conv2_w) + tf.nn.l2_loss(conv3_w)
                  + tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc3_w))

    return logits_loc, l2_regular


""" Training Pipeline """

tf.disable_eager_execution()
tf.set_random_seed(12345)  # set random seed for dropout

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, None)
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, out_size)
logits, regularize = my_net(x, keep_prob)

global_step = tf.Variable(0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy + lambda_reg * regularize)

learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.8, staircase=True)
training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_operation, global_step=global_step)  # uses backpropagation

""" Model Evaluation """

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))  # compare logit to one-hot
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess_loc = tf.get_default_session()
    for offs in range(0, num_examples, BATCH_SIZE):  # batch the dataset
        batch_xx, batch_yy = x_data[offs:offs+BATCH_SIZE], y_data[offs:offs+BATCH_SIZE]
        accuracy = sess_loc.run(accuracy_operation, feed_dict={x: batch_xx, y: batch_yy, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_xx))
    return total_accuracy / num_examples


saver = tf.train.Saver()

""" Training Process """


def train_process(train_data, train_labels, valid_data, valid_labels):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examp = len(train_data)

        print("Training...")
        print()
        for i in range(EPOCHS):
            train_data, train_labels = shuffle(train_data, train_labels)  # important for radnomizing
            for offset in range(0, num_examp, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = train_data[offset:end], train_labels[offset:end]
                batch_x, batch_y = shuffle(batch_x, batch_y)
                _, loss = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                if offset % 600 == 0:
                    print("Sample:{:5};".format(offset) + " Loss:{:.4f} ".format(loss))

            validation_accuracy = evaluate(valid_data, valid_labels)
            print("EPOCH {} ...".format(i+1) + " Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
        saver.save(sess, './mynet')
        print("Model saved")


""" Test Process """


def test_process(tst_data, tst_labels, kind):
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(tst_data, tst_labels)
        print(kind + " Accuracy = {:.3f}".format(test_accuracy))


""" Demo """


def read_demo_images(plot=False):
    demo_folder = "demo/demo_img"
    demo_images = []
    categories = []
    num = 0
    for filename in os.listdir(demo_folder):
        img = cv2.imread(os.path.join(demo_folder, filename))
        if img is not None:
            img = cv2.resize(img, (32, 32))
            demo_images.append(img)
            cat = filename.split('_')
            cat = cat[-1].split('.')[0]
            categories.append(cat)
            if plot:
                num = num+1
                plt.subplot(2, 4, num)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.yticks([])
                plt.ylabel(cat, rotation=0)
    if plot:
        plt.show()
    return demo_images, categories


def top_guesses(images, labels):
    softmax_out = tf.nn.softmax(logits)
    top_k_tensor = tf.nn.top_k(softmax_out, k=5)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        guesses = []
        result = 0
        for img, cat in zip(images, labels):
            sess.run(softmax_out, feed_dict={x: img[np.newaxis, ...], keep_prob: 1.0})
            top_k_data = sess.run(top_k_tensor, feed_dict={x: img[np.newaxis, ...], keep_prob: 1.0})
            values, indices = top_k_data
            guesses.append([cat, indices])
            if int(cat) == indices[0][0]:
                result += 1
            with open(os.path.join("demo", 'image_stats.txt'), 'a') as f:
                np.set_printoptions(precision=3, floatmode='fixed')
                f.write('\ncategory: ' + cat + f' prediction: {indices}\n' + f"stats: {values*100}%\n")

    print('demo accuracy = {:.3f}%'.format(100 * result / len(labels)))
    return guesses


def output_feature_map(session_in, image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    """
    :param session_in: the related session
    :param image_input: the test image being fed into the network to produce the feature maps.
        Make sure to preprocess your image_input in a way your network expects with size, normalization, ect if needed
    :param tf_activation: a tf variable name used during your training procedure representing the calculated state
        of a specific weight layer
        An error 'tf_activation is not defined' may denote a trouble accessing the variable from inside a function
        Note: to get access to tf_activation, the session should be interactive; can be achieved with the commands:
            sess = tf.InteractiveSession()
            sess.as_default()
        Note: x should be the same name as your network's tensorflow data placeholder variable
    :param activation_min: can be used to view the activation contrast in more detail
    :param activation_max: can be used to view the activation contrast in more detail,
        by default matplot sets min and max to the actual min and max values of the output
    :param plt_num: plot out multiple different weight feature map sets on the same block,
        just extend the plt number for each new feature map entry
    """
    activation = tf_activation.eval(session=session_in, feed_dict={x: image_input})
    feature_maps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for feature_map in range(feature_maps):
        plt.subplot(6, 8, feature_map+1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(feature_map))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", cmap="gray")
        plt.show()


""" Main """
color_i = False
x_train_prep, y_train_prep = preprocess('train_prep.p', x_train, y_train, color=color_i)
x_valid_prep, y_valid_prep = preprocess('valid_prep.p', x_valid, y_valid, color=color_i)
train_process(x_train_prep, y_train_prep, x_valid_prep, y_valid_prep)
test_process(x_train_prep, y_train_prep, "Train")

x_test_prep, y_test_prep = preprocess('test_prep.p', test_data, test_labels, color=color_i)
test_process(x_test_prep, y_test_prep, "Test")

imgs, cats = read_demo_images()
x_random_n, y_random_n = preprocess(x_extended=imgs, y_extended=cats, color=color_i)
test_process(x_random_n, y_random_n, "Random Data")
guess = top_guesses(x_random_n, y_random_n)

# with tf.InteractiveSession() as i_sess:
#     i_sess.as_default()
#     saver.restore(i_sess, tf.train.latest_checkpoint('.'))
#     # i_sess.run(tf.global_variables_initializer())
#     i_sess.run(tf.initialize_all_variables())
#     logits, _, c1, c2, c3, c4 = my_net(x, keep_prob)
#     cross_entropy = tf.nn.softmax(logits)
#     in_img = x_train[5, :, :, :]
#     i_sess.run(cross_entropy, feed_dict={x: in_img[np.newaxis, ...], keep_prob: 1.0})
#     output_feature_map(i_sess, in_img[np.newaxis, ...], c1)
