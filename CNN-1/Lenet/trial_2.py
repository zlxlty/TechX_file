#coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections

def print_array (arrays):
    for array in arrays:
          print(array)

#将所有的图片重新设置尺寸
w = 64
h = 64
c = 3

#训练数据和测试数据保存地址
train_path = "data/train/"
test_path = "data/test/"
pb_file_path = "trial_2.pb"

#读取图片及其标签函数
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.jpg'):
            print("reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)

#读取训练数据及测试数据
train_data,train_label = read_image(train_path)
test_data,test_label = read_image(test_path)

#打乱训练数据及测试数据
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]



#搭建CNN
x = tf.placeholder(tf.float32,[None,w,h,c],name='input')
y_ = tf.placeholder(tf.int32,[None],name='labels')

def inference(input_tensor,train,regularizer):

    #第一层：卷积层，过滤器的尺寸为5×5，深度为6,不使用全0补充，步长为1。
    #尺寸变化：32×32×1->28×28×6
    with tf.variable_scope('layer1-conv1a'):
        conv1a_weights = tf.get_variable('weight',[3,3,c,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1a_biases = tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        conv1a = tf.nn.conv2d(input_tensor,conv1a_weights,strides=[1,1,1,1],padding='SAME')
        relu1a = tf.nn.relu(tf.nn.bias_add(conv1a,conv1a_biases))
        
    with tf.variable_scope('layer1-conv1b'):
        conv1b_weights = tf.get_variable('weight',[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1b_biases = tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        conv1b = tf.nn.conv2d(relu1a,conv1b_weights,strides=[1,1,1,1],padding='SAME')
        relu1b = tf.nn.relu(tf.nn.bias_add(conv1b,conv1b_biases))
        
    with tf.variable_scope('layer1-conv1c'):
        conv1c_weights = tf.get_variable('weight',[5,5,c,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1c_biases = tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        conv1c = tf.nn.conv2d(input_tensor,conv1c_weights,strides=[1,1,1,1],padding='SAME')
        relu1c = tf.nn.relu(tf.nn.bias_add(conv1c,conv1c_biases))

    output1b = tf.concat(axis=3, values=[input_tensor, relu1b])
    output1c = tf.concat(axis=3, values=[input_tensor, relu1c])
 
    with tf.name_scope('layer2-pool1b'):
        pool1b = tf.nn.max_pool(output1b,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('layer2-pool1c'):
        pool1c = tf.nn.max_pool(output1c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool1 = tf.concat(axis=3, values=[pool1b, pool1c], name='pool1')

 
    with tf.variable_scope('layer3-conv2a'):
        conv2a_weights = tf.get_variable('weight',[3,3,70,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2a_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv2a = tf.nn.conv2d(pool1,conv2a_weights,strides=[1,1,1,1],padding='SAME')
        relu2a = tf.nn.relu(tf.nn.bias_add(conv2a,conv2a_biases))
        
    with tf.variable_scope('layer3-conv2b'):
        conv2b_weights = tf.get_variable('weight',[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2b_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv2b = tf.nn.conv2d(relu2a,conv2b_weights,strides=[1,1,1,1],padding='SAME')
        relu2b = tf.nn.relu(tf.nn.bias_add(conv2b,conv2b_biases))
        
    with tf.variable_scope('layer3-conv2c'):
        conv2c_weights = tf.get_variable('weight',[5,5,70,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2c_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv2c = tf.nn.conv2d(pool1,conv2c_weights,strides=[1,1,1,1],padding='SAME')
        relu2c = tf.nn.relu(tf.nn.bias_add(conv2c,conv2c_biases))

    output2b = tf.concat(axis=3, values=[pool1, relu2b])
    output2c = tf.concat(axis=3, values=[pool1, relu2c])

    #第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    #尺寸变化：10×10×6->5×5×16
    with tf.name_scope('layer4-pool2b'):
        pool2b = tf.nn.max_pool(output2b,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('layer4-pool2c'):
        pool2c = tf.nn.max_pool(output2c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    pool2 = tf.concat(axis=3, values=[pool2b, pool2c], name='pool2')



    with tf.variable_scope('layer5-conv3a'):
        conv3a_weights = tf.get_variable('weight',[3,3,172,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3a_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv3a = tf.nn.conv2d(pool2,conv3a_weights,strides=[1,1,1,1],padding='SAME')
        relu3a = tf.nn.relu(tf.nn.bias_add(conv3a,conv3a_biases))
        
    with tf.variable_scope('layer5-conv3b'):
        conv3b_weights = tf.get_variable('weight',[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3b_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv3b = tf.nn.conv2d(relu3a,conv3b_weights,strides=[1,1,1,1],padding='SAME')
        relu3b = tf.nn.relu(tf.nn.bias_add(conv3b,conv3b_biases))
        
    with tf.variable_scope('layer5-conv3c'):
        conv3c_weights = tf.get_variable('weight',[5,5,172,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3c_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv3c = tf.nn.conv2d(pool2,conv3c_weights,strides=[1,1,1,1],padding='SAME')
        relu3c = tf.nn.relu(tf.nn.bias_add(conv3c,conv3c_biases))
        
   
    output3b = tf.concat(axis=3, values=[pool2, relu3b])
    output3c = tf.concat(axis=3, values=[pool2, relu3c])

    with tf.name_scope('layer6-pool2b'):
        pool3b = tf.nn.max_pool(output3b,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('layer6-pool2c'):
        pool3c = tf.nn.max_pool(output3c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    pool3 = tf.concat(axis=3, values=[pool3b, pool3c], name='pool3')


    with tf.variable_scope('layer7-conv4a'):
        conv4a_weights = tf.get_variable('weight',[3,3,376,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4a_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv4a = tf.nn.conv2d(pool3,conv4a_weights,strides=[1,1,1,1],padding='SAME')
        relu4a = tf.nn.relu(tf.nn.bias_add(conv4a,conv4a_biases))
        
    with tf.variable_scope('layer7-conv4b'):
        conv4b_weights = tf.get_variable('weight',[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4b_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv4b = tf.nn.conv2d(relu4a,conv4b_weights,strides=[1,1,1,1],padding='SAME')
        relu4b = tf.nn.relu(tf.nn.bias_add(conv4b,conv4b_biases))
        
    with tf.variable_scope('layer7-conv4c'):
        conv4c_weights = tf.get_variable('weight',[5,5,376,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4c_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv4c = tf.nn.conv2d(pool3,conv4c_weights,strides=[1,1,1,1],padding='SAME')
        relu4c = tf.nn.relu(tf.nn.bias_add(conv4c,conv4c_biases))

    output4b = tf.concat(axis=3, values=[pool3, relu4b])
    output4c = tf.concat(axis=3, values=[pool3, relu4c])
    #第二层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    #尺寸变化：28×28×6->14×14×6
    with tf.name_scope('layer8-pool4b'):
        pool4b = tf.nn.max_pool(output4b,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('layer8-pool4c'):
        pool4c = tf.nn.max_pool(output4c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool4 = tf.concat(axis=3, values=[pool4b, pool4c], name='pool4')


    with tf.variable_scope('layer8-conv5a'):
        conv5a_weights = tf.get_variable('weight',[3,3,784,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5a_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv5a = tf.nn.conv2d(pool4,conv5a_weights,strides=[1,1,1,1],padding='SAME')
        relu5a = tf.nn.relu(tf.nn.bias_add(conv5a,conv5a_biases))
        
    with tf.variable_scope('layer8-conv5b'):
        conv5b_weights = tf.get_variable('weight',[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5b_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv5b = tf.nn.conv2d(relu5a,conv5b_weights,strides=[1,1,1,1],padding='SAME')
        relu5b = tf.nn.relu(tf.nn.bias_add(conv5b,conv5b_biases))
        
    with tf.variable_scope('layer8-conv5c'):
        conv5c_weights = tf.get_variable('weight',[5,5,784,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5c_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv5c = tf.nn.conv2d(pool4,conv5c_weights,strides=[1,1,1,1],padding='SAME')
        relu5c = tf.nn.relu(tf.nn.bias_add(conv5c,conv5c_biases))

    output5b = tf.concat(axis=3, values=[pool4, relu5b])
    output5c = tf.concat(axis=3, values=[pool4, relu5c])
    #第二层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    #尺寸变化：28×28×6->14×14×6
    with tf.name_scope('layer8-pool5b'):
        pool5b = tf.nn.max_pool(output5b,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.name_scope('layer8-pool5c'):
        pool5c = tf.nn.max_pool(output5c,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    pool5 = tf.concat(axis=3, values=[pool5b, pool5c], name='pool5')


    #将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为5×5×16的矩阵，然而第五层全连接层需要的输入格式
    #为向量，所以我们需要把代表每张图片的尺寸为5×5×16的矩阵拉直成一个长度为5×5×16的向量。
    #举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,5,5,16),拉直为向量，nodes=5×5×16=400,尺寸size变为(64,400)
    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool3,[-1,nodes])

    #第五层：全连接层，nodes=5×5×16=400，400->120的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
    #训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    #这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    #本文最后训练时没有采用dropout，dropout项传入参数设置成了False，因为训练和测试写在了一起没有分离，不过大家可以尝试。
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.8)

    #第六层：全连接层，120->84的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[128,100],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[100],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2,0.8)

    #第七层：全连接层（近似表示），84->10的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    #即得到最后的分类结果。
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight',[100,3],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[3],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases
    return logit

#正则化，交叉熵，平均交叉熵，损失函数，最小化损失函数，预测和实际equal比较，tf.equal函数会得到True或False，
#accuracy首先将tf.equal比较得到的布尔值转为float型，即True转为1.，False转为0，最后求平均值，即一组样本的正确率。
#比如：一组5个样本，tf.equal比较为[True False True False False],转化为float型为[1. 0 1. 0 0],准确率为2./5=40%。
regularizer = tf.contrib.layers.l2_regularizer(0.001)

logit = inference(x,False,regularizer)

y = tf.nn.softmax(logit, name='softmax')

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)

cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

prediction_labels = tf.argmax(y, axis=1, name="output")

correct_prediction = tf.equal(tf.cast(prediction_labels,tf.int32),y_)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#每次获取batch_size个样本进行训练或测试
def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

#创建Session会话
with tf.Session() as sess:
    #初始化所有变量(权值，偏置等)
    sess.run(tf.global_variables_initializer())

    #将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
    #train_num可以设置大一些。
    train_num = 200
    batch_size = 16
    tr_loss = []
    tr_acc = []
    te_loss = []
    te_acc = []
    


    for i in range(train_num):

        train_loss,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in get_batch(train_data,train_label,batch_size):
            _,err,acc = sess.run([train_op,loss,accuracy],feed_dict={x:train_data_batch,y_:train_label_batch})
            train_loss+=err;train_acc+=acc;batch_num+=1
        print("train loss:",train_loss/batch_num)
        tr_loss.append(train_loss/batch_num)
        print("train acc:{:.6}".format(train_acc/batch_num))
        tr_acc.append(train_acc/batch_num)

        test_loss,test_acc,batch_num = 0, 0, 0
        for test_data_batch,test_label_batch in get_batch(test_data,test_label,batch_size):
            err,acc = sess.run([loss,accuracy],feed_dict={x:test_data_batch,y_:test_label_batch})
            test_loss+=err;test_acc+=acc;batch_num+=1
        print("test loss:",test_loss/batch_num)
        te_loss.append(test_loss/batch_num)
        print("test acc:{:.6}".format(test_acc/batch_num))
        te_acc.append(test_acc/batch_num)
        print("\n")
    
    print('tr_loss:')
    print_array(tr_loss)
    print('tr_acc:')
    print_array(tr_acc)
    print('te_loss:')
    print_array(te_loss)
    print('te_acc:')
    print_array(te_acc)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
