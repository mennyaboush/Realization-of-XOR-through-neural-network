import tensorflow as tf
def neural_network(input_mat, k, higeway,new_w1,new_b1,new_w2,new_b2):#higewayis the w og the bridge or None if there are no higway
    '''variabels'''
    dim = input_mat[0].__len__()
    nb_outputs = 1
    nb_hidden = k
    temp = 0.001
    nb_hbridge = nb_hidden
    if(higeway):
        nb_hbridge = nb_hidden + dim
    x = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.float32, [None, nb_outputs])
    t = tf.placeholder(tf.float32)#for loss function -> maybe I can use y_train instead

    w1 = tf.Variable(tf.zeros([dim, nb_hidden]), name="Weights1")
    w2 = tf.Variable(tf.zeros([nb_hbridge, nb_outputs]), name="Weights2")
    b1 = tf.Variable(tf.zeros([nb_hidden]), name="Biases1")
    b2 = tf.Variable(tf.zeros([nb_outputs]), name="Biases2")
    z1 = tf.matmul(x,w1) + b1  # [4,2]X[2,nb_hidden]  results in a vector [4,nb_hidden] of z’s
    hlayer = tf.sigmoid(z1 / temp)  # element wise
    hlayer1 = hlayer
    if(higeway):
        hlayer1 = tf.concat([hlayer1, x],axis=1)
    print(hlayer1)
    z2 = tf.matmul(hlayer1, w2) + b2
    out = tf.sigmoid(z2 / temp)

    # init the variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #assiagne the new values
    w1 = tf.assign(w1,new_w1)
    b1 = tf.assign(b1,new_b1)
    w2 = tf.assign(w2,new_w2)
    b2 = tf.assign(b2,new_b2)

    # start running graph
    x_train = input_mat#[[0,0],[0,1],[1,0],[1,1]]
    y_train = [[0],[1],[1],[0]]
    squared_deltas = tf.square(out - y_train)
    loss = tf.reduce_sum(squared_deltas)
    #print(sess.run(out, {x:x_train}))
    curr_w1, curr_b1, curr_w2, curr_b2, curr_hlayer, curr_out,curr_loss  = sess.run([w1, b1, w2, b2, hlayer1, out,loss ], {x: x_train, y: y_train})
    #print("\nw1:",curr_w1,"\nb1:", curr_b1,"\nw1:", curr_w2,"\nb2:", curr_b2,"\nhlayer:", curr_hlayer,"\nout:", curr_out,"\nloss:", curr_loss)
    return [curr_out,curr_loss]

result_file = open("result.txt","w")#file editor
input_mat = [[0,0],[0,1],[1,0],[1,1]]

'''the variables below change every time'''
w1 = [[0,-1,1,1],[0,1,-1,1]]
b1 = [0,-0.5,-0.5,-1.5]
w2 = [[0],[1],[1],[0],]
b2 = [-0.5]
''''start run nn'''
tt,loss = neural_network(input_mat.copy(),4,False,w1.copy(),b1.copy(),w2.copy(),b2.copy())
print("\n\nk = 4 no highway - this is tt:",tt,"\nthis is loss",loss)

w1 = [[-1,1],[1,-1]]
b1= [-0.5,-0.5]
w2 = [[1],[1]]
b2 = [-0.5]
tt,loss = neural_network(input_mat.copy(),2,False,w1.copy(),b1.copy(),w2.copy(),b2.copy())
print("\n\nk = 2 no highway - this is tt:",tt,"\nthis is loss",loss)


w1 = [[-1],[1]]
b1= [-0.5]
w2 = [[2],[1],[-1]]
b2 = [-0.5]
tt,loss = neural_network(input_mat.copy(),1,True,w1.copy(),b1.copy(),w2.copy(),b2.copy())
print("\n\nk = 1 and highway - this is tt:",tt,"\nthis is loss",loss)

