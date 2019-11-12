import numpy as np
from argparse import ArgumentParser


    

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x, y, y_pred):
    import matplotlib.pyplot as plt
    num = 0
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1], 'ro')
        else:
            plt.plot(x[i][0],x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if abs(y_pred[i] - y[i]) < 0.1:
            num+=1
        if y_pred[i] < 0.5:
            plt.plot(x[i][0],x[i][1], 'ro')
        else:
            plt.plot(x[i][0],x[i][1], 'bo')
 
    plt.show()
    np.set_printoptions(suppress=True)
    print(y_pred)
    print("accurancy : {}".format(num / x.shape[0] * 100))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
def loss_function(pred_y,y):
    return ((pred_y - y) ** 2) / 2.0
def derivative_loss_function(pred_y,y):
    return  (pred_y - y) * 1.0
def initialize_weight(s):
    return np.random.normal(loc = 0.0 ,scale = 0.5,size = s)
def clear_weight(s):
    return np.ndarray(shape=s, dtype=float, order='F')
def initialize_bias(s):
    return np.random.normal(loc = 0.0 ,scale = 0.5,size = s)

def forward(x):
    z[0] = sigmoid(np.dot(w[0].T,x) + b[0])
    z[1] = sigmoid(np.dot(w[1].T,z[0]) + b[1])
    return sigmoid(np.dot(w[2].T,z[1]) + b[2])

def backward(x,y_pred,y):
    g = [0] * 3
    bg = [0] * 3
 
    loss_sigmoid = np.multiply(derivative_loss_function(y_pred,y),derivative_sigmoid(y_pred))
    
    g[2] = np.dot(loss_sigmoid,np.matrix(z[1]))
    bg[2] = np.sum(loss_sigmoid)
    
    loss_sigmoid = np.multiply(np.dot(w[2],loss_sigmoid),derivative_sigmoid(z[1]))

    g[1] = np.dot(loss_sigmoid.T,z[0])
    bg[1] = np.sum(loss_sigmoid)

    loss_sigmoid = np.multiply(np.dot(w[1],loss_sigmoid),derivative_sigmoid(z[0])) 
  
    g[0] = np.dot(np.matrix(loss_sigmoid).T,np.matrix(x))
    bg[0] = np.sum(loss_sigmoid)

    for i in range(num_h+1):
        w_g[i] += g[i].T
        b_g[i] += bg[i]
        

def update():
    for i in range(num_h+1):
        w[i] -= learning_rate * w_g[i]
        b[i] -= learning_rate * b_g[i]
    
    w_g[0] = clear_weight((d_in,d_h))
    w_g[1] = clear_weight((d_h,d_h))
    w_g[2] = clear_weight((d_h,d_out)) 

    b_g[0] = 0
    b_g[1] = 0
    b_g[2] = 0


d_in, d_out, d_h = 2, 1, 30
num_h = 2
w, b, z, w_g, b_g= [], [], [], [], [] 
learning_rate = 0.1
x,y = generate_linear()
x_,y_ = generate_XOR_easy()

def main():
    data_type = 0

    parser = ArgumentParser()
    parser.add_argument("-d", dest="data_type" ,type=int)
   
    args = parser.parse_args()
  
    if args.data_type == 0:
        data_x = x
        data_y = y
    else:
        data_x = x_
        data_y = y_
    #data_x = np.concatenate((x,x_),axis = 0)
    #data_y = np.concatenate((y,y_),axis = 0)

    
    w.append(initialize_weight((d_in,d_h))) 
    w.append(initialize_weight((d_h,d_h))) 
    w.append(initialize_weight((d_h,d_out))) 
  
    w_g.append(clear_weight((d_in,d_h))) 
    w_g.append(clear_weight((d_h,d_h))) 
    w_g.append(clear_weight((d_h,d_out))) 


    for i in range(num_h+1):
        b.append(initialize_bias(1)) 
        b_g.append(0) 
    for i in range(num_h):
        z.append(0)

    epoch = 100000
    for i in range(epoch):
        total_loss = 0
        for idx in range(len(data_x)): 
            y_pred = forward(data_x[idx])
            loss = loss_function(y_pred,data_y[idx])
            total_loss += loss
            backward(data_x[idx],y_pred,data_y[idx])
        if (i+1) % 5000 == 0:
            print("epoch : {} ,average loss : {}".format(i+1,total_loss))
        if total_loss/len(data_x) < 0.0001:
            break
        update()
       

    y_pred = []
    for i in range(len(data_x)): 
        y_pred.append(forward(data_x[i]))
    show_result(data_x, data_y, np.array(y_pred))
      
     


if __name__ == '__main__':
    main()