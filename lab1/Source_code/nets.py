import numpy as np

class nn_net_func():
    def __init__(self, lr=0.001, epoch=1000, ground_truth=None, random_seed=0, num_classes=1, hidden_node = 5):
        self.lr = lr
        self.epoch = epoch
        self.random_seed = random_seed
        self.hid_layer_size = hidden_node
        self.ground_truth = ground_truth
        self.output_size = num_classes
        pass
    
    def linear(self, x, w, b):
        return np.dot(w, x)+b

        
    def nn_sigmoid(self, x):
        return (1/(1+np.exp(-x)))


    def nn_ReLU(self, x):
        return np.maximum(0, x)


    def feedforward(self, x, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        z1 = self.linear(x.reshape((2,1)), W1, b1)
        a1 = self.nn_ReLU(z1)
        z2 = self.linear(a1, W2, b2)
        a2 = self.nn_sigmoid(z2)
        z3 = self.linear(a2, W3, b3)
        a3 = self.nn_sigmoid(z3)

        tmp_parameters = {"z1": z1, "z2": z2, "z3": z3, "a1": a2, "a2": a2, "a3": a3}
        return a3, tmp_parameters


    def backpropagation(self, parameters, tmp_parameters, x, y):

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        W3 = parameters["W3"]
        A1 = tmp_parameters["a1"]
        A2 = tmp_parameters["a2"]
        A3 = tmp_parameters["a3"]
        Z1 = tmp_parameters["z1"]
        Z2 = tmp_parameters["z2"]
        Z3 = tmp_parameters["z3"]
        

        dA3 = - (np.divide(y, A3) - np.divide(1 - y, 1 - A3))
        
        temp_s = self.nn_sigmoid(Z3)
        dZ3 = dA3 * temp_s * (1-temp_s)        # Sigmoid (back propagation)
        
        dW3 = 1/2 * np.dot(dZ3, A2.T)
        db3 = 1/2 * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = np.dot(W3.T,dZ3)

        # dA2 = - (np.divide(y, A2) - np.divide(1 - y, 1 - A2))
        
        temp_s = self.nn_sigmoid(dA2)
        dZ2 = dA2 * temp_s * (1-temp_s)        # Sigmoid (back propagation)
        
        dW2 = 1/2 * np.dot(dZ2, A1.T)
        db2 = 1/2 * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T,dZ2)
        
        # ReLU (back propagation)
        dZ1 = np.array(dA1, copy=True) # just converting dz to a correct object.
        dZ1[Z1 <= 0] = 0   # When z <= 0, you should set dz to 0 as well. 
        # dZ1 = self.nn_ReLU(dA1)
        
        dW1 = 1/2 * np.dot(dZ1, x.reshape(1,2))
        db1 = 1/2 * np.sum(dZ1, axis=1, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
        return grads

    
    def cost_func(self, output, y):
        return -(1/2)*( np.sum( (y*np.log(output).T) + ( (1-y)*(np.log(1-output).T) ) ) )


    def update_net(self, parameters, gradient_data):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        dW1 = gradient_data["dW1"]
        db1 = gradient_data["db1"]
        dW2 = gradient_data["dW2"]
        db2 = gradient_data["db2"]
        dW3 = gradient_data["dW3"]
        db3 = gradient_data["db3"]

        W1 = W1 - self.lr*dW1
        b1 = b1 - self.lr*db1
        W2 = W2 - self.lr*dW2
        b2 = b2 - self.lr*db2
        W3 = W3 - self.lr*dW3
        b3 = b3 - self.lr*db3
        
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3" : W3, "b3" : b3}

        return parameters
    
    

    def init_weight(self, input):
        np.random.seed(self.random_seed)
        W1 = np.random.randn(self.hid_layer_size, input.shape[1])
        b1 = np.zeros([self.hid_layer_size, 1])
        W2 = np.random.randn(self.hid_layer_size, self.hid_layer_size)
        b2 = np.zeros([self.hid_layer_size, 1])
        W3 = np.random.randn(self.output_size, self.hid_layer_size)
        b3 = np.zeros([self.output_size, 1])
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3" : b3}
        return parameters


    def train(self, input):
        x_data, y_data = input
        parameters = self.init_weight(input=x_data)
        stop_flag = 0
        loss_list = []
        for i in range(self.epoch):
            if stop_flag:
                break
            loss = 0
            for idx, x in enumerate(x_data):
                y = y_data[idx]
                a3, tmp_parameters = self.feedforward(x=x, parameters=parameters)
                loss = self.cost_func(a3, y)
                grads = self.backpropagation(parameters=parameters, tmp_parameters=tmp_parameters, x=x, y=y)
                parameters = self.update_net(parameters=parameters, gradient_data=grads)
                del y, a3, tmp_parameters, grads
            # if i%50==0:
            print("epoch = "+str(i)+" ,loss = ", str(loss))
            loss_list.append(loss)
            if loss<0.001:
                stop_flag = 1
                break

        return parameters, loss_list


    def test(self, input, parameters):
        x_data, y_data = input
        predict_list = []
        for idx, x in enumerate(x_data):
            predict, tmp = self.feedforward(x, parameters)
            if predict > 0.5:
                predict_list.append(1)
            else:
                predict_list.append(0)
        acc = (np.sum((np.array(predict_list).reshape(y_data.shape) == y_data))/len(y_data))*100
        print("acc = ", acc, "%")
        from utiles import show_result
        show_result(x_data, y_data, predict_list)


