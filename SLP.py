import numpy as np
import matplotlib.pyplot as plot
from pathlib import Path

class SLP(object):
    def __init__(self, data, epoch, k_n = 5, learn_rate = 0.1):
        self.data = data
        self.epoch = epoch
        self.learn_rate = learn_rate        
        self.k_n = k_n
        self.dtheta = np.random.random(4)        
        self.weights = np.random.random(4)
        self.bias = np.random.random()
        self.kfold_split()
        self.graphof_sum_error = []
        self.graphof_accuracy = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_function(self, x):
        if x < 0.5:
            return 0
        else:
            return 1
        
    def up_weights(self, dtheta):
        curr_weights = self.weights.copy()
        for index, _ in enumerate(self.weights):
            self.weights[index] = curr_weights[index] - (self.learn_rate*dtheta[index])

    def up_dtheta(self, feature, sig):
        target = feature[-1]
        for index, _ in enumerate(self.dtheta):
            self.dtheta[index] = 2*(sig-target)*(1-sig)*sig*feature[index]

    def up_bias(self, dbias):
        self.bias = self.bias - (self.learn_rate*dbias)

    def train_data(self, data):
        sum_error = 0
        for feature in data:
            summation = np.dot(feature[:4], self.weights) + self.bias
            sig = self.sigmoid(summation)
            error = pow(abs(feature[-1]-sig),2)
            sum_error += error
            dbias = 2*(sig-feature[-1])*(1-sig)*sig*1
            self.up_bias(dbias)
            self.up_dtheta(feature, sig)
            self.up_weights(self.dtheta)
        self.graphof_sum_error.append([sum_error, self.cur_epoch+(self.cur_k_n/self.k_n)]) 
        print("epoch({0}) - kfold({1})\nSum error : {2}".format(self.cur_epoch,self.cur_k_n,sum_error))


    def test_data(self, data):
        predicted = []
        for feature in data:
            summation = np.dot(feature[:4], self.weights) + self.bias
            sig = self.sigmoid(summation)
            predict = int(self.activation_function(sig))
            if int(predict) == int(feature[-1]):
                predicted.append(bool(True))
            else:
                predicted.append(bool(False))
        accuracy = predicted.count(True)/len(predicted)*100
        self.graphof_accuracy.append([accuracy, self.cur_epoch+(self.cur_k_n/self.k_n)])
        print("Accuracy : {0}".format(accuracy))
            
    def epoch_run(self):
        print("-- Starting epoch {0} --".format(self.cur_epoch))
        for idx in range(0, self.k_n):
            self.cur_k_n = idx
            self.train_data(self.kfold_train_data[idx])
            self.test_data(self.kfold_test_data[idx])
        print("-- Epoch {0} finished --\n".format(self.cur_epoch))

    def kfold_split(self):
        fold = []
        fold_n = int(len(self.data)/self.k_n)
        for idx in range(0, self.k_n):
            fold.append(self.data[idx*fold_n:(idx+1)*fold_n])
        self.kfold_train_data = []
        self.kfold_test_data = []
        for idx in range(0, self.k_n):
            self.kfold_test_data.append(fold[idx])
            train_tmp = []
            for not_idx in [x for x in range(0, self.k_n) if x != idx]:
                train_tmp.extend(fold[not_idx])
            self.kfold_train_data.append(train_tmp)

    def run(self):
        print("Starting SLP\n\tLearning rate (alpha):{0}\n\tEpoch: {1}\n\tK-Fold: {2}\n".format(self.learn_rate,self.epoch,self.k_n+1))
        for idx in range(1, self.epoch+1):
            print("Starting Epoch {0}".format(idx))
            self.cur_epoch = idx
            self.epoch_run()
        #Create graph
        plot_index = [val[1] for val in self.graphof_sum_error]
        plot.title("Summary graph (α = {0})".format(self.learn_rate))
        plot.plot(plot_index,[val[0] for val in self.graphof_sum_error])
        plot.plot(plot_index,[val[0] for val in self.graphof_accuracy])
        plot.gca().legend(('sum_error','accuracy'))
        plot.ylabel("")
        plot.xlabel("Epoch")
        plot.show()
        #end

path = Path(__file__).parents[0]
inputFile = str(path) + "\\Iris.csv"
data = np.genfromtxt(inputFile, skip_header=True, delimiter=',') 

slp = SLP(data=data, epoch=300, k_n=5, learn_rate=0.1)

slp.run()