import numpy as np
import pandas as pd
from pathlib import Path
import pickle as p

class NeuralNetwork(object):  #Neural Network class is created  which is comptaible with increasing number of layers
    
    def __init__(self, conf):
        self.config=conf
        self.num_layers=len(conf)
        self.baises=[np.random.rand(y,1) for y in self.config[1:]]
        self.weights=[np.random.rand(y,x) for x,y in zip(self.config[:-1],self.config[1:])]
    
    def predicted_output(self,a):
        for w,b in zip(self.weights,self.baises):
            a=self.sigmoid_calc(w.dot(a)+b)
        return a
    def sigmoid_calc(self,inp):   #activation function
        return 1.0/(1.0+np.exp(-inp))
    
    def cross_entropy_cost_derivative(self,predicted_op,actual_op):
        return predicted_op-actual_op
    
    def prediction_result(self,data,actual_op):
        op_predicted=np.argmax(self.predicted_output(data))
        if(op_predicted==actual_op):
            return 1
        else:
            return 0
    def predicting(self,full_data):
        total_predictions=0
        for data in full_data:
            total_predictions+=self.prediction_result(data[0:85].reshape(85,1),data[85])
        return total_predictions/len(full_data)
        
    def gradient_descent(self,train_data,no_of_times,train_data_size,learning_rate,lamda): #gradient descent function
        data_size=len(train_data)
        first=True
        featured_inputs=[]
        for i in range(no_of_times):
            #random.shuffle(train_data)
            for j in range(0,train_data_size):
                temp_baises,temp_weights=self.back_prop(train_data[j,0:85].reshape(85,1),train_data[j,85])
                self.weights=[(1-learning_rate*(lamda/train_data_size))*w-(learning_rate)*t_w for w, t_w in zip(self.weights, temp_weights)]
                self.baises = [b-(learning_rate)*nb for b, nb in zip(self.baises, temp_baises)]
            print("Completed ",i,"  accuracy:--",self.predicting(train_data))
            
        #for i in range(0,no_of_times):
           # random.shuffle(training_data)
    def back_prop(self,data,output):  #back propogation algorithm
        op=np.zeros(10)
        op[output]=1
        temp_weights=[np.zeros(w.shape) for w in self.weights]
        temp_baises=[np.zeros(b.shape) for b in self.baises]
        activation=data
        activations=[data]       
        calc_weights=[]
        for b,w in zip(self.baises,self.weights):
            cal=w.dot(activation)+b
            calc_weights.append(cal)           #storing weights for each layer so as to be used later for calculating errors
            activation=self.sigmoid_calc(cal)   
            activations.append(activation)    #storing all the activations in a list
        #print(activations,calc_weights)
        #print(max_index,max_val,activations[-1],"\n")
        error=self.cross_entropy_cost_derivative(activations[-1],op.reshape(10,1))   #calculating cost derivative of last layer
        #print(op_achieved,op,output)                                                #
        temp_weights[-1]=np.dot(error,activations[-2].transpose())
        temp_baises[-1]=error
        for l in range(2,self.num_layers):
            cal=calc_weights[-l]
            sigma_der=self.sigmoidPrime(cal)
            error=np.dot(self.weights[-l+1].transpose(),error)*(sigma_der)  #calculating back propogation error
            temp_weights[-l]=np.dot(error,activations[-l-1].transpose())    #calculating weights and baises after backpropogation
            temp_baises[-l]=error
            #print(temp_weights)
        return temp_baises,temp_weights

    def cross_entropy_cost(self,pred_opt,opt):   #cross entropy cost function
    	return np.sum(-opt*np.log(pred_opt)-(1-opt)*np.log(1-pred_opt))

    def cal_Cost(self,data,lamda):  # calculating total cost after each epoches
    	total_cost=0.0
    	for d in data:
    		pred_op=self.predicted_output(d[0:85].reshape(85,1))
    		op=np.zeros(10)
    		op[d[85]]=1
    		total_cost+=self.cross_entropy_cost(pred_op,op)/len(data)
    	total_cost+=(lamda/2*len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
    	return total_cost



    def sigmoidPrime(self,x):   #sigmoid derivative used in backpropogating error to layers
        return self.sigmoid_calc(x)*(1-self.sigmoid_calc(x))


def get_encoded_data():   # function to convert an input in one-hot encoding
    data=pd.read_csv('train.csv')
    data=data.values
    file=open("featured_data.csv","a")
    featured_data=[]
    first=True
    print('Encoded file not present')
    for single_data in data:
        first=True
        for length in range(0,len(single_data[0:10]),2):
            a=np.zeros(13)
            b=np.zeros(4)
            a[single_data[length+1]-1]=1
            b[single_data[length]-1]=1
            c=np.concatenate((b,a))
            if first:
                d=c
                first=False
            else:
                d=np.concatenate((d,c))
        op=np.zeros(1)
        op[0]=single_data[10]
        d=np.concatenate((d,op))
        d.tofile(file,sep=",",format="%d")
        file.write("\n")
    return pd.read_csv('featured_data.csv')



def k_folded_train_data(data):   # function to create a set of five equal datas used for training and tesing 
	train_test_data=[]
	for i in range(0,len(data),160000):
		train_test_data.append(data[i:i+159999])
	return train_test_data




if __name__ == "__main__":
     encoded_data_file = Path("featured_data.csv")  #for checking if the encoded file is present or not
     if encoded_data_file.is_file():
    	 fdata=pd.read_csv('featured_data.csv')  # if present then loading encoded file using pandas.csv
    	 fdata=fdata.values
    	 print('encoded_file_present')
     else:
     	 #print('Encoded file not present')
    	 fdata=get_encoded_data()  # if not present then making an encoding file to be used in our model training
    	 fdata=fdata.values

     five_folded_data=k_folded_train_data(fdata)  #getting five folded data for training
     print('training neural networks')
     x=NeuralNetwork([85,100,20,10])    #creating object of neural network with 2 hidden layers and one input and output layer
     first=True
     for i in range(0,len(five_folded_data)):
     	 first=True
     	 for j in range(0,len(five_folded_data)):
     	 	 if i!=j:
     	 	 	 if first==True:
     	 	 		 train_data=five_folded_data[j]
     	 	 		 first=False
     	 	 	 else:
     	 	 	 	 train_data=np.append(train_data,five_folded_data[j],axis=0)  #combining all training data into one set to train our modeland excluding one test data
     	 x.gradient_descent(train_data,20,len(train_data),0.1,1) #calling gradient descent for every set of train data
     	 print(x.predicting(five_folded_data[i])) #prediction on test data


     #x.gradient_descent(fdata,50,len(fdata),0.0001,1)   # when training done on full data
     print(x.predicting(fdata))                          # calculating accuracy for entire train data
     p.dump(x.weights, open("weight_file.p", "wb"))		#storing best weight and baises in a pickle file
     p.dump(x.baises, open("baises_file.p", "wb"))
     test_data=pd.read_csv('test.csv')  #loading test data
     test_data=test_data.values
     pred=[]
     for single_data in test_data:                       #converting test data in encoded format to feed to our model for prediction
         first=True
         for length in range(0,len(single_data[0:10]),2):
             a=np.zeros(13)
             b=np.zeros(4)
             a[single_data[length+1]-1]=1
             b[single_data[length]-1]=1
             c=np.concatenate((b,a))
             if first:
                 d=c
                 first=False
             else:                	
                 d=np.concatenate((d,c))
         pred.append(np.argmax(x.predicted_output(d.reshape(85,1))))  #predicting for particular row in tet data
     pred_op=np.array(pred)                        #savinf file after prediction on all the data elements of test data
     pred_op=pd.DataFrame(pred_op)  
     pred_op.index = np.arange(0, len(pred_op))
     headers=['predicted_class']
     pred_op.index.name='id'
     pred_op.columns=headers
     pred_op.to_csv("output.csv", index=True, encoding='utf-8', header=True)


