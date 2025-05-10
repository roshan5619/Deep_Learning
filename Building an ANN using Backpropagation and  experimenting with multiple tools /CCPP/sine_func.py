import numpy as np
import matplotlib.pyplot as plt

# b1: Generate Training Data with Partitioning into 4 equal parts of each 250 points
np.random.seed(42)
domain = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
x_train = np.concatenate([np.linspace(domain[i * 250], domain[(i + 1) * 250 - 1], 250) for i in range(4)]).reshape(-1, 1)
y_train = np.sin(x_train)

# b2: Generate Validation Data
x_val = np.random.uniform(-2 * np.pi, 2 * np.pi, 300).reshape(-1, 1)
plt.scatter(x_train,y_train,color="red" ,label="Actual Output",alpha=0.6)

#Initialization Of ANN Architecture
input_size = 1
output_size = 1
hidden_size = 50

#Weights and bias initialization
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


# Forward Propagation
def Forwardpass(x):
    vect1 = np.dot(x_train, w1) + b1
    act1 = tanh(vect1)
    vect2 = np.dot(act1, w2) + b2
    output = tanh(vect2)
    return output


#Tanh Activation Function
def tanh(y, deriv=False):
    if(deriv==True):
        return 1-y**2
    return (np.exp(y)-np.exp(-y))/(np.exp(y)+np.exp(-y))


#Training the Neural Network
#Forward and backward propagation
num_epochs=50000
learning_rate=0.001
for i in range(num_epochs):
    vect1 = np.dot(x_train, w1) + b1
    act1 = tanh(vect1)
    vect2 = np.dot(act1, w2) + b2
    output = tanh(vect2)
    Err = y_train-output
    gradient_output_layer = tanh(output, deriv=True)
    gradient_hidden_layer = tanh(act1, deriv=True)
    d_output = Err * gradient_output_layer
    Error_at_hidden_layer = d_output.dot(w2.T)
    d_hiddenlayer = Error_at_hidden_layer * gradient_hidden_layer
    w2 += act1.T.dot(d_output) *learning_rate
    b2 += np.sum(d_output, axis=0,keepdims=True) *learning_rate
    w1 += x_train.T.dot(d_hiddenlayer) *learning_rate
    b1 += np.sum(d_hiddenlayer, axis=0,keepdims=True) *learning_rate


print("Input:\n" + str(x_train))
print("Actual Output:\n" + str(y_train))
print("predicted output:\n" + str(output))
print("Loss: " + str(np.mean(np.square(y_train - output))))


plt.scatter(x_train, y_train, label="Actual output",color="red")
plt.scatter(x_train, output, label= "Predicted output",color="blue")
plt.legend()
plt.title("Actua Output vs Predicted Output")
plt.show()

#Testing the Neural Network
validateoutput=Forwardpass(x_val)
print("Actual Output:\n" + str(y_train))
print("predicted output:\n" + str(validateoutput))
print("Loss: " + str(np.mean(np.square(y_train- validateoutput))))

plt.scatter(x_train, y_train, label="Actual output",color="red")
plt.scatter(x_train, validateoutput, label= "Predicted output",color="green")
plt.legend()
plt.title("Actual Output Vs Predicted Output(ValidationOutput)")
plt.show()

#Plotting the results
plt.scatter(x_train, y_train, label="Actual Output",color="red")
plt.scatter(x_train, output, label= "Training output",color="blue")
plt.scatter(x_train, validateoutput, label="Testing output",color="green")
plt.legend()
plt.show()
