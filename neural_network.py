'''
This script contains functions to define a universal TensorFlow neural network for any situation, to make it easier to build an initial model and
tune hyperparameters faster.

Author Name: Kallol Saha
GitHub: https://github.com/FailedMesh
File name: neural_network.py
Functions: (initialize_parameters, forward_propagate, compute_cost, model_train, predict, compute_accuracy)
'''

########### DEPENDENCIES-Do not Change ##########
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
#################################################

########################################################## FUNCTION ################################################################
def initialize_parameters(num_nodes, initializer):
    """
    Initializes the trainable parameters of the model
    
    Inputs:
    num_nodes - a python list containing the number of nodes in each layer of the neural network in order
    initializer - A Keras weight initializer
    
    Outputs:
    params - a python dictionary containing all the parameters of the model, with indexes 'W1', 'b1', 'W2', 'b2', etc.
    cache - a python dictionary storing other values like 'A1', 'Z1'
    """
    
    params = {}
    cache = {}
    
    for i in range(1, len(num_nodes)):
        params['W' + str(i)] = tf.Variable(initializer(shape = (num_nodes[i], num_nodes[i-1]), dtype = 'float64'))
        params['b' + str(i)] = tf.Variable(initializer(shape = (num_nodes[i], 1), dtype = 'float64'))
        
    return params, cache
#####################################################################################################################################


########################################################## FUNCTION #################################################################
def forward_propagate(X, num_layers, params, cache, activations):
    
    
    cache['A' + str(0)] = X
    
    for i in range(1, num_layers + 1):
        
        cache['Z' + str(i)] = tf.math.add(tf.linalg.matmul(params['W' + str(i)], cache['A' + str(i-1)]), params['b' + str(i)])
        activation = activations['L' + str(i)]
        cache['A' + str(i)] = activation(cache['Z' + str(i)])
        
    #Final prediction for Y:
    prediction  = cache['A' + str(i)]
    
    return prediction, cache
#####################################################################################################################################



########################################################## FUNCTION ################################################################
def compute_cost(prediction, Y, loss_function):
    
    losses = loss_function(Y, prediction)
    cost = tf.reduce_mean(losses)
    
    return cost
#####################################################################################################################################



########################################################## FUNCTION ################################################################
def model_train(X, Y, num_layers, num_nodes, learning_rate, num_epochs, initializer, activations, loss_function, optimizer, 
               print_cost = True, print_cost_per_epoch = 100):
    
    params, cache = initialize_parameters(num_nodes, initializer)
    
    trainable_variables =[]
    for i in range(1, num_layers + 1):
        trainable_variables.append(params['W' + str(i)])
        trainable_variables.append(params['b' + str(i)])
        
    cost_record = []
    
    for epoch in range(1, num_epochs + 1):
        
        #print(epoch)
        
        with tf.GradientTape() as tape:
            tape.watch(trainable_variables)
            prediction, cache = forward_propagate(X, num_layers, params, cache, activations)
            cost = compute_cost(prediction, Y, loss_function)
            cost_record.append(cost.numpy())
        
        if epoch % print_cost_per_epoch == 0:
            clear_output(wait = True)
            print("Cost at epoch", epoch, " = ", cost.numpy())
            
        grads = tape.gradient(cost, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
    
    plt.plot(np.squeeze(cost_record))
    plt.show()
        
    return params, cost
#####################################################################################################################################



########################################################## FUNCTION ################################################################
def predict(X, num_layers, params, activations):
    
    cache = {}
    cache['A' + str(0)] = X
    
    for i in range(1, num_layers + 1):
        
        cache['Z' + str(i)] = tf.math.add(tf.linalg.matmul(params['W' + str(i)], cache['A' + str(i-1)]), params['b' + str(i)])
        activation = activations['L' + str(i)]
        cache['A' + str(i)] = activation(cache['Z' + str(i)])
        
    #Final prediction for Y:
    prediction  = cache['A' + str(i)]
    prediction = prediction.numpy() > 0.5
    prediction = prediction.astype(float)
    prediction  = tf.constant(prediction)

    return prediction
#####################################################################################################################################



########################################################## FUNCTION ################################################################
def compute_accuracy(y_pred, y_true):

    pred = y_pred.numpy()
    Y = y_true.numpy()
    error_array = abs(pred - Y)
    error = np.sum(error_array)/error_array.size
    error = round(error, 3)
    
    accuracy = (1 - error)*100

    return accuracy
#####################################################################################################################################

#------------------------------------------------------ END OF FILE ----------------------------------------------------------------#