import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def plot_scatter(data, x, y, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    plt.scatter(data[x],data[y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(x+'_'+y+'.pdf')
    plt.show()
    
    
def plot_one_relative_scatter(data, metric1, metric2, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    plt.scatter(data[metric1],data[metric2]/data['population'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(metric1+'_'+metric2+'.pdf')
    plt.show()
    

def plot_relative_one_scatter(data, metric1, metric2, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    plt.scatter(data[metric1]/data['population'],data[metric2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(metric1+'_'+metric2+'.pdf')
    plt.show()
    
def plot_both_relative_scatter(data, metric1, metric2, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    plt.scatter(data[metric1]/data['population'],data[metric2]/data['population'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(metric1+'_'+metric2+'.pdf')
    plt.show()
    

    
def plot_fit_scatter(data,metric1,metric2,xlabel,ylabel):
    
    plt.figure(figsize=(10,6))
    
    X = np.array(data[metric1]).reshape(data[metric1].shape[0],1)
    y = np.array(data[metric2]).reshape(data[metric2].shape[0],1)
    
    model = LinearRegression()
    model.fit(X,y)
    # Print the intercept and ang. coefficient values
    print('ang. coef.: ',model.coef_) 
    print('intercept: ',model.intercept_)

    
    prediction = model.predict(X)
    # alternatively
    prediction = model.coef_*X + model.intercept_
    plt.plot(X, y, "b.")  # blue dots (data)
    plt.plot(X, prediction , color="red", markersize = 20, label='fit')  # red line (regression)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid()
    plt.savefig('fit_'+xlabel+'_'+ylabel+'.pdf')
    plt.show()
    
    
def plot_one_relative_fit_scatter(data,metric1,metric2,xlabel,ylabel):
    
    plt.figure(figsize=(10,6))
    
    #data[metric1] = data[metric1]/data['population']
    
    X = np.array(data[metric1]).reshape(data[metric1].shape[0],1)
    y = np.array(data[metric2]/data['population']).reshape(data[metric2].shape[0],1)
    
    model = LinearRegression()
    model.fit(X,y)
    # Print the intercept and ang. coefficient values
    print('ang. coef.: ',model.coef_) 
    print('intercept: ',model.intercept_)

    
    prediction = model.predict(X)
    # alternatively
    prediction = model.coef_*X + model.intercept_
    plt.plot(X, y, "b.")  # blue dots (data)
    plt.plot(X, prediction , color="red", markersize = 20, label='fit')  # red line (regression)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid()
    plt.savefig('fit_'+xlabel+'_'+ylabel+'.pdf')
    plt.show()
    
    
def plot_both_relative_fit_scatter(data,metric1,metric2,xlabel,ylabel):
    
    plt.figure(figsize=(10,6))
    
    #data[metric1] = data[metric1]/data['population']
    
    X = np.array(data[metric1]/data['population']).reshape(data[metric1].shape[0],1)
    y = np.array(data[metric2]/data['population']).reshape(data[metric2].shape[0],1)
    
    model = LinearRegression()
    model.fit(X,y)
    # Print the intercept and ang. coefficient values
    print('ang. coef.: ',linear.coef_) 
    print('intercept: ',linear.intercept_)

    
    prediction = model.predict(X)
    # alternatively
    prediction = model.coef_*X + model.intercept_
    plt.plot(X, y, "b.")  # blue dots (data)
    plt.plot(X, prediction , color="red", markersize = 20, label='fit')  # red line (regression)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid()
    plt.savefig('fit_'+xlabel+'_'+ylabel+'.pdf')
    plt.show()
    
    
def ID_low_events(data,metric,value):
    for i in range(data.shape[0]): # loop over all countries
        if data[metric][i] < value: 
            print(i,data['location'][i],data[metric][i])
            
def ID_high_events(data,metric,value):
    for i in range(data.shape[0]): # loop over all countries
        if data[metric][i] > value: 
            print(i,data['location'][i],data[metric][i])
            
            
                        
def make_low_pd_cut(data, metric1, metric2, value):
    if metric2 is not None:
        new_data = data[data[metric1] < value]
        final_data = new_data[data[metric2] < value]
    else:
        final_data = data[data[metric1] < value]
    return final_data

def make_high_pd_cut(data, metric1, metric2, value):
    if metric2 is not None:
        new_data = data[data[metric1] > value]
        final_data = new_data[data[metric2] > value]
    else:
        final_data = data[data[metric1] > value]
    return final_data
    
    
