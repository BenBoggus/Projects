import pandas as pd
import scipy as sc
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def ProblemOne():
    fields = ['Zero','Three'] #I added column headers to the csv file to make seperating them easier
    Data_D = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/D.csv', skipinitialspace=True, usecols=fields)
    #My pathing doesn't work sometimes so I'm direct pathing it just incase
    X = np.array(Data_D['Zero']).reshape(-1,1)
    Y = np.array(Data_D['Three']).reshape(-1,1)
    model1 = LinearRegression()
    Z = model1.fit(X,Y)
    new_X = np.array([0.3,0.5,0.8]).reshape(-1,1)
    print("Predicted values of Y from new X:\n", model1.predict(new_X))

    plt.scatter(Data_D['Zero'], Data_D['Three'], color="lightcoral")
    plt.title("Predicted Values of Y from X")
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.box(False)
    plt.show()
    
    #Part 2
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)
    cv_scores_train = cross_val_score(model1, x_train,y_train,cv=5, scoring='neg_mean_squared_error')
    cv_scores_train = -cv_scores_train
    print("\nCross-Validation Scores Training:\n", cv_scores_train)
    cv_scores_test = cross_val_score(model1, x_test, y_test, cv=5,scoring='neg_mean_squared_error')
    cv_scores_test = -cv_scores_test
    print("\nCross-Validation Scores Testing:\n", cv_scores_test)

    #Part 3
    #Define the degrees of polynomial regression
    degrees = [2, 3, 4, 5]

    #Dictionary to store polynomial models
    polynomial_models = {}

    #Fit polynomial regression models and report them
    for degree in degrees:
        #Create polynomial features
        poly_features = PolynomialFeatures(degree)
        X_poly_train = poly_features.fit_transform(x_train)
        X_poly_test = poly_features.transform(x_test)

        #Fit polynomial regression model
        polynomial_model = LinearRegression()
        polynomial_model.fit(X_poly_train, y_train)

        #Make predictions
        y_train_pred = polynomial_model.predict(X_poly_train)
        y_test_pred = polynomial_model.predict(X_poly_test)
        
        #Calculate mean squared error
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        #Store the polynomial model along with its train and test MSE
        polynomial_models[degree] = {
            'model': polynomial_model,
            'train_mse': train_mse,
            'test_mse': test_mse
        }

        #Print the polynomial model and its MSE
        print(f"\nPolynomial Regression (Degree {degree}):")
        print("Coefficients:", polynomial_model.coef_)
        print("Intercept:", polynomial_model.intercept_)
        print("Train MSE:", train_mse)
        print("Test MSE:", test_mse)
    

    #Predict y for each x for each degree ChatGPT helped me with this part of the problem until Part 4 of the problem
    for degree, model_info in polynomial_models.items():
        X_range = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 100).reshape(-1, 1)
        X_range_poly = PolynomialFeatures(degree).fit_transform(X_range)
        y_range_pred = model_info['model'].predict(X_range_poly)

        #Visualize the predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(x_train, y_train, color='blue', label='Training data')
        plt.scatter(x_test, y_test, color='red', label='Test data')
        plt.plot(X_range, y_range_pred, color='green', label=f'Polynomial Regression (Degree {degree})')
        plt.title(f'Polynomial Regression (Degree {degree})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    #Values of x for prediction
    x_values = np.array([0.3, 0.5, 0.8]).reshape(-1, 1)

    #Predict y for each x value using each polynomial regression model
    predictions = {}
    for degree, model_info in polynomial_models.items():
        model = model_info['model']
        #Transform x values into polynomial features
        x_values_poly = PolynomialFeatures(degree).fit_transform(x_values)
        #Predict y values
        y_values_pred = model.predict(x_values_poly)
        #Store the predictions
        predictions[degree] = y_values_pred

    #Print predictions
    for degree, y_values_pred in predictions.items():
        print(f"\nPredictions for Polynomial Regression (Degree {degree}):")
        for x, y_pred in zip(x_values.ravel(), y_values_pred):
            print(f"x = {x}, Predicted y = {y_pred}")
    
    #Part 4
    cv_scores_train = cross_val_score(polynomial_model, X_poly_train,y_train_pred,cv=5, scoring='neg_mean_squared_error')
    cv_scores_train = -cv_scores_train
    print("\nCross-Validation Scores Training:\n", cv_scores_train)
    cv_scores_test = cross_val_score(polynomial_model, X_poly_test, y_test_pred, cv=5,scoring='neg_mean_squared_error')
    cv_scores_test = -cv_scores_test
    print("\nCross-Validation Scores Testing:\n", cv_scores_test)
    #Find the degree of the best polynomial model based on test MSE
    best_degree = min(polynomial_models, key=lambda x: polynomial_models[x]['test_mse'])

    #Print the degree of the best polynomial model and its corresponding test MSE
    print(f"\nThe best polynomial model is of degree {best_degree} with a test MSE of {polynomial_models[best_degree]['test_mse']}.")

def ProblemTwo():
    fields = ['Zero','One','Two','Three']
    df = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/D.csv', skipinitialspace=True, usecols=fields)
    #Splitting data set into features (x1,x2,x3) and target (y)
    X = df[['Zero', 'One', 'Two']]
    Y = df[['Three']]
    #Create model and linear fit X and Y
    model = LinearRegression()
    model.fit(X,Y)
    #Coefficients
    print("Intercept: ", model.intercept_)
    print("Coefficients: ", model.coef_)
    model.fit(X.values, Y)
    predictions = model.predict([[0.3,0.4,0.1], [0.5,0.2,0.4], [0.8,0.2,0.7]])
    i = 0
    while i < len(predictions):
        print(f"\nPredicted Value of given numbers {i+1}: \n", predictions[i])
        i+=1
    cv_scores = cross_val_score(model,X,Y, cv=5, scoring='neg_mean_squared_error')
    cv_scores = -cv_scores
    j = 0
    while j < len(cv_scores):
        print(f'\nCross-Validation score {j+1}: \n',cv_scores[j])
        j+=1
    #The best model is 4 with the cross-validation score closest to 0 because the MSE is minimized
    
def ProblemThree():
    field = ['X']
    field2 = ['Y']
    dX = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/x.csv', skipinitialspace=True, usecols = field)
    dY = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/y.csv', skipinitialspace=True, usecols = field2)
    X = dX[["X"]]
    Y = dY[["Y"]]
    model = LinearRegression()
    model.fit(X,Y)
    plt.scatter(dX['X'], dY['Y'], color="lightcoral")
    plt.title("Predicted Values Full of Y from X")
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.box(False)
    plt.show()
    print("Intercept Full: ", model.intercept_)
    print("Coefficients Full: ", model.coef_)

    #Part Two
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=False)
    model.fit(x_train,y_train)
    plt.scatter(x_train, y_train, color="lightcoral")
    plt.title("Predicted Values Training of Y from X")
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.box(False)
    plt.show()
    print("\nIntercept Train: ", model.intercept_)
    print("Coefficients Train: ", model.coef_)
    model.fit(x_test, y_test)
    print("\nIntercept Test: ", model.intercept_)
    print("Coefficient Test: ", model.coef_)
    plt.scatter(x_test, y_test, color="lightcoral")
    plt.title("Predicted Values Test of Y from X")
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.box(False)
    plt.show()

    #Part Three
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    residuals = abs(y_test - y_pred_test)
    print("Residual Vector Test: \n", residuals)
    model.fit(x_test,y_test)
    y_pred_train = model.predict(x_train)
    residuals2 = abs(y_train - y_pred_train)
    print("Residual Vector Train: \n", residuals2)
    residual_norm_test = np.linalg.norm(residuals, axis=1)
    residual_norm_train = np.linalg.norm(residuals2, axis=1)
    print("\nNorm 2 Test: \n", residual_norm_test)
    print("\nNorm 2 Train: \n" , residual_norm_train)

    #Part 4
    poly = PolynomialFeatures(degree=5,include_bias=False)
    X_expand = poly.fit_transform(X)
    print("\nFirst 5 Rows of Expanded X: \n", X_expand[:5])
    x_expand_train, x_expand_test, y_expand_train, y_expand_test = train_test_split(X_expand, Y, train_size = 70)
    model.fit(x_expand_train, y_expand_train)
    y_expand_predict = model.predict(x_expand_test)
    residual_expand_train = abs(y_expand_train - y_expand_predict)
    residual_expand_norm_train = np.linalg.norm(residual_expand_train, axis=1)
    print("\nNorm 2 Train Expand: ", residual_expand_norm_train)

def ProblemEight():
    fields = ['One', 'Two', 'Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve','Thirteen','Fourteen','Fifteen']
    fieldsY = ['One']
    ridge_val = [1,5,10,15,20,25,30]
    dXX = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/XX.csv', skipinitialspace=True, usecols=fields)
    dYY = dXX = pd.read_csv('C:/Users/benbo/OneDrive/Documents/School Papers n Stuff/YY.csv', skipinitialspace=True, usecols=fieldsY)
    xTran = np.transpose(dXX)
    x_train, x_test, y_train, y_test = train_test_split(dXX, dYY, test_size = 0.2)
    model = LinearRegression()
    model.fit(x_train,y_train)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_train, y_train)
    least_sq_pred = model.predict(x_test)
    ridge_pred = ridge_model.predict(x_test)
    print("Coefficient of Least Square: \n", least_sq_pred)
    print("\nCoefficient of Ridge: \n", ridge_pred)
    p1 = np.linalg.inv(np.matmul(xTran,dXX))
    p2 = np.matmul(xTran,dYY)
    beta = np.matmul(p1,p2)
    print("\nLeast Squares Coefficient: " , beta)
    xTranY = np.matmul(xTran, dYY)
    xTranX = np.matmul(xTran,dXX)
    i = 0
    sn = np.array([])
    while i < len(ridge_val):
        snI = (ridge_val[i]/len(dXX))
        snI2 = np.append(sn, snI)
        betaS = np.matmul((np.linalg.inv((xTranX + snI2))), xTranY)
        print(f"\nBeta S Coefficient through Ridge Value {ridge_val[i]}: \n", betaS)
        i+=1
    

    #Part2
    #define x,y
    x1 = dXX[:66]
    y1 = dYY[:66]
    x2 = dXX[33:100]
    y2 = dYY[33:100]
    x3pt1 = dXX[:33]
    x3pt2 = dXX[66:]
    y3pt1 = dYY[:33]
    y3pt2 = dYY[66:]
    x3 = np.concatenate((x3pt1,x3pt2), axis=0)
    y3 = np.concatenate((y3pt1,y3pt2),axis=0)
    #Building Models
    #x1
    x1Tran = np.transpose(x1)
    x1TranY1 = np.matmul(x1Tran,y1)
    x1Tranx1 = np.linalg.inv(np.matmul(x1Tran,x1))
    print("\nLeast Squares x1: \n", np.matmul(x1Tranx1,x1TranY1))
    j = 0
    x1sn = np.array([])
    while j < len(ridge_val):
        x1snI = ridge_val[j]/len(x1)
        x1snI2 = np.append(x1sn,x1snI)
        x1betaS = np.matmul((np.linalg.inv((xTranX + x1snI2))), x1TranY1)
        print(f"\nBeta S through Ridge Value {ridge_val[j]}: \n", x1betaS)
        j+=1
    
    #x2 
    x2Tran = np.transpose(x2)
    x2TranY2 = np.matmul(x2Tran,y2)
    x2Tranx2 = np.matmul(x2Tran,x2)
    print("\nLeast Squares x2: \n", np.matmul(x2Tranx2,x2TranY2))
    k = 0
    x2sn = np.array([])
    while k < len(ridge_val):
        x2snI = ridge_val[k]/len(x2)
        x2snI2 = np.append(x2sn,x2snI)
        x2betaS = np.matmul((np.linalg.inv((x2Tranx2 + x2snI2))), x2TranY2)
        print(f"\nBeta S through Ridge Value {ridge_val[k]}: \n", x1betaS)
        k+=1
    
    #x3
        x3Tran = np.transpose(x3)
        x3TranY3 = np.matmul(x3Tran, y3)
        x3Tranx3 = np.matmul(x3Tran,x3)
        print("\nLeast Squares x3: \n", np.matmul(x3Tranx3,x3TranY3))
        l = 0
        x3sn = np.array([])
        while l < len(ridge_val):
            x3snI = ridge_val[l]/len(x3)
            x3snI2 = np.append(x3sn, x3snI)
            x3betaS = np.matmul((np.linalg.inv((x3Tranx3 + x3snI2))), x3TranY3)
            print(f"\nBeta S through Ridge Value {ridge_val[l]}: \n", x3betaS)
            l+=1

    #Part 4,
    #Ridge Regression Works best, with the value of S = 30 in our ridge value.
    




                

    


    







    




