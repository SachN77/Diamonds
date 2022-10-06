from multiprocessing.util import LOGGER_NAME
from symbol import parameters
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import math
import sklearn
import keras


class Regressor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit(self, model, type = ''):
        if type == 'A':
            model.fit(self.X_train, self.y_train, batch_size = 20, epochs=20, validation_data= (self.X_test, self.y_test))
        else:
            trained_model = model.fit(self.X_train, self.y_train)
        return
    
    def predict(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred

    def score(self, y_pred):
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
       
        r2score = r2_score(self.y_test, y_pred)
        mse_score = mean_squared_error(self.y_test, y_pred)
        mae_score = mean_absolute_error(self.y_test, y_pred)
        rmse_score = math.sqrt(mse_score)

        return r2score , mse_score, mae_score, rmse_score

    def calculateRMSE(self, predictions):
        SSE = 0
        SSR = 0
        y_test_mean = self.y_test.mean() # Compare to the assumption of simplest relationship (linear)
        length = len(self.y_test)
        
        for i in range(0,length):
            SSE = SSE + np.square(predictions[i] - self.y_test.iloc[i]) # Compares preditction to actual
            SSR = SSR + np.square(predictions[i] - y_test_mean) # Compares prediction to simplest model

        SST= SSR + SSE # Total sum of squares represents total variability of y_test
        r2 = np.float64(SSR/SST) # Calculate the coefficient of determination (R-Squared value)
        print(r2)
        
        return r2

    # Defining Linear Regression function to create a Linear Regression estimator.
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.
    def linearRegression(self):
        scores_list = []
        from sklearn.linear_model import LinearRegression
        lr_model = LinearRegression()
        self.fit(lr_model)
        lr_y_pred = self.predict(lr_model)
        lr_r2score , lr_mse_score, lr_mae_score, lr_rmse_score= self.score(lr_y_pred)
        scores_list.extend([lr_r2score, lr_mse_score, lr_mae_score, lr_rmse_score])
        print('Linear Regression Scores:')
        print(f'r2 score {scores_list[0]}, The MSE score {scores_list[1]}, MAE score: {scores_list[2]} and RMSE score: {scores_list[3]}')
        return
        #return lr_r2score, lr_y_pred, scores_list

    # Defining KNN Regression function to create a KNN Regression estimator.
    # Function loops through the neighbors list using different n_neighbors
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.    
    def knnRegression(self):
        from sklearn.neighbors import KNeighborsRegressor
        KNN_list = []
        score_list = []
        high_score = 0
        best_K = 0
        best_knn_mse = 0
        best_knn_mae = 0
        best_knn_rmse = 0
        neighbors = [1, 2, 3, 4, 5, 6, 7]
        for i in neighbors:
            KNN_model = KNeighborsRegressor(n_neighbors=i)
            self.fit(KNN_model)
            KNN_y_pred = self.predict(KNN_model)
            KNN_r2_score, KNN_mse, KNN_mae, KNN_rmse= self.score(KNN_y_pred)
            KNN_list.append(KNN_r2_score)
            if KNN_r2_score > high_score:
                high_score = KNN_r2_score
                best_knn_mse = KNN_mse
                best_knn_mae = KNN_mae
                best_knn_rmse = KNN_rmse
                best_K = neighbors.index(i)
        KNN_r2_score = high_score
        KNN_mse = best_knn_mse
        KNN_mse = best_knn_mae
        KNN_mse = best_knn_rmse

        print(f'The Optimal K is: {neighbors[best_K]}')
        score_list.extend([KNN_r2_score, KNN_mse, KNN_mae, KNN_rmse])
        print('KNN Regression Scores:')
        print(f'r2 score {score_list[0]}, The MSE score {score_list[1]}, MAE score: {score_list[2]} and RMSE score: {score_list[3]}')
        return
        #return KNN_r2_score, KNN_y_pred, KNN_list, score_list

    # Defining Decision Tree Regression function to create a Decision Tree estimator.
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.
    def decision_tree_regression(self):
        from sklearn.tree import DecisionTreeRegressor
        criterion_list = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        dt_scores = []
        score_list = []
        high_score = 0
        best_Criterion = ''
        best_dt_mse = 0
        best_dt_mae = 0
        best_dt_rmse = 0

        for i in criterion_list:
            dt_model = DecisionTreeRegressor(criterion=i)
            self.fit(dt_model)
            dt_y_pred = self.predict(dt_model)
            dt_r2_score, dt_mse, dt_mae, dt_rmse = self.score(dt_y_pred)
            dt_scores.append(dt_r2_score)
            if dt_r2_score > high_score:
                high_score = dt_r2_score
                best_dt_mse = dt_mse
                best_dt_mae = dt_mae
                best_dt_rmse = dt_rmse
                best_Criterion = criterion_list.index(i)
        dt_r2_score = high_score
        dt_mse = best_dt_mse
        dt_mae = best_dt_mae
        dt_rmse = best_dt_rmse

        print(f'The Optimal Criterion is: {criterion_list[best_Criterion]}')
        score_list.extend([dt_r2_score, dt_mse, dt_mae, dt_rmse])
        print('Decision Tree Regression Scores:')
        print(f'r2 score {score_list[0]}, The MSE score {score_list[1]}, MAE score: {score_list[2]} and RMSE score: {score_list[3]}')
        return
        #return dt_r2_score, dt_y_pred, dt_scores, score_list

    # Defining Random Forest Regression function to create a Random Forest Regression estimator.
    # Function loops through the n_estimators and criterion list using different n_estimators and criterion in the model
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.  
    def random_forest_regression(self):
        score_list = []
        n_estimators = [2, 5, 10, 15]
        criterion = ['squared_error', 'absolute_error', 'poisson']
        high_score = 0
        best_mae = 0
        best_mse = 0
        best_rmse = 0

        from sklearn.ensemble import RandomForestRegressor
        for i in n_estimators:
            for j in criterion:
                rf_model = RandomForestRegressor(n_estimators=i, criterion=j)
                #rf_model = GridSearchCV(RandomForestRegressor(), param_grid)
                self.fit(rf_model)
                rf_y_pred = self.predict(rf_model)
                rf_r2_score, rf_mse, rf_mae, rf_rmse = self.score(rf_y_pred)
                if rf_r2_score > high_score:
                    high_score = rf_r2_score
                    best_mse = rf_mse
                    best_mae = rf_mae
                    best_rmse = rf_rmse
        rf_r2_score = high_score
        rf_mse = best_mse
        rf_mae = best_mae
        rf_rmse = best_rmse
        score_list.extend([rf_r2_score, rf_mse, rf_mae, rf_rmse])
        print('Random Forest Regression Scores:')
        print(f'r2 score {score_list[0]}, The MSE score {score_list[1]}, MAE score: {score_list[2]} and RMSE score: {score_list[3]}')
        return
        #return rf_r2_score, rf_y_pred, score_list

    # Defining Support Vector Regression function to create a Support Vector Regression estimator.
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.  
    def svr_regression(self):
        from sklearn.svm import SVR
        score_list = []
        
        svr_model = SVR(kernel='rbf', C=1, gamma=0.1)
        self.fit(svr_model)
        svr_y_pred = self.predict(svr_model)
        svr_r2_score, svr_mse, svr_mae, svr_rmse = self.score(svr_y_pred)
        score_list.extend([svr_r2_score, svr_mse, svr_mae, svr_rmse])
        print('Support Vector Regression Scores:')
        print(f'r2 score {score_list[0]}, The MSE score {score_list[1]}, MAE score: {score_list[2]} and RMSE score: {score_list[3]}')
        return
        #return svr_r2_score, svr_y_pred, score_list

    # Defining Neural Network Regression function to create a Neural Network Regression estimator.
    # Function loops through the opt list using different optimizers in the model
    # Function returns a list of r2 scores and the mae, mse, and rmse scores.  
    def ann_regression(self):
        import keras
        import tensorflow as tf
        from keras.models import Sequential
        opt = ['adam', 'RMSprop', 'sgd']
        high_score = 0
        best_ann_mae = 0
        best_ann_mse = 0
        best_ann_rmse = 0
        score_list = []

        ann_model = Sequential()

        from keras.layers import Dense
        for i in opt:
            input_layer = Dense(units=32, activation='relu')
            ann_model.add(input_layer)

            hidden_layer = Dense(units=16, activation='relu', kernel_initializer='normal')
            ann_model.add(hidden_layer)
            
            hidden_layer = Dense(units=16, activation='relu', kernel_initializer='normal')
            ann_model.add(hidden_layer)

            output_layer = Dense(units=1, activation='linear', kernel_initializer='normal')
            ann_model.add(output_layer)

            ann_model.compile(optimizer = i ,loss = "mean_squared_error", metrics = ['accuracy'])
            
            self.fit(ann_model, 'A')
            
            ann_y_pred = self.predict(ann_model)
            
            ann_r2_score, ann_mse, ann_mae, ann_rmse = self.score(ann_y_pred)

            if ann_r2_score > high_score:
                high_score = ann_r2_score
                best_ann_mae = ann_mse
                best_ann_mse = ann_mae
                best_ann_rmse = ann_rmse
        ann_r2_score = high_score
        ann_mse = best_ann_mse
        ann_mse = best_ann_mae
        ann_mse = best_ann_rmse
        
        score_list.extend([ann_r2_score, ann_mse, ann_mae, ann_rmse])
        print('ANN Regression Scores:')
        print(f'r2 score {score_list[0]}, The MSE score {score_list[1]}, MAE score: {score_list[2]} and RMSE score: {score_list[3]}')
        return
        #return ann_r2_score,score_list