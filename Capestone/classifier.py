import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras

# def fit(model, X_train, y_train):
#         trained_model = model.fit(X_train, y_train)
#         return trained_model
    
# def predict(model, X_test):
#         y_pred = model.predict(X_test)
#         return y_pred

# def score(model, X_test, y_test):
#         best_score = 0
#         s = model.score(X_test, y_test)
#         if s > best_score:
#             best_score = s
#         return best_score
# k = [1, 2, 3, 4, 5]
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(random_state=0)

# from sklearn.neighbors import KNeighborsClassifier
# for i in k:
#     KNeighborsClassifier(n_neighbors=i)

class Classifier:
    
    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    # Defining the Fit function which takes the type estimator and a type that identifies if it is a neural network. Default for type is set blank as this only used by neural network.
    def fit(self, model, type = ''):
        if type == 'A':
            model.fit(self.X_train, self.y_train, batch_size = 20, epochs=50, validation_data= (self.X_test, self.y_test))

        trained_model = model.fit(self.X_train, self.y_train)
        return

    # Defining the Predict function that takes the type of estimator
    def predict(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred

    # Defining the score function which takes the type of estimator the predicitions and type which is used by the Neural Network.
    def score(self, model, y_pred='', type = ''):
        from sklearn.metrics import accuracy_score
        if type == 'A':
            loss, ann_accuracy_score = model.evaluate(self.X_test, self.y_test)
            model_score = ann_accuracy_score
        else:
            #model_score = model.score(self.X_test, self.y_test)
            model_score = accuracy_score(self.y_test, y_pred)
        return model_score

    # Function to create the Logisitc Regression estimator
    def logisticRegression(self):
        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression(random_state=0)
        #Calling the fit function and passing the type of estimator
        self.fit(lr_model)
        #Calling the predict function and passing the type of estimator
        lr_y_pred = self.predict(lr_model)
        #Calling the score function and passing the type of estimator and predictions
        lr_score = self.score(lr_model, lr_y_pred)
        print(f'Logisitic Regression Score: {lr_score}')
        self.display_confusion_matrix(lr_y_pred, 'Logistic Regression')
        return
        #return lr_score, lr_y_pred

    def knnClassifier(self):
        k = [1, 2, 3, 4, 5, 6, 7]
        best_score = 0
        from sklearn.neighbors import KNeighborsClassifier
        for i in k:
            KNN_model = KNeighborsClassifier(n_neighbors=i)
            self.fit(KNN_model)
            KNN_y_pred = self.predict(KNN_model)
            KNN_score = self.score(KNN_model, KNN_y_pred)
            
            if KNN_score > best_score:
                best_score = KNN_score
                best_k = k.index(i)
        print(f'The optimal K is: {k[best_k]}')
        KNN_score = best_score
        print(f'KNN Model Score: {KNN_score}')
        self.display_confusion_matrix(KNN_y_pred, 'KNN')
        return
        #return KNN_score, KNN_y_pred

    def decision_tree_classifer(self):
        c = ['gini', 'entropy', 'log_loss']
        best_score = 0
        from sklearn.tree import DecisionTreeClassifier
        for i in c:
            dt_model = DecisionTreeClassifier(criterion=i, random_state = 0)
            self.fit(dt_model)
            dt_y_pred = self.predict(dt_model)
            dt_score = self.score(dt_model, dt_y_pred)
            if dt_score > best_score:
                best_score = dt_score
        dt_score = best_score
        print(f'Decision Tree Model Score: {dt_score}')
        self.display_confusion_matrix(dt_y_pred, 'Decision Tree')
        return
        #return dt_score, dt_y_pred
    
    # Defining the random forest function to create the random forest estimator that loops through the estimator using different n_estimators and criterion
    def random_forest_classifier(self):
        c = ['gini', 'entropy', 'log_loss']
        n = [2, 5, 10, 15]
        best_score = 0
        from sklearn.ensemble import RandomForestClassifier
        for i in n:
            for j in c:
                rf_model = RandomForestClassifier(n_estimators=i, criterion=j, random_state=0)
                self.fit(rf_model)
                rf_y_pred = self.predict(rf_model)
                rf_score = self.score(rf_model, rf_y_pred)

                if rf_score > best_score:
                    best_score = rf_score
        rf_score = best_score
        print(f'Random Forest Model Score: {rf_score}')
        self.display_confusion_matrix(rf_y_pred, 'Random Forest')
        return
        #return rf_score, rf_y_pred

    # Defining the Support Vector Classifier function to create the Support Vector estimator.
    def svc_classifier(self):
        from sklearn.svm import SVC
        svc_model = SVC(C=0.2, kernel='rbf', gamma='auto')
        self.fit(svc_model)
        svc_y_pred = self.predict(svc_model)
        svc_score = self.score(svc_model, svc_y_pred)
        print(f'Support Vector Classification Model Score: {svc_score}')
        self.display_confusion_matrix(svc_y_pred, 'Support Vector')
        return
        #return svc_score, svc_y_pred

    # Defining the Ann Classifier function to create a Neural Network that loops with a different optimizer.
    def ann_classifier(self):
    
        opti = ['adam', 'Nadam', 'Adadelta', 'Adamax']
        best_score = 0
        from keras.models import Sequential
        from keras import optimizers
        ann_model = Sequential()

        from keras.layers import Dense
        for i in opti:
            input_layer = Dense(units=128, activation='relu', kernel_initializer='uniform')
            ann_model.add(input_layer)

            hidden_layer = Dense(units=64, activation='relu', kernel_initializer='uniform')
            ann_model.add(hidden_layer)
            
            hidden_layer = Dense(units=64, activation='relu', kernel_initializer='uniform')
            ann_model.add(hidden_layer)

            output_layer = Dense(units=5, activation='softmax', kernel_initializer='uniform')
            ann_model.add(output_layer)
        
            ann_model.compile(optimizer = i,  loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
            
            self.fit(ann_model, 'A')
            
            ann_y_pred = self.predict(ann_model)
            accuracy_score = self.score(ann_model,'', 'A')

            if accuracy_score > best_score:
                best_score = accuracy_score
        accuracy_score = best_score
        print(f'ANN Model Score: {accuracy_score}')
        self.display_confusion_matrix(ann_y_pred, 'ANN', 'A')
        return
        #return accuracy_score, ann_y_pred

    def display_confusion_matrix(self, y_pred, name, m=''):
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.metrics import confusion_matrix
        
        if m == 'A':
            y_pred = np.argmax(y_pred,axis=1)
        cm = confusion_matrix(self.y_test, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp = ConfusionMatrixDisplay(cm)
        plt.figure(figsize = (15,15))
        cm_plot = sns.heatmap(cm, square=True, annot=True, cbar=True, cmap=plt.cm.Blues, fmt='.0f')
        #disp.plot(cmap='Blues')
        filename = name
        plt.title(name)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()
        
        cm_plot.figure.savefig(f'{filename}.png')
        return