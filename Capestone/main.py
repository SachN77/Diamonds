import Analyzer
from classifier import Classifier
import classifier
from regressor import Regressor
from cluster import Cluster
import warnings

warnings.filterwarnings('ignore')

#Intailizing Analyzer module an setting it to a variable call data
data = Analyzer

#Calling the read csv file function to read the diamond csv file and store it in variable called df
df = data.read_dataset()

# Calling display_dataset function to display the dataset that was read through Pandas. Passing the df variable into function
data.display_dataset(df)

# Prints the input features attribute types
data.describe_data(df)

# data.check_null(df)
# Calling the drop function to drop the Unnamed: 0 column.  Passing the df variable into function
df = data.drop_col(df)

# Calling plot correlation matrix to plot and display a correlation matrix for the diamond dataset.  Passing the df variable into function
data.plot_correlationMatrix(df)

# Calling the plot pair plot function to plot and display a pairplot for the diamond dataset.  Passing the df variable into function
data.plot_pairPlot(df)

# Calling remove zero value function to set all zero values in the x, y, z columns to NAN.
df = data.remove_zeroValues(df)

#Calling the check null function to find the nan values.
data.check_null(df)

#Dropping all the nan values from the dataset.
df = data.drop_nanValues(df)

#Checking to see if all nan values have been removed
data.check_null(df)

# Plotting the histogram with the categorical values
data.plot_histograms_categorical(df)

# Plotting the numerical attributes on a histogram
data.plot_histograms_numeric(df)

# Plotting a box plot with all the dataset attributes
data.plot_boxPlot(df)

# Calling label encoding function to encode the label for the classifier label (Cut).
encoded_df = data.label_encoding(df)

# Calling the encoding function to encode the categorical features
encoded_df = data.encoding(encoded_df)

# Calling the shuffle function to shuffle the input data randomly
encoded_df = data.shuffle_dataset(encoded_df)

# Asking the user to enter a floating number between 0.0 and 1.0.
data_sample_size = float(input('Please enter a number from 0.0 to 1.0 for sample size to be used: '))

# calling the function sample to get the sample size of the dataset by using the number that the user entered.
encoded_df = data.sample(data_sample_size, encoded_df)




'''
Calling and Running the Classifier class from classifier.py

'''


# Setting the target or label value to the y_classifier variable
# The 'C' argument is for the function to identify what attribute to set as the label
y_classifier = data.set_target(encoded_df, 'C')

# Dropping the label value from the dataset. To make it easier to set X values.
data_df = data.drop_col(encoded_df, 'C')

# Setting the X values for the Classifier models.
X = data.set_x_value(data_df)

# Scaling the X values with a standard scaler.
X_scaled = data.scale_data(X)

# Passing the X scaled data and the label to the function to create a train and test split.
X_train, X_test, y_train, y_test = data.create_train_test_data(X_scaled, y_classifier)

# Instantiate the Classifier class and pass the train and test data to the class.
c_df = Classifier(X_train, X_test, y_train, y_test)

# Calling the Logistic Regression function to run the Logistic Regression estimator.
c_df.logisticRegression()


# Calling the KNN Classifier function to run the KNN Classifier estimator.
c_df.knnClassifier()

# Calling the Decision Tree Classifier function to run the Desicion Tree estimator.
c_df.decision_tree_classifer()

# Calling the Decision Tree Classifier function to run the Desicion Tree estimator.
c_df.random_forest_classifier()

# Calling the Support Vector Classifier function to run the Support Vector estimator.
c_df.svc_classifier()


# Calling the ANN Classifier function to run the Artifical Neural Network estimator.
c_df.ann_classifier()


"""
Instantiating Regressor class to run the regressor models

"""




#Setting the target or label value to the y_regression variable
#The 'R' argument is for the function to identify what attribute to set as the label
y_regression = data.set_target(encoded_df, 'R')

y_regression = data.scale_data(y_regression.array.reshape(-1, 1))

# Dropping the label value from the dataset. To make it easier to set X values.
r_data_df = data.drop_col(encoded_df, 'R')

# Setting the X values for the Regression models.
X = data.set_x_value(r_data_df)

# Scaling the data for the regression models
X_scaled = data.scale_data(X)

# Passing the X scaled data and the label to the function to create a train and test split.
X_train, X_test, y_train, y_test = data.create_train_test_data(X_scaled, y_regression)

# # Intstantiating the Regressor class from regressor.py and passing in the train and test data.
r_df = Regressor(X_train, X_test, y_train, y_test)

# Calling the Linear Regression function to run the Linear Regression estimator.
r_df.linearRegression()

# Calling the KNN Regression function to run the KNN Regression estimator.
r_df.knnRegression()

# Calling the Decision Tree function to run the Decision Tree Regression estimator.
r_df.decision_tree_regression()


# Calling the Random Forest Regression function to run the Random Forest estimator.
r_df.random_forest_regression()

# Calling the Support Vector Regression function to run the Support Vector Regression estimator.
r_df.svr_regression()


# Calling the ANN Regression function to run the ANN Regression estimator.
r_df.ann_regression()

"""
Instantiating Clustering class to run the cluster models

"""



#Setting the X values
X = data.set_x_value(encoded_df, 'CL')

#Calling the scale data function which takes the X values.
X_scaled = data.scale_data(X)

# Intstantiating the Cluster class from cluster.py
cl_df = Cluster(X)

# # Calling the KmClustering function to create and run the Kmeans cluster model.
cl_df.kmClustering()

# # Calling the Hierarchcal Clustering function to create and run the Hierarchal cluster model.
cl_df.hierarchicalClustering()

# # Calling the Mean Shift Clustering function to create and run the Mean Shift cluster model.
cl_df.meanshiftClustering()


# # #print(done)
# # #print(f'The Inertia value for KMeans Clustering: {intertia}')
# # # print(finished)

# # print('\n')
# # print('\n')

print('FINISHED RUNNING ALL MODELS!!!!')