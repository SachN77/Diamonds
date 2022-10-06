
from matplotlib import colorbar, colors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

#class DataProcessing():

# Fuction that reads the csv file and saves it in a variable and returns the variable
def read_dataset():
    df = pd.read_csv('diamonds.csv')
    return df

def display_dataset(df):
    print(df)
    return

def check_null(df):
    print(df.isnull().sum())
    return 

def plot_pairPlot(df):
    #plt.figure(figsize=(10,15))
    sns.pairplot(df)
    plt.show()
    return

def shuffle_dataset(df):
    from sklearn.utils import shuffle
    df = shuffle(df)
    return df

def label_encoding(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['cut'] = le.fit_transform(df['cut'])
    #df['color'] = le.fit_transform(df['color'])
    #df['clarity'] = le.fit_transform(df['clarity'])
    
    return df

def encoding(df):
    from sklearn.preprocessing import OneHotEncoder
    import category_encoders as ce
    encoder = ce.OneHotEncoder(df, cols=['color', 'clarity'], use_cat_names=True)

    df = encoder.fit_transform(df)
    return df

def set_x_value(df, type =''):
    if type == 'CL':
        X = df.iloc[:, 1:20:18].values
    else:
        X = df.iloc[:, :].values
    return X

def scale_data(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False)

    scaled_df = scaler.fit_transform(X)
    return scaled_df

def create_train_test_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

def drop_col(df, type = ''):
    
    if type == 'C':
        #df = df.drop(['Unnamed: 0', 'cut'], axis=1)
        df = df.drop(['cut'], axis=1)
    elif type == 'R':
         #df = df.drop(['Unnamed: 0', 'price'], axis=1)
         df = df.drop(['price'], axis=1)
    else:
        #df = df.drop(['Unnamed: 0'].index, inplace=True)
        df = df.iloc[:, ~df.columns.isin(['Unnamed: 0'])]

    return df


    
def set_target(df, type):
    if type == 'C':
        label = df.iloc[:, 1]
    elif type == 'R':
        label = df.iloc[:, 19]
    return label
    

def sample(num, df):
    df = df.sample(frac = num)
    return df

def display_confusion_matrix(y_test, y_pred, name, m=''):
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    
    if m == 'A':
        y_pred = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(y_test, y_pred)
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
    

def plot_histograms_categorical(df):
    #df.hist(bins = 100, figsize = (20, 15))
    ig, axs = plt.subplots(1, 3, figsize=(7, 7))

    axs[0].hist(data=df, x='cut', bins=20, color='blue')
    axs[1].hist(data=df, x='color', bins=20, color='blue')
    axs[2].hist(data=df, x='clarity', bins=20, color='blue')

    axs[0].set_title('Cut')
    axs[1].set_title('Color')
    axs[2].set_title('Clarity')
    plt.show()
    return

def plot_histograms_numeric(df):
    df.hist(bins = 100, grid=False,figsize = (20, 15), color='blue')
    plt.show()

    return

def plot_boxPlot(df):
    fig, axs = plt.subplots(1, 7, figsize=(20, 15))
    axs[0].boxplot(data=df, x='carat')
    axs[1].boxplot(data=df, x='depth')
    axs[2].boxplot(data=df, x='table')
    axs[3].boxplot(data=df, x='price')
    axs[4].boxplot(data=df, x='x')
    axs[5].boxplot(data=df, x='y')
    axs[6].boxplot(data=df, x='z')
    


    #Setting titles to each subplot
    axs[0].set_title('Carat')
    axs[1].set_title('Depth')
    axs[2].set_title('Table')
    axs[3].set_title('Price')
    axs[4].set_title('X')
    axs[5].set_title('Y')
    axs[6].set_title('Z')
    
    # Setting the y axis label font
    axs[0].tick_params(labelsize=8)
    axs[1].tick_params(labelsize=8)
    axs[2].tick_params(labelsize=8)
    axs[3].tick_params(labelsize=8)
    axs[4].tick_params(labelsize=8)
    axs[5].tick_params(labelsize=8)
    axs[6].tick_params(labelsize=8)
  
    
    fig.tight_layout()
    plt.show()
    return

def plot_correlationMatrix(df):
    plt.figure(figsize=(15,15))
    sns.heatmap(df.corr(), annot=True)
    plt.show()
    return

def remove_zeroValues(df):
    df[['x','y','z']] = df[['x','y','z']].replace(0,np.NaN)
    return df

def drop_nanValues(df):
    df = df.dropna()
    return df

def describe_data(df):
    print(df.describe())
    return


     # ig, axs = plt.subplots(3, 3, figsize=(20, 15))
    # axs[0,0].hist(data=df, x='carat', bins=20, color='blue')
    # axs[0,1].hist(data=df, x='depth', bins=20, color='blue')
    # axs[0,2].hist(data=df, x='table', bins=20, color='blue')
    # axs[1,0].hist(data=df, x='price', bins=20, color='blue')
    # axs[1,1].hist(data=df, x='x', bins=20, color='blue')
    # axs[1,2].hist(data=df, x='y', bins=20, color='blue')
    # axs[2,0].hist(data=df, x='z', bins=20, color='blue')

    # #Setting titles to each subplot
    # axs[0, 0].set_title('Carat')
    # axs[0, 1].set_title('Depth')
    # axs[0, 2].set_title('Table')
    # axs[1, 0].set_title('Price')
    # axs[1, 1].set_title('X')
    # axs[1, 2].set_title('Y')
    # axs[2, 0].set_title('Z')