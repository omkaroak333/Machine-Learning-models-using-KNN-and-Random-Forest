
###The entire code will require 20 - 25 mins to run
## I have no used any custom paths in this code##

##Just make sure that all the files are located inside your default path##
## The ZIP file required to be unziped and the unziped folder must be in the default path##


import pandas_datareader.data as web
import quandl
quandl.ApiConfig.api_key="Insert_your_key"
import pandas
import math
import numpy
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from quandl.errors.quandl_error import NotFoundError

import glob
import os
from os import listdir



###########
#User Defined Functions
###########

#Function to calculate the bollinger bands for given data
def BollingerBands(data, n=20):
    MiddleBand = data['Adj. Close'].rolling(window=n).mean()
    UpperBand = data['Adj. Close'].rolling(window=n).mean() + data['Adj. Close'].rolling(window=n).std()*2
    LowerBand = data['Adj. Close'].rolling(window=n).mean() - data['Adj. Close'].rolling(window=n).std()*2

    return MiddleBand,UpperBand,LowerBand

#Function to calculate RSI for given data
def RSI(data, n=14):
    delta = data['Adj. Close'].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=n).mean()
    RolDown = dDown.rolling(window=n).mean().abs()

    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    return rsi

#Function to calculate multiple factor variables for given data
def calculateFeatures(data,numPeriods = 10):

    data['Return'] = data['Adj. Close'].pct_change(numPeriods)
    data['changeInVolume'] = data['Adj. Volume'].pct_change(14)

    data['SMA_5'] = data['Adj. Close'].rolling(window = 5).mean()
    data['SMA_22'] = data['Adj. Close'].rolling(window = 22).mean()
    data['SMA_200'] = data['Adj. Close'].rolling(window = 200).mean()
    data['EMA_200'] = data['Adj. Close'].ewm(span=200).mean()
    data['EMA_50'] = data['Adj. Close'].ewm(span=50).mean()
    data['MACD(12,26)'] = data['Adj. Close'].ewm(span=12).mean() - data['Adj. Close'].ewm(span=26).mean()
    data['Y'] = numpy.where(data['Return'].shift(-1*numPeriods) > 0, 1, 0)

    data = data.dropna()

    data = data.drop(data.index[len(data)-numPeriods:])

    data['RSI'] = RSI(data,14)

    data['MiddleBand'],data['UpperBand'],data['LowerBand'] = BollingerBands(data,20)

    data = data.replace([numpy.inf,-numpy.inf],numpy.nan)

    data = data.dropna()

    return data

###########
#Main
###########

#Establishing start and end dates

os.chdir(r'C:\Users\TRANSFORMER\Desktop\QCF CLasswork\Computational finance\project 2')
owd = os.getcwd()

start = "2000-1-1"
end = "2018-03-27"

#Getting list of tickers and converting to list

tickersDF = pandas.read_csv('tickers.csv')

tickerList = tickersDF['Ticker'].tolist()

#####Building Models for time period = 10 days, k (KNN) = 5, number of estimators (RF) = 75#####

#Initializing blank data frame
data = pandas.DataFrame()

for ticker in tickerList:

    #Getting data from Quandl
    tempData = quandl.get("WIKI/"+ticker, start_date = start, end_date = end)[['Adj. Close', 'Adj. Volume']]

    #Forward Filling missing data
    tempData = tempData.fillna(method = 'ffill')

    #Dropping blank values in the beginning of dataframe
    tempData = tempData.dropna()

    #Calculating factor variables for data
    tempData = calculateFeatures(tempData)

    #Adding to the main data frame
    data = pandas.concat([data,tempData])

#Splitting training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:,data.columns != 'Y'], data.loc[:,'Y'], test_size=0.40)

#Normalizing training and testing data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Fitting KNN Model
classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train, Y_train)

#Fitting RF Model
classifier_RF = RandomForestClassifier(n_estimators=75)
classifier_RF.fit(X_train,Y_train)

#####Model testing on out-of-sample data#####

counter = 1
errCounter = 0
errTickers = []

#Initializing blank data frame
performanceMetrics = pandas.DataFrame()

for ticker in tickerList:

    counter = counter + 1

    ticker = ticker.strip()

    try:
        #Getting data from Quandl
        data = quandl.get("WIKI/"+ticker.replace('.','_'), start_date = start, end_date = end)[['Adj. Close', 'Adj. Volume']]

        #Forward Filling missing data
        data = data.fillna(method = 'ffill')

        #Dropping blank values in the beginning of dataframe
        data = data.dropna()

        #Calculating factor variables for data
        data = calculateFeatures(data)

        #Normalizing the data
        dataTest = scaler.transform(data.loc[:,data.columns != 'Y'])

        #Getting KNN predicted values for target variable
        predicted_Y_KNN = classifier_KNN.predict(dataTest)

        #Confusion matrix for KNN
        confusion_matrix_KNN = metrics.confusion_matrix(data.loc[:,'Y'], predicted_Y_KNN)

        #ROC Curve variables for KNN
        Y_pred_prob_KNN = classifier_KNN.predict_proba(dataTest)[::,1]
        fpr, tpr, _ = metrics.roc_curve(data.loc[:,'Y'],  Y_pred_prob_KNN)

        #Getting RF predicted values for target variable
        predicted_Y_RF = classifier_RF.predict(dataTest)

        #Confusion matrix for RF
        confusion_matrix_RF = metrics.confusion_matrix(data.loc[:,'Y'], predicted_Y_RF)

        #ROC Curve variables for RF
        Y_pred_prob_RF = classifier_RF.predict_proba(dataTest)[::,1]
        fpr, tpr, _ = metrics.roc_curve(data.loc[:,'Y'],  Y_pred_prob_RF)

        #Adding performance metrics row to data frame
        tempDF = pandas.DataFrame()

        tempDF.loc[ticker,'Ticker'] = ticker
        tempDF.loc[ticker,'Accuracy-KNN'] = metrics.accuracy_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Area Under Curve (ROC)-KNN'] = metrics.roc_auc_score(data.loc[:,'Y'], Y_pred_prob_KNN)
        tempDF.loc[ticker,'Precision-KNN'] = metrics.precision_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Recall-KNN'] = metrics.recall_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Logarithmic Loss-KNN'] = metrics.log_loss(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Accuracy-RF'] = metrics.accuracy_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Area Under Curve (ROC)-RF'] = metrics.roc_auc_score(data.loc[:,'Y'], Y_pred_prob_RF)
        tempDF.loc[ticker,'Precision-RF'] = metrics.precision_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Recall-RF'] = metrics.recall_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Logarithmic Loss-RF'] = metrics.log_loss(data.loc[:,'Y'], predicted_Y_RF)

        performanceMetrics = pandas.concat([performanceMetrics,tempDF])

    except NotFoundError:
        #Exception Handling
        errCounter = errCounter + 1
        errTickers.append(ticker)


#Calculating Average Accuracy column
performanceMetrics['Average Accuracy'] = (performanceMetrics['Accuracy-KNN'] + performanceMetrics['Accuracy-RF'])/2

#Sorting Data by Average Accuracy
performanceMetrics = performanceMetrics.sort_values(by=['Average Accuracy'], ascending = False)

#Exporting to CSV with 10 tickers
performanceMetrics.to_csv('Kulkarni_Omkar_10_Stocks.csv',index = False)


##Model testing on large universe of stocks##


### In this case I have directly downloaded the data from Quandl for the large universe as well###
### I have not used the provided CSV files as I was having normalization errors with same###

## the starting date is 02/01/2015 and the ending date is 26/10/2018 as per the dates in CSV##
## the above dates are similar to those in the CSV files##
## I am using the 507 tickers as per the csv files provided by you###

##I am getting a better output by using the direct method##



##Model testing on large universe of stocks##

#defining starting and ending dates
start = "2015-2-1"
end = "2018-03-27"



os.chdir("./stock_dfs")

#function defined to get the ticker names
def find_csv_filenames( path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

#have included the path of the stock_dfs folder as below

filenames = find_csv_filenames(".")

newfilename = [x[:-4] for x in filenames]


#Converting list to set
setTickers = set(newfilename)

counter = 1
errCounter = 0
errTickers = []

#Initializing blank data frame
performanceMetrics = pandas.DataFrame()

##the below for loop took me 15 to 20 minutes to run. Apologies for this inconvenience##
## for loop for getting the data for larger universe of 507 stocks from the zip folder##

for ticker in setTickers:
    
    counter = counter + 1

    ticker = ticker.strip()

    try:
        #importing data for 507 tickers
        data = quandl.get("WIKI/"+ticker.replace('.','_'), start_date = start, end_date = end)[['Adj. Close', 'Adj. Volume']]

        #Forward Filling missing data
        data = data.fillna(method = 'ffill')

        #Dropping blank values in the beginning of dataframe
        data = data.dropna()

        #Calculating factor variables for data
        data = calculateFeatures(data)

        #Normalizing the data
        dataTest = scaler.transform(data.loc[:,data.columns != 'Y'])

        #Getting KNN predicted values for target variable
        predicted_Y_KNN = classifier_KNN.predict(dataTest)

        #Confusion matrix for KNN
        confusion_matrix_KNN = metrics.confusion_matrix(data.loc[:,'Y'], predicted_Y_KNN)

        #ROC Curve variables for KNN
        Y_pred_prob_KNN = classifier_KNN.predict_proba(dataTest)[::,1]
        fpr, tpr, _ = metrics.roc_curve(data.loc[:,'Y'],  Y_pred_prob_KNN)

        #Getting RF predicted values for target variable
        predicted_Y_RF = classifier_RF.predict(dataTest)

        #Confusion matrix for RF
        confusion_matrix_RF = metrics.confusion_matrix(data.loc[:,'Y'], predicted_Y_RF)

        #ROC Curve variables for RF
        Y_pred_prob_RF = classifier_RF.predict_proba(dataTest)[::,1]
        fpr, tpr, _ = metrics.roc_curve(data.loc[:,'Y'],  Y_pred_prob_RF)

        #Adding performance metrics row to data frame
        tempDF = pandas.DataFrame()

        tempDF.loc[ticker,'Ticker'] = ticker
        tempDF.loc[ticker,'Accuracy-KNN'] = metrics.accuracy_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Area Under Curve (ROC)-KNN'] = metrics.roc_auc_score(data.loc[:,'Y'], Y_pred_prob_KNN)
        tempDF.loc[ticker,'Precision-KNN'] = metrics.precision_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Recall-KNN'] = metrics.recall_score(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Logarithmic Loss-KNN'] = metrics.log_loss(data.loc[:,'Y'], predicted_Y_KNN)
        tempDF.loc[ticker,'Accuracy-RF'] = metrics.accuracy_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Area Under Curve (ROC)-RF'] = metrics.roc_auc_score(data.loc[:,'Y'], Y_pred_prob_RF)
        tempDF.loc[ticker,'Precision-RF'] = metrics.precision_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Recall-RF'] = metrics.recall_score(data.loc[:,'Y'], predicted_Y_RF)
        tempDF.loc[ticker,'Logarithmic Loss-RF'] = metrics.log_loss(data.loc[:,'Y'], predicted_Y_RF)

        performanceMetrics = pandas.concat([performanceMetrics,tempDF])

    except:
        continue
    
#Avg accuracy column calculation
performanceMetrics['Average Accuracy'] = (performanceMetrics['Accuracy-KNN'] + performanceMetrics['Accuracy-RF'])/2

#Sorting Data by Average Accuracy
performanceMetrics = performanceMetrics.sort_values(by=['Average Accuracy'], ascending = False)

#resetting the current working directory
os.chdir(owd)

#Getting the best predicted 20 stocks
bestPredictedStocks = performanceMetrics.iloc[:20,]
bestPredictedStocks.to_csv('Best Predicted Stocks_2.csv',index = False)











