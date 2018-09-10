#In this script there were created functions for the models that provide the results of the models and additional graphs
#Also there are some models that were not used or were not succeful. These tries are not in a function
#The techinques that were not used are in comments so these comments can be removed in order to use them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

df = pd.read_csv("man2_for_matlab_with_weather_avg_airline.csv")    #read the data from the file
data = pd.read_csv('AircraftTypeCodes.csv')     #read the aircraft model and size file
#print data
#print df.dtypes   #change the wrong data types.
data.columns = ['a']    #set a column name 'a'
data = data['a'].str.split()    #split the data in the file
#print data
#print data[1][1]
#data["a"].str.replace(r".*\t","")
#d = re.sub(r'.*\t',"", data['a'])
#print data
#print data[3][1]
model = []
size = []
for i in range(0,390):  #in the range of the rows in the file
    if len(data[i]) > 4:    #because the data that are needed have more than 4 values in the row
        model.append(data[i][1])    #append the data of model to the list model
        size.append(data[i][-1])    #append the data of size to the list size
#sdf = pd.DataFrame(size)
#print model
#print size
#print len(model), len(size)

#model = [x for x in model if x != 'n/a']
#model.remove('n/a')
#model.remove('Code')
#print model
#print len(model)

df_sm = pd.DataFrame({'model_aircraft':model})  #create a dataframe with the data of the list model
df_sm['size_aircraft'] = size   #add the data of the list size to the data frame
df_sm = df_sm[df_sm.model_aircraft != 'n/a']    #remove n/a values
df_sm = df_sm[df_sm.model_aircraft != 'Code']   #remove 'Code' values
#df_sm = df_sm[df_sm.size != 'Category']
df_sm = df_sm[df_sm.size_aircraft != 'n/a']     #remove n/a from size_aircraft
df_sm = df_sm.drop_duplicates()     #remove duplicates
df_sm.reset_index(drop = True, inplace = True)  #reset the index of the data frame
#print df_sm

#Import average speed (distance/TaxiTime) in the datase
df['Average_Speed'] = df.distance / df.TaxiTime     #divide the distance and the TaxiTime in order to find the average speed


df = df[df.flightNumber != 'null']  #remove null values
df = df[df.airline != 'null']
df = df[df.aircraftModel != 'null']
df = df.drop('operation_mode', 1)   #delete operation mode column
df = df.drop('flightNumber', 1)   #delete flight Number column
df = df.drop('airline', 1)   #delete airline column
##df = df.drop('aircraftModel', 1)   #delete aircraft Model column
#df = df.drop('isSnow', 1)   #delete is Snow column
#df = df.drop('isDrizzle', 1)   #delete is Drizzle column
#df = df.drop('isFog', 1)   #delete is Fog column
#df = df.drop('isMist', 1)   #delete is Mist column
df = df.drop('isHaze', 1)   #delete is Haze column
df = df.drop('isHail', 1)   #delete is Hail column

def remove_outlier(df_in, col_name):    #I did not use this function because I was losing a lot of data
    q1 = df_in[col_name].quantile(0.20) #first quantile with values less than the 20%
    q3 = df_in[col_name].quantile(0.80) #third quantile with values after the 80% of the sample
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def remove_outlier2(df_in, col_name):   #import the datadrame and the variable that I want to remove the outliers
    q = df_in[col_name].quantile(0.99)  #only 0.01% percent of the data are removed by the last data value in the sample
    return df_in[df_in[col_name] > q]   #return the dataframe without the outlier

def remove_outlier3(df_in, col_name):   #import the datadrame and the variable that I want to remove the outliers
    q = df_in[col_name].quantile(0.01)  #only 0.01% percent of the data are removed by the first data value in the sample
    return df_in[df_in[col_name] < q]   #return the dataframe without the outlier

def remove_outlier4(df_in, col_name):
    q = df_in[col_name].quantile(0.995) #only 0.005% percent of the data are removed by the last data value in the sample
    return df_in[df_in[col_name] > q]

def remove_outlier5(df_in, col_name):
    q = df_in[col_name].quantile(0.95)  #only 0.005% percent of the data are removed by the last data value in the sample
    return df_in[df_in[col_name] > q]

def remove_outlier6(df_in, col_name):
    q = df_in[col_name].quantile(0.05)  #only 0.05% percent of the data are removed by the first data value in the sample
    return df_in[df_in[col_name] < q]


def histogram(df_in, col_name, title, valuex, valuey):  #import the dataframe, the variable, the title of the histogram,value of x axis, value of y axis
    plt.hist(df_in[col_name])   #plot the variable that was set
    plt.title(title)    #plot's title
    plt.xlabel(valuex)  #x axis
    plt.ylabel(valuey)  #y axis
    return plt.show()   #plot



#the outliers are added in a list in order to check for duplicate values of the outliers
outlier_df = remove_outlier2(df, 'TaxiTime')
outlier_df = outlier_df.append(remove_outlier2(df, 'distance'))
outlier_df = outlier_df.append(remove_outlier2(df, 'angle_error'))
outlier_df = outlier_df.append(remove_outlier2(df, 'distance_long'))
outlier_df = outlier_df.append(remove_outlier4(df, 'angle_sum'))
outlier_df = outlier_df.append(remove_outlier2(df, 'QDepDep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'QDepArr'))
outlier_df = outlier_df.append(remove_outlier2(df, 'QArrDep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'QArrArr'))
outlier_df = outlier_df.append(remove_outlier5(df, 'NArrDep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'NDepDep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'NDepArr'))
outlier_df = outlier_df.append(remove_outlier2(df, 'NArrArr'))
outlier_df = outlier_df.append(remove_outlier6(df, 'VisibilityInMeters'))
outlier_df = outlier_df.append(remove_outlier2(df, 'WindSpeedInMPS'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast5Dep'))
outlier_df = outlier_df.append(remove_outlier3(df, 'AvgSpdLast5Dep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast5Arr'))
outlier_df = outlier_df.append(remove_outlier3(df, 'AvgSpdLast5Arr'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast5'))
outlier_df = outlier_df.append(remove_outlier3(df, 'AvgSpdLast5'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast10Dep'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast10Arr'))
outlier_df = outlier_df.append(remove_outlier3(df, 'AvgSpdLast10Arr'))
outlier_df = outlier_df.append(remove_outlier2(df, 'AvgSpdLast10'))
outlier_df = outlier_df.append(remove_outlier2(df, 'Average_Speed'))
outlier_df = outlier_df.drop_duplicates()
#print outlier_df
df = df[~df.isin(outlier_df)].dropna()  #remove the duplicate values of outliers
#print df[~df.isin(outlier_df)].dropna()
#print df


df.reset_index(drop = True, inplace = True) #reset the index in the dataframe
#df[["depArr", "isRain", "isSnow", "isDrizzle", "isFog", "isMist", "isHaze", "isHail"]] = df[["depArr", "isRain", "isSnow", "isDrizzle", "isFog", "isMist", "isHaze", "isHail"]].astype("object")    #change the incorrect data types    
#print df.dtypes
#Data standardisation
# transform TaxiTime from seconds to minutes
df['TaxiTime'] = df["TaxiTime"]*60
#print df.head()    #print the first values in the data frame
#print df.tail()    #print the last values in the data frame



#Process in order to import the aircraft size in the dataset
df_list = df['aircraftModel'].tolist()  #put the column aircraftModel in a list

df_mod = df_sm['model_aircraft'].tolist()   #put the column model_aircraft in a list
df_siz = df_sm['size_aircraft'].tolist()    #put the column size_aircraft in a list
list_null_mod=[]
df_new= []  #the new list with the size of the aircraft that will be imported in df data frame
index = []  #list that keeps track of the indexes
indexi = None   #set indexi as None
for i in range(8001):   #range of the number of data in df_list
    for j in range(175):    #range in the number of data in df_mod and df_siz
        if df_list[i] == df_mod[j]:     #if the model is the same
            df_new.append(df_siz[j])    #append the size of the aircraft
            index.append(i)     #append the index
            indexi = i  #change this variable with the value of i
            break
    if indexi != i:     #if this is true, then there was no value in the new list so we can put 'null'
        df_new.append('L')   #append 'null' in the list
#        list_null_mod.append(df_mod[j])     #list with all the models in the dataframe with null values

df['aircraft_size'] = df_new     #add this new column to the original data frame df
#set numerical values to the size of aircrafts. Specifically, L=1, M=2, H=3
df['aircraft_size'] = df['aircraft_size'].replace('L', 1)
df['aircraft_size'] = df['aircraft_size'].replace('M', 2)
df['aircraft_size'] = df['aircraft_size'].replace('H', 3)

df.reset_index(drop = True, inplace = True) #reset the index on the data points
#Create dummies for the aircraft sizw
#dummy_variable_1 = pd.get_dummies(df["aircraft_size"])
#dummy_variable_1.rename(columns={'H':'Heavy_aircrafts', 'M':'Medium_aircrafts', 'L':'Light_aircrafts'}, inplace=True)
#print dummy_variable_1.head()
# merge data frame "df" and "dummy_variable_1" 
#df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
#df.drop("aircraft_size", axis = 1, inplace=True)
#print df    #print the data frame

#Histogram for every variable
#histogram(df, 'TaxiTime', 'Histogram of Taxi Time' , "seconds", 'instances')
#'''df = remove_outlier(df, 'TaxiTime')'''
#histogram(df, 'TaxiTime', 'Histogram of Taxi Time' , "seconds", 'instances')
#histogram(df, 'depArr', 'Histogram of Departures - Arrivals' , "deparure-0, arrivals-1", 'instances')
#histogram(df, 'distance', 'Histogram of distance' , "meters", 'instances')
#histogram(df, 'angle_error', 'Histogram of angle error' , "number of angles", 'instances')
#histogram(df, 'distance_long', 'Histogram of distance long' , "meters", 'instances')
#histogram(df, 'angle_sum', 'Histogram of angle sum' , "number of angles", 'instances')
#histogram(df, 'QDepDep', 'Histogram of QDepDep' , "", 'instances')
#histogram(df, 'QDepArr', 'Histogram of QDepArr' , "", 'instances')
#histogram(df, 'QArrDep', 'Histogram of QArrDep' , "", 'instances')
#histogram(df, 'QArrArr', 'Histogram of QArrArr' , "", 'instances')
#histogram(df, 'NDepDep', 'Histogram of NDepDep' , "", 'instances')
#histogram(df, 'NDepArr', 'Histogram of NDepArr' , "", 'instances')
#histogram(df, 'NArrDep', 'Histogram of NArrDep' , "", 'instances')
#histogram(df, 'NArrArr', 'Histogram of NArrArr' , "", 'instances')
#histogram(df, 'Pressure', 'Histogram of Pressure' , "", 'instances')
#histogram(df, 'VisibilityInMeters', 'Histogram of Visibility In Meters' , "meters", 'instances')
#histogram(df, 'TemperatureInCelsius', 'Histogram of Temperature In Celsius' , "Celsius", 'instances')
#histogram(df, 'WindSpeedInMPS', 'Histogram of Wind Speed In MPS' , "meters per second", 'instances')
###histogram(df, 'isRain', 'Histogram of Rain' , "", 'instances')
#histogram(df, 'isSnow', 'Histogram of Snow' , "", 'instances')
###histogram(df, 'isDrizzle', 'Histogram of Drizzle' , "", 'instances')
#histogram(df, 'isFog', 'Histogram of Fog' , "", 'instances')
#histogram(df, 'isMist', 'Histogram of Mist' , "", 'instances')
###histogram(df, 'isHaze', 'Histogram of Haze' , "", 'instances')
###histogram(df, 'isHail', 'Histogram of Hail' , "", 'instances')
###histogram(df, 'flightNumber', 'Histogram of flight Number' , "", 'instances')
###histogram(df, 'airline', 'Histogram of airline' , "", 'instances')
###histogram(df, 'aircraftModel', 'Histogram of aircraft model' , "", 'instances')
#histogram(df, 'AvgSpdLast5Dep', 'Histogram of AvgSpdLast5Dep' , "m/min", 'instances')
#histogram(df, 'AvgSpdLast5Arr', 'Histogram of AvgSpdLast5Arr' , "m/min", 'instances')
#histogram(df, 'AvgSpdLast5', 'Histogram of AvgSpdLast5' , "m/min", 'instances')
#histogram(df, 'AvgSpdLast10Dep', 'Histogram of AvgSpdLast10Dep' , "m/min", 'instances')
#histogram(df, 'AvgSpdLast10Arr', 'Histogram of AvgSpdLast10Arr' , "m/min", 'instances')
#histogram(df, 'AvgSpdLast10', 'Histogram of AvgSpdLast10' , "m/min", 'instances')
##histogram(df, 'aircraft_size', 'Histogram of aicraft size', '', 'instances')
#histogram(df, 'Average_Speed', 'Histogram of average Speed', 'm/sec', 'instances')
#histogram(df, 'Medium_aircrafts', 'Histogram of Medium aircrafts', '', 'instances')
#histogram(df, 'Light_aircrafts', 'Histogram of Light aircrafts', '', 'instances')
#histogram(df, 'Heavy_aircrafts', 'Histogram of heavy aircrafts', '', 'instances')
#histogram(df, 'aircraft_size', 'Histogram of aircraft size', '', 'instances')

#df.to_csv('clean_df2.csv')     #crate a csv of the data frame


#Exploratory Data Analysis
#print df.describe() #provide with all the statistical elements for every variable
#print df.corr() #the correlation of every variable with each other

#I can find the correlation between continuous numerical variables
#print df[["depArr", "TaxiTime"]].corr()
#sns.boxplot(x="depArr", y="TaxiTime", data=df)

#Scater plots and box plots that show the correlation between the variables and the Taxi Time
#print df[["distance", "TaxiTime"]].corr()
#sns.regplot(x="distance", y="TaxiTime", data=df)

#print df[["angle_error", "TaxiTime"]].corr()
#sns.boxplot(x="angle_error", y="TaxiTime", data=df)
#sns.regplot(x="angle_error", y="TaxiTime", data=df)

#print df[["distance_long", "TaxiTime"]].corr()
#sns.regplot(x="distance_long", y="TaxiTime", data=df)

#print df[["angle_sum", "TaxiTime"]].corr()
#sns.regplot(x="angle_sum", y="TaxiTime", data=df)

#print df[["QDepDep", "TaxiTime"]].corr()
#sns.boxplot(x="QDepDep", y="TaxiTime", data=df)

#print df[["QDepArr", "TaxiTime"]].corr()
#sns.boxplot(x="QDepArr", y="TaxiTime", data=df)

#print df[["QArrDep", "TaxiTime"]].corr()
#sns.boxplot(x="QArrDep", y="TaxiTime", data=df)

#print df[["QArrArr", "TaxiTime"]].corr()
#sns.boxplot(x="QArrArr",y="TaxiTime", data=df)

#print df[["NDepDep", "TaxiTime"]].corr()
#sns.boxplot(x="NDepDep", y="TaxiTime", data=df)

#print df[["NDepArr", "TaxiTime"]].corr()
#sns.boxplot(x="NDepArr", y="TaxiTime", data=df)

#print df[["NArrDep", "TaxiTime"]].corr()
#sns.boxplot(x="NArrDep", y="TaxiTime", data=df)

#print df[["NArrArr", "TaxiTime"]].corr()
#sns.boxplot(x="NArrArr", y="TaxiTime", data=df)

#print df[["Pressure", "TaxiTime"]].corr()
#sns.regplot(x="Pressure", y="TaxiTime", data=df)

#print df[["VisibilityInMeters", "TaxiTime"]].corr()
#sns.regplot(x="VisibilityInMeters", y="TaxiTime", data=df)

#print df[["TemperatureInCelsius", "TaxiTime"]].corr()
#sns.regplot(x="TemperatureInCelsius", y="TaxiTime", data=df)

#print df[["WindSpeedInMPS", "TaxiTime"]].corr()
#sns.regplot(x="WindSpeedInMPS", y="TaxiTime", data=df)

#print df[["isRain", "TaxiTime"]].corr()
#sns.boxplot(x="isRain", y="TaxiTime", data=df)

#print df[["isSnow", "TaxiTime"]].corr()
#sns.boxplot(x="isSnow", y="TaxiTime", data=df)

#print df[["isDrizzle", "TaxiTime"]].corr()
#sns.boxplot(x="isDrizzle", y="TaxiTime", data=df)

#print df[["isFog", "TaxiTime"]].corr()
#sns.boxplot(x="isFog", y="TaxiTime", data=df)

#print df[["AvgSpdLast5Dep", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast5Dep", y="TaxiTime", data=df)

#print df[["AvgSpdLast5Arr", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast5Arr", y="TaxiTime", data=df)

#print df[["AvgSpdLast5", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast5", y="TaxiTime", data=df)

#print df[["AvgSpdLast10Dep", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast10Dep", y="TaxiTime", data=df)

#print df[["AvgSpdLast10Arr", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast10Arr", y="TaxiTime", data=df)

#print df[["AvgSpdLast10", "TaxiTime"]].corr()
#sns.regplot(x="AvgSpdLast10", y="TaxiTime", data=df)

#Correlation for nominal variables

#sns.boxplot(x="flightNumber", y="TaxiTime", data=df)

#sns.boxplot(x="airline", y="TaxiTime", data=df)

#sns.boxplot(x="aircraftModel", y="TaxiTime", data=df)

#print df[["AvgSpdLast10Dep", "TaxiTime"]].corr()
#sns.boxplot(x="aircraft_size", y="TaxiTime", data=df)

#print df[["Heavy_aircrafts", "TaxiTime"]].corr()
#sns.boxplot(x="Heavy_aircrafts", y="TaxiTime", data=df)

#print df[["Medium_aircrafts", "TaxiTime"]].corr()
#sns.boxplot(x="Medium_aircrafts", y="TaxiTime", data=df)

#print df[["Light_aircrafts", "TaxiTime"]].corr()
#sns.boxplot(x="Light_aircrafts", y="TaxiTime", data=df)

#print df[["Average_Speed", "TaxiTime"]].corr()
#sns.regplot(x="Average_Speed", y="TaxiTime", data=df)

#print df[["aircraft_size", "TaxiTime"]].corr()
#sns.boxplot(x="aircraft_size", y="TaxiTime", data=df)

#print df.describe()
'''
#Pearson correlation and P-value
pearson_coef, p_value = stats.pearsonr(df['distance'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for distance is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['depArr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for Depparture and arrival is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['angle_error'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for angle error is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['distance_long'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for distance long is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['angle_sum'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for angle sum is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['QDepDep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for QDepDep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['QDepArr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for QDepArr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['QArrDep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for QArrDep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['QArrArr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for QArrArr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['NDepDep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for NDepDep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['NDepArr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for NDepArr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['NArrDep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for NArrDep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['NArrArr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for NArrArr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['Pressure'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for Pressure is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['VisibilityInMeters'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for Visibility in Meters is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['TemperatureInCelsius'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for Temperature in Celsius is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['WindSpeedInMPS'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for Wind Speed in MPS is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['isRain'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for isRain is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['isSnow'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for isSnow is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['isDrizzle'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for isDrizzle is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['isFog'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for isFog is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['isMist'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for isMist is", pearson_coef, " with a P-value of P =", p_value


pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast5Dep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for AvgSpdLast5Dep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast5Arr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for AvgSpdLast5Arr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast5'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for AvgSpdLast5 is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast10Dep'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for AvgSpdLast10Dep is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast10Arr'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for AvgSpdLast10Arr is", pearson_coef, " with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['AvgSpdLast10'], df['TaxiTime'])
print'The Pearson Correlation Coefficient for AvgSpdLast10 is', pearson_coef, 'with a P-value of P =', p_value

pearson_coef, p_value = stats.pearsonr(df['aircraft_size'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for aircraft size is", pearson_coef, "with a P-value of P =", p_value
'''



df = df.drop('aircraftModel', 1)   #delete aircraft Model column
df = df.drop('Average_Speed', 1)   #delete aircraft Model column



################################################################################################################
#  Linear Regression - Introduction
lm = LinearRegression()
X = df[['depArr']]
Y = df['TaxiTime']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]
#print "intercept", lm.intercept_
#print "slope", lm.coef_
#So: taxitime = 653.499331934 - 367.8457605*depArr

lm1 = LinearRegression()
lm1.fit(df[['distance']], df[['TaxiTime']])
# Slope 
#print lm1.coef_
## Intercept
#print lm1.intercept_
#So: taxitime = 184.59790162 +0.20443324*distance

lm2 = LinearRegression()
lm2.fit(df[['angle_error']], df[['TaxiTime']])
# Slope 
#print lm2.coef_
## Intercept
#print lm2.intercept_
#So: taxitime = 512.66903409 + 98.48629342*angle_error

lm3 = LinearRegression()
lm3.fit(df[['distance_long']], df[['TaxiTime']])
# Slope 
#print lm3.coef_
## Intercept
#print lm3.intercept_
#So: taxitime = 378.51251239 + 0.20695859*distance_long

lm4 = LinearRegression()
lm4.fit(df[['angle_sum']], df[['TaxiTime']])
# Slope 
#print lm4.coef_
## Intercept
#print lm4.intercept_
#So: taxitime = 223.51869963 + 0.67144551*angle_sum


lm5 = LinearRegression()
lm5.fit(df[['QDepDep']], df[['TaxiTime']])
# Slope 
#print lm5.coef_
## Intercept
#print lm5.intercept_
#So: taxitime = 413.53710849 + 123.64143635*QDepDep

lm6 = LinearRegression()
lm6.fit(df[['QDepArr']], df[['TaxiTime']])
# Slope 
#print lm6.coef_
## Intercept
#print lm6.intercept_
#So: taxitime = 508.4094549 + 160.32947531*QDepArr

lm7 = LinearRegression()
lm7.fit(df[['QArrDep']], df[['TaxiTime']])
# Slope 
#print lm7.coef_
## Intercept
#print lm7.intercept_
#So: taxitime = 594.76356181 - 198.13568547*QArrDep

lm8 = LinearRegression()
lm8.fit(df[['QArrArr']], df[['TaxiTime']])
# Slope 
#print lm8.coef_
## Intercept
#print lm8.intercept_
#So: taxitime = 583.0140771 - 240.5065771*QArrArr

lm9 = LinearRegression()
lm9.fit(df[['NDepDep']], df[['TaxiTime']])
# Slope 
#print lm9.coef_
## Intercept
#print lm9.intercept_
#So: taxitime = 455.52203996 + 87.98677698*NDepDep

lm10 = LinearRegression()
lm10.fit(df[['NDepArr']], df[['TaxiTime']])
# Slope 
#print lm10.coef_
## Intercept
#print lm10.intercept_
#So: taxitime = 555.6826137 + 94.5763766*NDepArr

lm11 = LinearRegression()
lm11.fit(df[['NArrDep']], df[['TaxiTime']])
# Slope 
#print lm11.coef_
## Intercept
#print lm11.intercept_
#So: taxitime = 617.93914332 - 166.06458255*NArrDep

lm12 = LinearRegression()
lm12.fit(df[['NArrArr']], df[['TaxiTime']])
# Slope 
#print lm12.coef_
## Intercept
#print lm12.intercept_
#So: taxitime = 586.46117337 - 297.55492337*NArrArr

lm13 = LinearRegression()
lm13.fit(df[['AvgSpdLast5Dep']], df[['TaxiTime']])
# Slope 
#print lm13.coef_
## Intercept
#print lm13.intercept_
#So: taxitime = 709.51741388 - 0.71909288*AvgSpdLast5Dep

lm14 = LinearRegression()
lm14.fit(df[['AvgSpdLast5Arr']], df[['TaxiTime']])
# Slope 
#print lm14.coef_
## Intercept
#print lm14.intercept_
#So: taxitime = 726.29340686 - 0.39967849*AvgSpdLast5Arr

lm15 = LinearRegression()
lm15.fit(df[['AvgSpdLast5']], df[['TaxiTime']])
# Slope 
#print lm15.coef_
## Intercept
#print lm15.intercept_
#So: taxitime = 778.46742131 - 0.82660697*AvgSpdLast5

lm16 = LinearRegression()
lm16.fit(df[['AvgSpdLast10Dep']], df[['TaxiTime']])
# Slope 
#print lm16.coef_
## Intercept
#print lm16.intercept_
#So: taxitime = 751.47183296 - 0.93423191*AvgSpdLast10Dep

lm17 = LinearRegression()
lm17.fit(df[['AvgSpdLast10Arr']], df[['TaxiTime']])
# Slope 
#print lm17.coef_
## Intercept
#print lm17.intercept_
#So: taxitime = 877.47227752 - 0.79470445*AvgSpdLast10Arr

lm18 = LinearRegression()
lm18.fit(df[['AvgSpdLast10']], df[['TaxiTime']])
# Slope 
#print lm18.coef_
## Intercept
#print lm18.intercept_
#So: taxitime = 839.74623619 - 1.0693648*AvgSpdLast10

lm19 = LinearRegression()
#lm19.fit(df[['Heavy_aircrafts']], df[['TaxiTime']])
# Slope 
#print lm19.coef_
## Intercept
#print lm19.intercept_
#So: taxitime = 560.01394642 + 115.70673869*Heavy_aircrafts

lm20 = LinearRegression()
#lm20.fit(df[['Medium_aircrafts']], df[['TaxiTime']])
# Slope 
#print lm20.coef_
## Intercept
#print lm20.intercept_
#So: taxitime = 675.32864137 - 131.11535411*Medium_aircrafts

lm21 = LinearRegression()
#lm21.fit(df[['Light_aircrafts']], df[['TaxiTime']])
# Slope 
#print lm21.coef_
# Intercept
#print lm21.intercept_
#So: taxitime = 558.22031996 + 116.76825147*Light_aircrafts

lm22 = LinearRegression()
lm22.fit(df[['aircraft_size']], df[['TaxiTime']])
# Slope 
#print lm22.coef_
## Intercept
#print lm22.intercept_
#So: taxitime = 966.14269237 - 1.61823971*aircraft_size

#  Multiple Linear Regression
lm23 = LinearRegression()
Z = df[['depArr', 'angle_error', 'distance_long', 'angle_sum', 'AvgSpdLast5Dep','QDepDep', 'QDepArr', 'QArrDep', 'QArrArr', 'NDepDep', 'NArrArr', 'AvgSpdLast5Dep', 'AvgSpdLast5Arr', 'AvgSpdLast5', 'AvgSpdLast10Dep', 'AvgSpdLast10Arr', 'AvgSpdLast10','aircraft_size']]
lm23.fit(Z, df['TaxiTime'])
## Slope
#print lm23.coef_
## Intercept
#print lm23.intercept_
##TaxiTime = 654.534314167 + 1.73155472e+01 x depArr + ( -1.90659946e+00) x distance + (3.73307918e+01) x angle_error + (1.17381459e-01) x distance_long......
#
#width = 12
#height = 10
#plt.figure(figsize=(width, height))
#sns.residplot(df['angle_sum'], df['TaxiTime'])
#plt.show()


'''
##Visualisation Multiple Linear Regression
Yhat1 = lm23.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.distplot(Yhat1, hist=False, color="b", label="Fitted Values")
sns.distplot(df['TaxiTime'], hist=False, color="r", label="Actual Value" , ax=ax1)


plt.title('Actual vs Fitted Values for Taxi Time')
plt.xlabel('Taxi time')
plt.ylabel('')

plt.show()
plt.close()
'''

'''  
##Part 3: Polynomial Regression and Pipelines
def PlotPolly(model,independent_variable,dependent_variabble, Name):
    x_new = np.linspace(10, 1000, 2000)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Taxi Time ~ angle_sum')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Taxi Time')

    plt.show()
    plt.close()
  
x = df['angle_sum']
y = df['TaxiTime']
## Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPolly(p,x,y, 'angle_sum')
np.polyfit(x, y, 3)
'''
'''
#Create 11 order polynomial model with the variables x and y from above
# calculate polynomial
# Here we use a polynomial of the 3rd order (cubic) 
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'angle_sum')
'''  
'''
#We create a PolynomialFeatures object of degree 2: 
pr=PolynomialFeatures(degree=2)
Z = df[['depArr', 'angle_error', 'distance_long', 'angle_sum', 'AvgSpdLast5Dep','QDepDep', 'QDepArr', 'QArrDep', 'QArrArr', 'NDepDep', 'NArrArr', 'AvgSpdLast5Dep', 'AvgSpdLast5Arr', 'AvgSpdLast5', 'AvgSpdLast10Dep', 'AvgSpdLast10Arr', 'AvgSpdLast10','Average_Speed', 'Light_aircrafts','Medium_aircrafts','Heavy_aircrafts']]
Z_pr=pr.fit_transform(Z)
print Z.shape
print Z_pr.shape
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
print ypipe[0:4]

Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
print ypipe[0:10]
'''
'''
#Part 4: Measures for In-Sample Evaluation
#Model 1: Simple Linear Regression
# Find the R^2
print lm.score(df[['depArr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'depArr'."
Yhat=lm.predict(df[['depArr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm1.score(df[['distance']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'distance'."
Yhat=lm.predict(df[['distance']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm2.score(df[['angle_error']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'angle_error'."
Yhat=lm.predict(df[['angle_error']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)


print lm3.score(df[['distance_long']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'distance_long'."
Yhat=lm.predict(df[['distance_long']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm4.score(df[['angle_sum']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'angle_sum'."
Yhat=lm.predict(df[['angle_sum']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm5.score(df[['QDepDep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'QDepDep'."
Yhat=lm.predict(df[['QDepDep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm6.score(df[['QDepArr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'QDepArr'."
Yhat=lm.predict(df[['QDepArr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm7.score(df[['QArrDep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'QArrDep'."
Yhat=lm.predict(df[['QArrDep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm8.score(df[['QArrArr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'QArrArr'."
Yhat=lm.predict(df[['QArrArr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm9.score(df[['NDepDep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'NDepDep'."
Yhat=lm.predict(df[['NDepDep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm10.score(df[['NDepArr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'NDepArr'."
Yhat=lm.predict(df[['NDepArr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm11.score(df[['NArrDep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'NArrDep'."
Yhat=lm.predict(df[['NArrDep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm12.score(df[['NArrArr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'NArrArr'."
Yhat=lm.predict(df[['NArrArr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm13.score(df[['AvgSpdLast5Dep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast5Dep'."
Yhat=lm.predict(df[['AvgSpdLast5Dep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm14.score(df[['AvgSpdLast5Arr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast5Arr'."
Yhat=lm.predict(df[['AvgSpdLast5Arr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm15.score(df[['AvgSpdLast5']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast5'."
Yhat=lm.predict(df[['AvgSpdLast5']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm16.score(df[['AvgSpdLast10Dep']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast10Dep'." 
Yhat=lm.predict(df[['AvgSpdLast10Dep']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm17.score(df[['AvgSpdLast10Arr']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast10Arr'."
Yhat=lm.predict(df[['AvgSpdLast10Arr']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm18.score(df[['AvgSpdLast10']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'AvgSpdLast10'."
Yhat=lm.predict(df[['AvgSpdLast10']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm19.score(df[['Heavy_aircrafts']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'Heavy_aircrafts'."
Yhat=lm.predict(df[['Heavy_aircrafts']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm20.score(df[['Medium_aircrafts']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'Medium_aircrafts'."
Yhat=lm.predict(df[['Medium_aircrafts']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm21.score(df[['Light_aircrafts']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'Light_aircrafts'."
Yhat=lm.predict(df[['Light_aircrafts']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

print lm21.score(df[['Average_Speed']], df[['TaxiTime']]), " of the variation of the TaxiTime is explained by this simple linear model 'Average_Speed'."
Yhat=lm.predict(df[['Average_Speed']])
#mean_squared_error(Y_true, Y_predict)
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Yhat)

#Model 2: Multiple Linear Regression

# Find the R^2
print lm23.score(Z, df['TaxiTime']), " of the variation of TaxiTime is explained by this multiple linear regression 'multi_fit'."
Y_predict_multifit = lm23.predict(Z)
#mean_squared_error
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], Y_predict_multifit)
'''
'''
# Model 3: Polynomial Fit
#We apply the function to get the value of r^2
r_squared = r2_score(y, p(x))
print r_squared, 'of the variation of price is explained by this polynomial fit'
print "the mean squared error is: ", mean_squared_error(df['TaxiTime'], p(x))

#Part 5: Prediction and Decision Making
#Prediction
#Create a  new input 
new_input=np.arange(1,100,1).reshape(-1,1)
lm.fit(X, Y)
print lm
yhat=lm.predict(new_input)
print yhat[0:5]
plt.plot(new_input,yhat)
plt.show()
'''
######################################################################################################################################################
#Modelling Part
#Model Evaluation and Refinement 
def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title ):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName)
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName, ax=ax2)
    plt.title(Title)
    plt.xlabel('Taxi Time (in seconds)')
    plt.ylabel('Proportion of aircrafts')

    plt.show()
    plt.close()

def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(),xtest.values.max()])

    xmin=min([xtrain.values.min(),xtest.values.min()])

    x=np.arange(xmin,xmax,0.1)


    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel('Taxi Time')
    plt.legend()

y_data=df['TaxiTime']   #data with only the TaxiTime
x_data=df.drop('TaxiTime',axis=1)   #dataset without the TaxiTime
x_data.to_csv('x_data.csv')
#y_data.to_csv('y_data.csv')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)
#xy_train = x_train
#xy_train['TaxiTime'] = y_train.values
#print xy_train
#xy_test = x_test
#xy_test['TaxiTime']= y_test.values
#xy_test.to_csv('test_air.csv')
#xy_train.to_csv('train_air.csv')


#print("number of test samples :", x_test.shape[0])
#print("number of training samples:",x_train.shape[0])


#Lasso Regression
#select predictor variables and target variable as separate data sets  
predvar= df[['depArr','distance','angle_error','distance_long','angle_sum','QDepDep',
'QDepArr','QArrDep','QArrArr','NDepDep','NDepArr','NArrDep','NArrArr','Pressure',
'VisibilityInMeters','TemperatureInCelsius','WindSpeedInMPS','isRain','isSnow','isDrizzle','isFog','isMist',
'AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','aircraft_size']]

# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
predictors['depArr']=preprocessing.scale(predictors['depArr'].astype('float64'))    #transform the data in order to have mean = 0 and std = 1
predictors['distance']=preprocessing.scale(predictors['distance'].astype('float64'))
predictors['angle_error']=preprocessing.scale(predictors['angle_error'].astype('float64'))
predictors['distance_long']=preprocessing.scale(predictors['distance_long'].astype('float64'))
predictors['angle_sum']=preprocessing.scale(predictors['angle_sum'].astype('float64'))
predictors['QDepDep']=preprocessing.scale(predictors['QDepDep'].astype('float64'))
predictors['QDepArr']=preprocessing.scale(predictors['QDepArr'].astype('float64'))
predictors['QArrDep']=preprocessing.scale(predictors['QArrDep'].astype('float64'))
predictors['QArrArr']=preprocessing.scale(predictors['QArrArr'].astype('float64'))
predictors['NDepDep']=preprocessing.scale(predictors['NDepDep'].astype('float64'))
predictors['NDepArr']=preprocessing.scale(predictors['NDepArr'].astype('float64'))
predictors['NArrDep']=preprocessing.scale(predictors['NArrDep'].astype('float64'))
predictors['NArrArr']=preprocessing.scale(predictors['NArrArr'].astype('float64'))
predictors['Pressure']=preprocessing.scale(predictors['Pressure'].astype('float64'))
predictors['VisibilityInMeters']=preprocessing.scale(predictors['VisibilityInMeters'].astype('float64'))
predictors['TemperatureInCelsius']=preprocessing.scale(predictors['TemperatureInCelsius'].astype('float64'))
predictors['WindSpeedInMPS']=preprocessing.scale(predictors['WindSpeedInMPS'].astype('float64'))
predictors['isRain']=preprocessing.scale(predictors['isRain'].astype('float64'))
predictors['isSnow']=preprocessing.scale(predictors['isSnow'].astype('float64'))
predictors['isDrizzle']=preprocessing.scale(predictors['isDrizzle'].astype('float64'))
predictors['isFog']=preprocessing.scale(predictors['isFog'].astype('float64'))
predictors['isMist']=preprocessing.scale(predictors['isMist'].astype('float64'))
predictors['AvgSpdLast5Dep']=preprocessing.scale(predictors['AvgSpdLast5Dep'].astype('float64'))
predictors['AvgSpdLast5Arr']=preprocessing.scale(predictors['AvgSpdLast5Arr'].astype('float64'))
predictors['AvgSpdLast5']=preprocessing.scale(predictors['AvgSpdLast5'].astype('float64'))
predictors['AvgSpdLast10Dep']=preprocessing.scale(predictors['AvgSpdLast10Dep'].astype('float64'))
predictors['AvgSpdLast10Arr']=preprocessing.scale(predictors['AvgSpdLast10Arr'].astype('float64'))
predictors['AvgSpdLast10']=preprocessing.scale(predictors['AvgSpdLast10'].astype('float64'))
predictors['aircraft_size']=preprocessing.scale(predictors['aircraft_size'].astype('float64'))
#predictors['Average_Speed']=preprocessing.scale(predictors['Average_Speed'].astype('float64'))
#predictors['Heavy_aircrafts']=preprocessing.scale(predictors['Heavy_aircrafts'].astype('float64'))
#predictors['Light_aircrafts']=preprocessing.scale(predictors['Light_aircrafts'].astype('float64'))
#predictors['Medium_aircrafts']=preprocessing.scale(predictors['Medium_aircrafts'].astype('float64'))

def min_percent(y_pred, y_test):
    error = abs(y_pred - y_test)    #subtraction of the actual and predicted values in seconds
    error = error/60    #convert error to minutes, because y_pred and y_test are in seconds
    #counters
    count1 = 0
    count2 = 0
    count3 = 0
    for i in error:
        if i <= 1:  #if lower than 1 minute
            count1 =count1 +1   
        elif i <= 3:    #if lower than 3 minutes
            count2 = count2 + 1
        elif i <= 5:    #if lower than 5 minutes
            count3  = count3 +1
    count2 = count2+count1  #lower than 3 minutes is the lower than 1 and 3
    count3 = count3+count2  #lower than 5 minutes is lower than 1, 3 and 5
    #percentages of the values that are lower than the monutes that are set
    per1 = 100*count1/len(error)    
    per2 = 100*count2/len(error)
    per3 = 100*count3/len(error)
    percentage = [per1,per2,per3]   #list of percentages
    return percentage

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, y_data, 
                                                              test_size=.3, random_state=123)

#Normalize the data
normalized_df=(df-df.mean())/df.std()   #formula for normalisation
normalized_df = normalized_df.drop('isMist', 1) #remove Mist
normalized_df = normalized_df.drop('isSnow', 1) #remove Snow
x_data2=normalized_df.drop('TaxiTime',axis=1)
pred2_train, pred2_test, tar2_train, tar2_test = train_test_split(x_data2, y_data, 
                                                              test_size=.3, random_state=123)
#print pred2_train.columns

# specify the lasso regression model
def LASSOLARS(pred_train, tar_train, pred_test, tar_test):  #function for LASSO, where we put the training data, the corresponding output value, the test data and the corresponding output value
    model = LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)  #fit LASSO LARS model with training data
    # print variable names and regression coefficients
    print dict(zip(predictors.columns, model.coef_))    #print dictionary with columns as variables and regression coefficient
    d = dict(zip(predictors.columns, model.coef_))
    s = pd.DataFrame(d.items(), columns=['Variable', 'Value'])  #convert the dictionary into data frame for easier readibility
    print s
    #s.to_csv('lassoLars.csv')  #s to csv file

    # plot coefficient progression
    m_log_alphas = -np.log10(model.alphas_) #alpha (lambda) of LASSO LARS
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.xlabel('-log(alpha)')
    plt.title('Regression Coefficients Progression for Lasso Paths')

    # plot mean square error for each fold
    m_log_alphascv = -np.log10(model.cv_alphas_)
    plt.figure()
    plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
    plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean squared error')
    plt.title('Mean squared error on each fold')
         

    # MSE from training and test data

    train_error = mean_squared_error(tar_train, model.predict(pred_train))  #train mean squared error
    test_error = mean_squared_error(tar_test, model.predict(pred_test)) #test mean squared error
    print ('training data MSE')
    print(train_error)
    print ('test data MSE')
    print(test_error)

    # R-square from training and test data
    rsquared_train=model.score(pred_train,tar_train)
    rsquared_test=model.score(pred_test,tar_test)
    print ('training data R-square')
    print(rsquared_train)
    print ('test data R-square')
    print(rsquared_test)

    y_lasso = model.predict(pred_test)
    #Figure 2: Plot of predicted value compared to the actual value using the test data.
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data Lasso'
    DistributionPlot(tar_test,y_lasso,"Actual Values (Test)","Predicted Values (Test)",Title)
    #print lr.score(Z, df['TaxiTime']), " of the variation of TaxiTime is explained by this multiple linear regression 'multi_fit'."
    # Calculate the absolute errors
    errors = abs(y_lasso - tar_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error for multiple linear regression:', round(np.mean(errors), 2), 'degrees.'
    print 'Mean Absolute error', mean_absolute_error(tar_test, y_lasso)
    print 'Mean Squared error', mean_squared_error(tar_test, y_lasso)  
    print 'Root Mean Squared error', sqrt(mean_squared_error(tar_test, y_lasso))
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / tar_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'

    #precentages of values that have deviation from the actual values less than 1, 3 and 5 minutes
    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_lasso,tar_test)[0],"%, <= 3 minutes are ", min_percent(y_lasso,tar_test)[1],"% and <= 5 are ", min_percent(y_lasso,tar_test)[2],"% of the predicted values"
#LASSOLARS(pred_train, tar_train, pred_test, tar_test)  #use the function for LASSO LARS

'''
lre=LinearRegression()
lre.fit(x_train[['depArr']],y_train)
print "R^2 for test data", lre.score(x_test[['depArr']],y_test)
print "R^2 for training data", lre.score(x_train[['depArr']],y_train)

#Cross-validation Score
Rcross=cross_val_score(lre,x_train, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
print"The mean of the folds are", Rcross.mean(),"and the standard deviation is" ,Rcross.std()
yhat=cross_val_predict(lre,x_train, y_train,cv=10)
print yhat[0:5]
'''
'''
#Part 2: Overfitting, Underfitting and Model Selection
lr=LinearRegression()
lr.fit(x_train,y_train)

yhat_train=lr.predict(x_train)
print 'Prediction using training data: ', yhat_train[0:5]
yhat_test=lr.predict(x_test)
print 'Prediction using test data: ', yhat_test[0:5]
#Figure 1: Plot of predicted values using the training data compared to the training data. 
Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title)
#Figure 2: Plot of predicted value compared to the actual value using the test data.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
#print lr.score(Z, df['TaxiTime']), " of the variation of TaxiTime is explained by this multiple linear regression 'multi_fit'."
print 'Mean Absolute error', mean_absolute_error(y_test, yhat_test)
print 'Mean Squared error', mean_squared_error(y_test, yhat_test) 
print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat_test)) 
print "R^2 for test data", lr.score(x_test,y_test)
print "R^2 for training data", lr.score(x_train,y_train)
'''
def Linear_Reg(x_train, y_train, x_test, y_test):   #import training and test data
    lr=LinearRegression()   #Multiple Linear Regression model
    lr.fit(x_train,y_train) #fit the model
    Rcross=cross_val_score(lr,x_train, y_train,cv=10)    #We input the object, the features, the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", Rcross.mean(),"and the standard deviation is" ,Rcross.std()
    yhat=cross_val_predict(lr,x_test, y_test,cv=10) #taxi time predictions

    #Figure 2: Plot of predicted value compared to the actual value using the test data.
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)
    #print lr.score(Z, df['TaxiTime']), " of the variation of TaxiTime is explained by this multiple linear regression 'multi_fit'."
    # Calculate the absolute errors
    errors = abs(yhat - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error for multiple linear regression:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    feature_list = list(df.columns)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", lr.score(x_test,y_test)
    print "R^2 for training data", lr.score(x_train,y_train)
    print 'Mean Absolute error', mean_absolute_error(y_test, yhat)
    print 'Mean Squared error', mean_squared_error(y_test, yhat) 
    print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat))  
#Linear_Reg(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test)

#Function that gives the best degree for polynomial regression
def Pol_degree(x_train, y_train, x_test, y_test):   #import training and test data
    Rsqu_test=[]
    lr=LinearRegression()
    order=[1,2,3]   #list with the degrees that are checked
    for n in order: #n takes the values of order list
        pr=PolynomialFeatures(degree=n) #polynomial model
    
        x_train_pr=pr.fit_transform(x_train)    #Fit to data, then transform it.
    
        x_test_pr=pr.fit_transform(x_test)    
    
        lr.fit(x_train_pr,y_train)  #fit the linear model with the trasformed data
    
        Rsqu_test.append(lr.score(x_test_pr,y_test))    #append the results of the test data in a list
    print Rsqu_test
    plt.plot(order,Rsqu_test)
    plt.xlabel('order')
    plt.ylabel('R^2')
    plt.title('R^2 Using Test Data')
    plt.text(3, 0.75, 'Maximum R^2 ')  
#Pol_degree(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test)

#Polynomial regression
def PolReg(x_train, y_train, x_test, y_test,degree2):
    pr=PolynomialFeatures(degree=2)
    x_train_pr=pr.fit_transform(x_train)
    x_test_pr=pr.fit_transform(x_test)
    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)
#    print x_train_pr
    print pr.get_feature_names(x_train.columns)
    #Cross-validation Score
    Rcross=cross_val_score(poly,x_train_pr, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", Rcross.mean(),"and the standard deviation is" ,Rcross.std()
    yhat=cross_val_predict(poly,x_test_pr, y_test,cv=10)
    #yhat=poly.predict(x_test_pr )
    print y_test.values
    print"Predicted values:", yhat[0:4]
    print"True values:",y_test[0:4].values
    lytest = y_test.values.tolist()
    print "R^2 for test data", poly.score(x_test_pr,y_test)
    print "R^2 for training data", poly.score(x_train_pr,y_train)
    print 'Mean Absolute error', mean_absolute_error(y_test, yhat)
    print 'Mean Squared error', mean_squared_error(y_test, yhat)
    print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat)) 
    print 'Relative absolute error', mean_absolute_error(y_test, yhat)

    # Calculate the absolute errors
    errors = abs(yhat - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
#    print poly.coef_
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)
#PolReg(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test,2)

#Part 3: Ridge Regression
def Opt_Ridge_Pol_Reg(x_train, y_train, x_test, y_test,degree2):    #import training, test data and degree
    pr=PolynomialFeatures(degree=degree2)   #Polynomial algorithm with the degree that was set in the function
    x_train_pr=pr.fit_transform(x_train)    #transform the training data into polynomial form
    x_test_pr=pr.fit_transform(x_test)  #transform the test data into polynomial form
    print x_train_pr.shape  #print the shape of transformed training data
    RigeModel=Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)  #create a Ridge regression object, setting the regularization parameter to 0.001     
    RigeModel.fit(x_train_pr,y_train)   #fit the model using the method fit
    yhat=RigeModel.predict(x_test_pr)   #obtain a prediction
    print'predicted:', yhat[0:4]    #print the first 4 predicted values
    print'test set :', y_test[0:4].values   #print the first 4 values of test data taxi time

    # Calculate the absolute errors
    errors = abs(yhat - y_test) #the absolute value of the deviation between predicted and actual value
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat)) 

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data",  round(RigeModel.score(x_test_pr,y_test), 2)
    print "R^2 for training data",  round(RigeModel.score(x_train_pr,y_train), 2)
    print "The predicted taxi time that is lower than 1 minute deviation from the actual taxi time is", min_percent(yhat,y_test)[0],"%, lower than 3 minutes ", min_percent(yhat,y_test)[1],"% and lower than 5 minutes is", min_percent(yhat,y_test)[2],"% of the predicted values"
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)
#Opt_Ridge_Pol_Reg(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test,2)


#Ridge Regression for Multiple Linear
def Mult_Linear_Ridge(x_train, y_train, x_test, y_test):   #import training, test data and degree 
    lr=LinearRegression()   #the algorithm for liner regression
    lr.fit(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']],y_train)  #fit the linear model
    print x_train.shape
    RigeModel=Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)  #create a Ridge regression object, setting the regularization parameter to 10
    RigeModel.fit(x_train,y_train)   #fit the model using the method fit
    yhat=RigeModel.predict(x_test)   #obtain a prediction
    print'predicted:', yhat[0:4]
    print'test set :', y_test[0:4].values

    # Calculate the absolute errors
    errors = abs(yhat - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'


    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data",  round(RigeModel.score(x_test,y_test), 2)
    print "R^2 for training data",  round(RigeModel.score(x_train,y_train), 2)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat))
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    print "The predicted taxi time that is lower than 1 minute deviation from the actual taxi time in percentage is", min_percent(yhat,y_test)[0],"%, lower than 3 minutes ", min_percent(yhat,y_test)[1],"% and lower than 5 minutes is", min_percent(yhat,y_test)[2],"% of the predicted values"
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)
#Mult_Linear_Ridge(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test)


#Part 4: Grid Search
def Grid_Search_Ridge(x_train, y_train, x_test, y_test):    #import training and test data
    parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000], 'normalize': [True,False]}]   #the parameters and their values that are checked
    RR=Ridge()  #ridge model
    Grid1 = GridSearchCV(RR, parameters1,cv=10)  #Create a ridge grid search objec
    Grid1.fit(x_train,y_train)   #Fit the model
    BestRR=Grid1.best_estimator_    #The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR 
    print "The best parameter values", BestRR
    scores = Grid1.cv_results_  #results of grid search - parameters, r-squared for training and test data
    for param,mean_val,mean_test in zip(scores['params'],scores['mean_test_score'],scores['mean_train_score']): #loop with the values of the results
        print param, "R^2 on train data: ", mean_val, "R^2 on test data: ", mean_test
    print"By testing our model the R^2 on the test data: ", BestRR.score(x_test,y_test)
#Grid_Search_Ridge(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test)


from sklearn import linear_model
def LASSO(pred_train,tar_train,pred_test,tar_test): #import training and test data
    reg = linear_model.Lasso(alpha=0.1) #Linear model with lasso regression and regularisation parameter lambda (alpha) equal to 0.1
    reg.fit(pred_train, tar_train)  #fit the model
    y_pred = reg.predict(pred_test) #prediction values
    print y_pred    #print the prediction values
    # Calculate the absolute errors
    errors = abs(y_pred - tar_test)
    # Print out the mean absolute error (mae)

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / tar_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", reg.score(pred_test,tar_test)
    print "R^2 for training data", reg.score(pred_train,tar_train)
    print 'Mean Absolute error', mean_absolute_error(tar_test, y_pred)
    print 'Mean Squared error', mean_squared_error(tar_test, y_pred)
    print 'Root Mean Squared error', sqrt(mean_squared_error(tar_test, y_pred))
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(tar_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)
#LASSO(pred_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']],tar_train,pred_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']],tar_test)

def Mult_Lin_Ridge(x_train, y_train, x_test, y_test):   #import training and test data
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0,100.0, 1000.0])   #this checks the alpha parameter (easier way)
    reg.fit(x_train,y_train)       #fit the model
    y_pred = reg.predict(x_test)    #prediction values
#    print y_pred
    print 'the alpha parameter: ', reg.alpha_                                      

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", reg.score(x_test,y_test)
    print "R^2 for training data", reg.score(x_train,y_train)
    print 'Mean Absolute error', mean_absolute_error(y_test, y_pred)
    print 'Mean Squared error', mean_squared_error(y_test, y_pred)
    print "The predicted taxi time that is lower than 1 minute deviation from the actual taxi time in percentage is", min_percent(y_pred,y_test)[0],"%, lower than 3 minutes ", min_percent(y_pred,y_test)[1],"% and lower than 5 minutes is", min_percent(y_pred,y_test)[2],"% of the predicted values"
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)
#Mult_Lin_Ridge(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_train, x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog']], y_test)

###################################################################################################################################################################

#Random forest
#print x_train.shape
#print x_test.shape
#print y_train.shape
#print y_test.shape
from sklearn.ensemble import RandomForestRegressor

def Random_Forest(x_train, y_train, x_test, y_test):    #import training and test data
    #Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(random_state = 3, max_depth=None, max_features='auto', min_samples_leaf=10, n_estimators=150) #Random Forest with the hyper-parameters that were set
    regressor.fit(x_train,y_train)  #fit the model
    #Cross-validation Score
    Rcross=cross_val_score(regressor,x_train, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", round(Rcross.mean(),2),"and the standard deviation is" ,round(Rcross.std(),2)
    y_pred=cross_val_predict(regressor,x_test, y_test,cv=10)    #make predictions with 10-fold cross validation 
    #print regressor.feature_importances_
    print y_pred    #print the predicted values

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    feature_list = list(df.columns) #list with the variable names
    feature_list.remove('TaxiTime') #remove the TaxiTime from the list with the variable names
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", round(regressor.score(x_test,y_test), 2)
    print "R^2 for training data", round(regressor.score(x_train,y_train), 2)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared Error', round(sqrt(mean_squared_error(y_test, y_pred)), 2)
    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_pred,y_test)[0],"%, <= 3 minutes are ", min_percent(y_pred,y_test)[1],"% and <= 5 are ", min_percent(y_pred,y_test)[2],"% of the predicted values"
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)

    #print "Features sorted by their score:", sorted(zip(map(lambda x_train: round(x_train, 4), regressor.feature_importances_), feature_list), 
    #             reverse=True)

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    ## Print out the feature and importances (in ordee to get a table with the variables and their importances remove the hashtags)
    #for pair in feature_importances:
    #    print('Variable: {:20} Importance: {}'.format(*pair)) 

#Random Forest , which gives the average accuracy mean abdolute error and root mean squared error, from the results of the model with 30 different random_states    
def Random_Forest2(x_train, y_train, x_test, y_test):   #import training and test data
    #create list that will take the values for accuracy, mean absolute error and root mean squared error
    list_acc = []
    list_mae =[]
    list_rmse = []
    for rand in range(1,31):    #loop that takes the value from 1 to 30 that are the values for rndom_state
        #Fitting Random Forest Regression to the dataset
        regressor = RandomForestRegressor(random_state = rand, max_depth=None, max_features='auto', min_samples_leaf=10, n_estimators=150)  #Random forest algorithm the the set hyper-parameters
        regressor.fit(x_train,y_train)  #fit the model
        #Cross-validation Score
        Rcross=cross_val_score(regressor,x_train, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
#       print"The mean of the folds is", round(Rcross.mean(),2),"and the standard deviation is" ,round(Rcross.std(),2)
        y_pred=cross_val_predict(regressor,x_test, y_test,cv=10)    #make predictions with 10-fold cross validation
        #print yhat[0:5]
        #y_pred = regressor.predict(x_test) #Predicting a new result
        #print regressor.feature_importances_
#        print y_pred

        # Calculate the absolute errors
        errors = abs(y_pred - y_test)
        # Print out the mean absolute error (mae)
#       print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / y_test)
        feature_list = list(df.columns)
        feature_list.remove('TaxiTime')
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        list_acc.append(accuracy)   #append the accuracy that was found to the list_acc
        list_mae.append(np.mean(errors))    #append the mean absolute error that was found to the list_mae
        list_rmse.append(sqrt(mean_squared_error(y_test, y_pred)))  #append the root mean squared error that was found to the list_rmse
        print'Accuracy:', round(accuracy, 2), '%.'
#        print "R^2 for test data", round(regressor.score(x_test,y_test), 2)
#        print "R^2 for training data", round(regressor.score(x_train,y_train), 2)
#        print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
#        print 'Root Mean Squared Error', round(sqrt(mean_squared_error(y_test, y_pred)), 2)
#        print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_pred,y_test)[0],"%, <= 3 minutes are ", min_percent(y_pred,y_test)[1],"% and <= 5 are ", min_percent(y_pred,y_test)[2],"% of the predicted values"
#        Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#        DistributionPlot(y_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)

        #print "Features sorted by their score:", sorted(zip(map(lambda x_train: round(x_train, 4), regressor.feature_importances_), feature_list), 
        #             reverse=True)

        # Get numerical feature importances
        importances = list(regressor.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        ## Print out the feature and importances 
        #for pair in feature_importances:
        #    print('Variable: {:20} Importance: {}'.format(*pair)) 
    average_acc = sum(list_acc) / float(len(list_acc))  #calculate the average of accuracy that was found with the 30 different random_states
    average_mae = sum(list_mae) / float(len(list_mae))  #calculate the average of mean absolute error that was found with the 30 different random_states
    average_rmse = sum(list_rmse) / float(len(list_rmse))   #calculate the average of root mean squared error that was found with the 30 different random_states
    print 'the average accuracy of 30 seed is: ', round(average_acc,2), '%'
    print 'the average Mean Absolute Error of 30 seed is: ', round(average_mae,2)
    print 'the average accuracy of 30 seed is: ', round(average_rmse,2)
#Random_Forest(x_train[['QDepDep', 'depArr', 'distance', 'angle_sum','AvgSpdLast5Dep','AvgSpdLast10Dep','AvgSpdLast10','distance_long','QDepArr', 'NDepDep', 'Pressure', 'TemperatureInCelsius', 'WindSpeedInMPS', 'AvgSpdLast5Arr', 'AvgSpdLast5', 'AvgSpdLast10Arr', 'aircraft_size', 'angle_error']], y_train, x_test[['QDepDep', 'depArr', 'distance', 'angle_sum','AvgSpdLast5Dep','AvgSpdLast10Dep','AvgSpdLast10','distance_long','QDepArr', 'NDepDep', 'Pressure', 'TemperatureInCelsius', 'WindSpeedInMPS', 'AvgSpdLast5Arr', 'AvgSpdLast5', 'AvgSpdLast10Arr', 'aircraft_size', 'angle_error']], y_test)


'''
from sklearn.ensemble import RandomForestRegressor
#Grid Search (it was not because it was time consuming)
def evaluate(model, test_features, test_labels):    #function that finds the accuracy of the model
    predictions = model.predict(test_features)  #make predictions of the model
    errors = abs(predictions - test_labels)     #calculate the error
    mape = 100 * np.mean(errors / test_labels) #mean absolute percent error
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(x_train[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_train)
base_accuracy = evaluate(base_model, x_test[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_test)

# Create the parameter grid based on the results of random search 
param_grid1 = {
    'bootstrap': [True],
    'max_depth': [80],
    'max_features': [2],
    'min_samples_leaf': [3],
    'min_samples_split': [8],
    'n_estimators': [100]
    
}

# Create a based model
rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid1, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(x_train[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_train)

print grid_search.best_params_


best_grid = grid_search.best_estimator_
print best_grid
grid_accuracy = evaluate(best_grid, x_test[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_test)
print 'grid accuracy', grid_accuracy
#Model Performance
#Average Error: 3.6561 degrees.
#Accuracy = 93.83%.

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
'''
'''
#Grid Search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(x_train[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_train)
base_accuracy = evaluate(base_model, x_test[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_test)

# Create the parameter grid based on the results of random search (Again it was running for too long)
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Create a based model
#rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(x_train[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_train)

print grid_search.best_params_


best_grid = grid_search.best_estimator_
print best_grid
grid_accuracy = evaluate(best_grid, x_test[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_test)
print 'grid accuracy', grid_accuracy
#Model Performance
#Average Error: 3.6561 degrees.
#Accuracy = 93.83%.

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
'''
'''
#2nd way to do grid search
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]    #checks the n_estimators from 200 to 2000 by increasing the value for 10 every time
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]    #ckecks max_septh from 10 to 110 by increasing the value for 10 every time
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print random_grid
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(x_train[['depArr','QDepDep','distance','angle_sum','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5','Medium_aircrafts','angle_error','Light_aircrafts']], y_train)
print "The best hyper-parameters are: ", rf_random.best_params_
'''
#######################################################################################################################################################

#Neural Networks


#Variable Selection with Variance Threshold
from sklearn.feature_selection import VarianceThreshold
def Var_Thershold(pred2_train, pred2_test): #import normalised training and test
    # Arbitrarily set threshold to 0.1
    sel = VarianceThreshold(threshold = 0.1)
    print pred2_train
    sel.fit(pred2_train)
    threshold=0.1
    idx = np.where(sel.variances_ > threshold)[0]
    print idx
    list_col =  pred2_train.columns
    print list_col[idx]
    print pred2_train.shape
    # Subset features
    x_new = sel.transform(pred2_train)
    x_test_new = sel.transform(pred2_test)
    print x_new.shape
    #print x_new
#Var_Thershold(pred2_train, pred2_test)

#Multilayer Perceptron
from sklearn.neural_network import MLPRegressor

def MLP(pred2_train, tar2_train, pred2_test, tar2_test):    #import training and test data
    clf = MLPRegressor(random_state=17, activation = 'identity', hidden_layer_sizes = 100, learning_rate_init = 0.001, max_iter = 200)  #Multilayer perceptron model with the set hyper-parameters
    clf.fit(pred2_train, tar2_train)    #fit the model with traing data
    #Cross-validation Score
    Rcross=cross_val_score(clf,pred2_train, tar2_train)
    #Cross-validation Score, tar2_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", round(Rcross.mean(), 2),"and the standard deviation is" ,round(Rcross.std(), 2)
    y_pred=cross_val_predict(clf,pred2_test, tar2_test,cv=10)
    #y_pred = clf.predict(x_test) #Predicting a new result
    print y_pred[0:5]
    #print clf.coefs_

    # Calculate the absolute errors
    errors = abs(y_pred - tar2_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / tar2_test)
    feature_list = list(df.columns)
    feature_list.remove('TaxiTime')
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", round(clf.score(pred2_test, tar2_test), 2)
    print "R^2 for training data", round(clf.score(pred2_train, tar2_train), 2)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared Error', round(sqrt(mean_squared_error(tar2_test, y_pred)), 2)
    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_pred,tar2_test)[0],"%, <= 3 minutes are ", min_percent(y_pred,tar2_test)[1],"% and <= 5 are ", min_percent(y_pred,tar2_test)[2],"% of the predicted values"
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(tar2_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)

#Multilayer Perceptron , which gives the average accuracy mean abdolute error and root mean squared error, from the results of the model with 30 different random_states      
def MLP2(pred2_train, tar2_train, pred2_test, tar2_test):
    #create list that will take the values for accuracy, mean absolute error and root mean squared error
    list_acc = []
    list_mae =[]
    list_rmse = []
    for rand in range(1,31):
        clf = MLPRegressor(random_state=rand, activation = 'identity', hidden_layer_sizes = (100,10,20))
        clf.fit(pred2_train, tar2_train) 
        #Best hyperparameters: hidden_layer_sizes = (100,100,10), activation = 'identity', solver = 'adam',alpha = 0.5, learning_rate  = 'invscaling',momentum = 0.87, learning_rate_init = 0.002, max_iter = 200
        #[['depArr','distance','angle_error','distance_long','angle_sum','QDepDep','QDepArr','QArrDep','NDepDep','NDepArr','NArrDep','VisibilityInMeters','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr', 'AvgSpdLast10', 'aircraft_size']]
        #Cross-validation Score
        Rcross=cross_val_score(clf,pred2_train, tar2_train)
        #Cross-validation Score, tar2_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
#        print"The mean of the folds is", round(Rcross.mean(), 2),"and the standard deviation is" ,round(Rcross.std(), 2)
        y_pred=cross_val_predict(clf,pred2_test, tar2_test,cv=10)
        #y_pred = clf.predict(x_test) #Predicting a new result
#        print y_pred[0:5]
        #print clf.coefs_

        # Calculate the absolute errors
        errors = abs(y_pred - tar2_test)
        # Print out the mean absolute error (mae)
#        print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / tar2_test)
        feature_list = list(df.columns)
        feature_list.remove('TaxiTime')
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print'Accuracy:', round(accuracy, 2), '%.'
        list_acc.append(accuracy)
        list_mae.append(np.mean(errors))
        list_rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
#        print "R^2 for test data", round(clf.score(pred2_test, tar2_test), 2)
#        print "R^2 for training data", round(clf.score(pred2_train, tar2_train), 2)
#        print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
#        print 'Root Mean Squared Error', round(sqrt(mean_squared_error(tar2_test, y_pred)), 2)
#        print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_pred,tar2_test)[0],"%, <= 3 minutes are ", min_percent(y_pred,tar2_test)[1],"% and <= 5 are ", min_percent(y_pred,tar2_test)[2],"% of the predicted values"
#        Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
#        DistributionPlot(tar2_test,y_pred,"Actual Values (Test)","Predicted Values (Test)",Title)
    average_acc = sum(list_acc) / float(len(list_acc))
    average_mae = sum(list_mae) / float(len(list_mae))
    average_rmse = sum(list_rmse) / float(len(list_rmse))
    print 'the average accuracy of 30 seed is: ', round(average_acc,2), '%'
    print 'the average Mean Absolute Error of 30 seed is: ', round(average_mae,2)
    print 'the average accuracy of 30 seed is: ', round(average_rmse,2)
#MLP2(pred2_train, tar2_train, pred2_test, tar2_test)

'''
#Grid search for Multilayer Perceptron, but it is very time consuming
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
clf = MLPRegressor()
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(pred2_train, tar2_train)
grid_result.fit(pred2_train, tar2_train)

print grid_result.best_params_


best_grid = grid_result.best_estimator_
print best_grid
grid_accuracy = evaluate(best_grid, pred2_test, tar2_test)
print 'grid accuracy', grid_accuracy
'''
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train[['depArr','distance','angle_error','distance_long','angle_sum','QDepDep','QDepArr','QArrDep','NDepDep', 'NDepArr','NArrDep', 'VisibilityInMeters','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','Average_Speed','Medium_aircrafts']], y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

