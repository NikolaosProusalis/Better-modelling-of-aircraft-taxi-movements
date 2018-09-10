#The process is the same with the first dataset and here two new variable that contain predicted taxi times are added
#Also it was modelled the aircraft type B744 in this script

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

       
data_model = pd.read_csv('man_times_from_ac_model.txt') #read data from the previous forecasting models
#print data_model
#Add the variable with the predicted taxi times to the dataframe that we ue for the modelling
df['timeFromAcModel1'] = data_model['timeFromAcModel1'] #create in the main data frame the new variable for the predicted time
df['timeFromAcModel2'] = data_model['timeFromAcModel2'] #create in the main data frame the new variable for the predicted time


df_sm = pd.DataFrame({'model_aircraft':model})  #create a dataframe with the data of the list model
df_sm['size_aircraft'] = size   #add the data of the list size to the data frame
df_sm = df_sm[df_sm.model_aircraft != 'n/a']    #remove n/a values
df_sm = df_sm[df_sm.model_aircraft != 'Code']   #remove 'Code' values
#df_sm = df_sm[df_sm.size != 'Category']
df_sm = df_sm[df_sm.size_aircraft != 'n/a']     #remove n/a from size_aircraft
df_sm = df_sm.drop_duplicates()     #remove duplicates
df_sm.reset_index(drop = True, inplace = True)  #reset the index of the data frame
#print df_sm


df = df[df.flightNumber != 'null']
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

#Dataframe with only aircraft model B744
df_2 = df
#df_2 =  df_2[df_2.aircraftModel == 'B744']
df_2 = df_2.loc[df_2['aircraftModel'] == 'B744']
#df_2 = df_2.drop('aircraft_size', 1)
df_2 = df_2.drop('aircraftModel',1)
df_2.reset_index(drop = True, inplace = True)
#print df_2

def remove_outlier(df_in, col_name):    #I did not use this function because I was losing a lot of data
    q1 = df_in[col_name].quantile(0.20) #first quantile with values less than the 20%
    q3 = df_in[col_name].quantile(0.80) #third quantile with values after the 80% of the sample
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def remove_outlier2(df_in, col_name):   #call the datadrame and the variable that I want to remove the outliers
    q = df_in[col_name].quantile(0.99)  #only 0.01% percent of the data are removed by the last data value in the sample
    return df_in[df_in[col_name] > q]   #return the dataframe without the outlier

def remove_outlier3(df_in, col_name):   #call the datadrame and the variable that I want to remove the outliers
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


def histogram(df_in, col_name, title, valuex, valuey):  #call the dataframe, the variable, the title of the histogram,value of x axis, value of y axis
    plt.hist(df_in[col_name])   #plot the variable that was set
    plt.title(title)    #plot's title
    plt.xlabel(valuex)  #x axis
    plt.ylabel(valuey)  #y axis
    return plt.show()



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
#outlier_df = outlier_df.append(remove_outlier2(df, 'Average_Speed'))
outlier_df = outlier_df.append(remove_outlier2(df, 'timeFromAcModel1'))
outlier_df = outlier_df.append(remove_outlier2(df, 'timeFromAcModel2'))

outlier_df = outlier_df.drop_duplicates()   #remove the duplicate values of outliers
#print outlier_df
df = df[~df.isin(outlier_df)].dropna()
#print df[~df.isin(outlier_df)].dropna()


df.reset_index(drop = True, inplace = True)
#df[["depArr", "isRain", "isSnow", "isDrizzle", "isFog", "isMist", "isHaze", "isHail"]] = df[["depArr", "isRain", "isSnow", "isDrizzle", "isFog", "isMist", "isHaze", "isHail"]].astype("object")    #change the incorrect data types    
#print df.dtypes
#Data standardisation
# transform TaxiTime from seconds to minutes
df['TaxiTime'] = df["TaxiTime"]*60
#print df.head()
#print df.tail()
#d

#Process in order to import the aircraft size in the dataset
df_list = df['aircraftModel'].tolist()  #put the column aircraftModel in a list

df_mod = df_sm['model_aircraft'].tolist()   #put the column model_aircraft in a list
df_siz = df_sm['size_aircraft'].tolist()    #put the column size_aircraft in a list
list_null_mod=[]
df_new= []  #the new list with the size of the aircraft that will be imported in df data frame
index = []  #list that keeps track of the indexes
indexi = None   #set indexi as None
for i in range(8029):   #range of the number of data in df_list
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

df.reset_index(drop = True, inplace = True)
df = df.drop('aircraftModel', 1)   #delete aircraft Model column

#Histogram of the new variable
#histogram(df, 'timeFromAcModel1', 'Histogram of timeFromAcModel1', '', 'instances')
#histogram(df, 'timeFromAcModel2', 'Histogram of timeFromAcModel2', '', 'instances')
'''
#Pearson correlation and P-value
pearson_coef, p_value = stats.pearsonr(df['timeFromAcModel1'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for timeFromAcModel1 is", pearson_coef, "with a P-value of P =", p_value

pearson_coef, p_value = stats.pearsonr(df['timeFromAcModel2'], df['TaxiTime'])
print"The Pearson Correlation Coefficient for timeFromAcModel2 is", pearson_coef, "with a P-value of P =", p_value
'''
'''
lm1 = LinearRegression()
lm1.fit(df[['timeFromAcModel1']], df[['TaxiTime']])
# Slope 
print lm1.coef_
## Intercept
print lm1.intercept_
#So: taxitime = 181.22821196 - 1.97678688*timeFromAcModel1

lm2 = LinearRegression()
lm2.fit(df[['timeFromAcModel2']], df[['TaxiTime']])
# Slope 
print lm2.coef_
## Intercept
print lm2.intercept_
#So: taxitime = 236.4369202 -  1.44568591*timeFromAcModel2
'''
###########################################################################################################################
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



x_data=df.drop('TaxiTime',axis=1)
y_data=df['TaxiTime']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)
#xy_train = x_train
#xy_train['TaxiTime'] = y_train.values

#print xy_train
#xy_test = x_test
#xy_test['TaxiTime']= y_test.values
#xy_test.to_csv('test_air.csv')
#xy_train.to_csv('train_air2.csv')


#print("number of test samples :", x_test.shape[0])
#print("number of training samples:",x_train.shape[0])


#Lasso Regression
#select predictor variables and target variable as separate data sets  
predvar= df[['depArr','distance','angle_error','distance_long','angle_sum','QDepDep',
'QDepArr','QArrDep','QArrArr','NDepDep','NDepArr','NArrDep','NArrArr','Pressure',
'VisibilityInMeters','TemperatureInCelsius','WindSpeedInMPS','isRain','isSnow','isDrizzle','isFog','isMist',
'AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','aircraft_size','timeFromAcModel1','timeFromAcModel2']]

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
predictors['timeFromAcModel1']=preprocessing.scale(predictors['timeFromAcModel1'].astype('float64'))
predictors['timeFromAcModel2']=preprocessing.scale(predictors['timeFromAcModel2'].astype('float64'))
#predictors['Average_Speed']=preprocessing.scale(predictors['Average_Speed'].astype('float64'))
#predictors['Heavy_aircrafts']=preprocessing.scale(predictors['Heavy_aircrafts'].astype('float64'))
#predictors['Light_aircrafts']=preprocessing.scale(predictors['Light_aircrafts'].astype('float64'))
#predictors['Medium_aircrafts']=preprocessing.scale(predictors['Medium_aircrafts'].astype('float64'))

def min_percent(y_pred, y_test):
    error = abs(y_pred - y_test)
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
normalized_df=(df-df.mean())/df.std()

normalized_df = normalized_df.drop('isMist', 1)
normalized_df = normalized_df.drop('isSnow', 1)
x_data2=normalized_df.drop('TaxiTime',axis=1)
pred2_train, pred2_test, tar2_train, tar2_test = train_test_split(x_data2, y_data, 
                                                              test_size=.3, random_state=123)

#data2 = pd.read_csv('train_air.csv')
#x_train2=data2.drop('TaxiTime',axis=1)
#y_train2=data2['TaxiTime']
#
#data3 = pd.read_csv('train_air.csv')
#x_test2=data3.drop('TaxiTime',axis=1)
#y_test2=data3['TaxiTime']


# specify the lasso regression model
def lasso_reg(pred_train, tar_train, pred_test, tar_test):  #function for LASSO, where we put the training data, the corresponding output value, the test data and the corresponding output value
    model = LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)  #fit LASSO LARS model with training data
    # print variable names and regression coefficients
    print dict(zip(predictors.columns, model.coef_))    #print dictionary with columns as variables and regression coefficient
    d = dict(zip(predictors.columns, model.coef_))
    s = pd.DataFrame(d.items(), columns=['Variable', 'Value'])  #convert the dictionary into data frame for easier readibility
    print s
    #s.to_csv('lassoLars3.csv')

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
    plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean squared error')
    plt.title('Mean squared error on each fold')
         

    # MSE from training and test data

    train_error = mean_squared_error(tar_train, model.predict(pred_train))
    test_error = mean_squared_error(tar_test, model.predict(pred_test))
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
    feature_list = list(df.columns)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'


    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(y_lasso,tar_test)[0],"%, <= 3 minutes are ", min_percent(y_lasso,tar_test)[1],"% and <= 5 are ", min_percent(y_lasso,tar_test)[2],"% of the predicted values"

#lasso_reg(pred_train, tar_train, pred_test, tar_test)

#Multiple Linear Regression
def Mult_Linear(x_train, y_train, x_test, y_test):  #import training and test data
    lr=LinearRegression()   #Multiple Linear Regression model
    lr.fit(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog','timeFromAcModel1','timeFromAcModel2']],y_train)
    Rcross=cross_val_score(lr,x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog','timeFromAcModel1','timeFromAcModel2']], y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", Rcross.mean(),"and the standard deviation is" ,Rcross.std()
    yhat=cross_val_predict(lr,x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog','timeFromAcModel1','timeFromAcModel2']], y_test,cv=10)

    #Figure 2: Plot of predicted value compared to the actual value using the test data.
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)
    # Calculate the absolute errors
    errors = abs(yhat - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error for multiple linear regression:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "R^2 for test data", round(lr.score(x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog','timeFromAcModel1','timeFromAcModel2']],y_test),2)
    print "R^2 for training data", round(lr.score(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog','timeFromAcModel1','timeFromAcModel2']],y_train),2)
    print 'Mean Absolute error', round(mean_absolute_error(y_test, yhat),2)
    print 'Mean Squared error', round(mean_squared_error(y_test, yhat))
    print 'Root Mean Squared error', round(sqrt(mean_squared_error(y_test, yhat)),2)


#Ridge Regression for Multiple Linear
def Mult_Linear2(x_train, y_train, x_test, y_test):
    lr=LinearRegression()
    lr.fit(x_train[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_train)
    #x_train_pr=lr.fit_transform(x_train[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog''timeFromAcModel1','timeFromAcModel2']])
    #x_test_pr=lr.fit_transform(x_test[['angle_sum', 'TemperatureInCelsius', 'NArrDep', 'QArrDep','AvgSpdLast5Arr','AvgSpdLast5Dep','NDepDep','AvgSpdLast10Arr','AvgSpdLast10', 'QArrArr', 'NArrArr', 'Pressure', 'NDepArr', 'VisibilityInMeters', 'angle_error', 'isRain', 'WindSpeedInMPS', 'distance', 'depArr', 'AvgSpdLast5','AvgSpdLast10Dep','QDepArr','isDrizzle','QDepDep','isFog''timeFromAcModel1','timeFromAcModel2']])
    print x_train.shape
    RigeModel=Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)  #create a Ridge regression object, setting the regularization parameter to 0.1
    RigeModel.fit(x_train[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_train)   #fit the model using the method fit
    yhat=RigeModel.predict(x_test[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']])   #obtain a prediction
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
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared error', round(sqrt(mean_squared_error(y_test, yhat)), 2)
    print "R^2 for test data", round(lr.score(x_test[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_test),2)
    print "R^2 for training data", round(lr.score(x_train[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_train),2)
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(yhat,y_test)[0],"%, <= 3 minutes are ", min_percent(yhat,y_test)[1],"% and <= 5 are ", min_percent(yhat,y_test)[2],"% of the predicted values"
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)

#Mult_Linear(x_train, y_train, x_test, y_test)

#Function the gives the best degree for polynomial regression
def Pol_degree(x_train, y_train, x_test, y_test):   #import training and test data
    Rsqu_test=[]
    lr=LinearRegression()
    order=[1,2,3]   #list with the degrees that are checked
    for n in order:
        pr=PolynomialFeatures(degree=n)
    
        x_train_pr=pr.fit_transform(x_train)    #Fit to data, then transform it.
    
        x_test_pr=pr.fit_transform(x_test)    
    
        lr.fit(x_train_pr,y_train)
    
        Rsqu_test.append(lr.score(x_test_pr,y_test))
    print Rsqu_test     #r-squared of the degree 1,2 and 3
    plt.plot(order,Rsqu_test)
    plt.xlabel('order')
    plt.ylabel('R^2')
    plt.title('R^2 Using Test Data')
    plt.text(3, 0.75, 'Maximum R^2 ')  

def PolReg(x_train, y_train, x_test, y_test,degree2):
    pr=PolynomialFeatures(degree=degree2)
    x_train_pr=pr.fit_transform(x_train)
    x_test_pr=pr.fit_transform(x_test)
    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)
    #Cross-validation Score
    Rcross=cross_val_score(poly,x_train_pr, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", Rcross.mean(),"and the standard deviation is" ,Rcross.std()
    yhat=cross_val_predict(poly,x_test_pr, y_test,cv=10)
    #yhat=poly.predict(x_test_pr )
#    print y_test.values
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
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)

#PolReg(x_train, y_train, x_test, y_test,2)
    
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


#Part 3: Ridge Regression
def Opt_Ridge_Pol_Reg(x_train, y_train, x_test, y_test, alpha2, normalise): #import training, test data and degree
    pr=PolynomialFeatures(degree=2) #Polynomial algorithm with the degree that was set in the function
    x_train_pr=pr.fit_transform(x_train)   #transform the training data into polynomial form
    x_test_pr=pr.fit_transform(x_test)  #transform the test data into polynomial form
    print x_train_pr.shape  #print the shape of transformed training data
    RigeModel=Ridge(alpha=alpha2, copy_X=True, fit_intercept=True, max_iter=None, normalize=normalise, random_state=None, solver='auto', tol=0.001)  #create a Ridge regression object, setting the regularization parameter to 0.001
    RigeModel.fit(x_train_pr,y_train)   #fit the model using the method fit
    yhat=RigeModel.predict(x_test_pr)   #obtain a prediction
    print'predicted:', yhat[0:4]
    print'test set :', y_test[0:4].values

    # Calculate the absolute errors
    errors = abs(yhat - y_test)
    # Print out the mean absolute error (mae)

    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'
    print 'Root Mean Squared error', sqrt(mean_squared_error(y_test, yhat)) 

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print'Accuracy:', round(accuracy, 2), '%.'
    print "The predicted taxi times that have <= 1 minute deviation from the actual taxi time are ", min_percent(yhat,y_test)[0],"%, <= 3 minutes are ", min_percent(yhat,y_test)[1],"% and <= 5 are ", min_percent(yhat,y_test)[2],"% of the predicted values"
    Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
    DistributionPlot(y_test,yhat,"Actual Values (Test)","Predicted Values (Test)",Title)

#Opt_Ridge_Pol_Reg(x_train, y_train, x_test, y_test)

#-----------------------------------------------------------------------------------------------------------------------------

#Random forest
#print x_train.shape
#print x_test.shape
#print y_train.shape
#print y_test.shape
from sklearn.ensemble import RandomForestRegressor

def Random_Forest(x_train, y_train, x_test, y_test):
    #Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(random_state = 28, max_depth=20, max_features='auto',min_samples_leaf=7, n_estimators=200)
    regressor.fit(x_train,y_train)  #fit the model
    #Cross-validation Score
    Rcross=cross_val_score(regressor,x_train, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
    print"The mean of the folds is", round(Rcross.mean(),2),"and the standard deviation is" ,round(Rcross.std(),2)
    y_pred=cross_val_predict(regressor,x_test, y_test,cv=10)
#    print yhat[0:5]
    #y_pred = regressor.predict(x_test) #Predicting a new result
    #print regressor.feature_importances_
#    print y_pred

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    print'Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.'

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    feature_list = list(df.columns)
    feature_list.remove('TaxiTime')
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

    ## Print out the feature and importances 
#    for pair in feature_importances:
#        print('Variable: {:20} Importance: {}'.format(*pair)) 
#Random_Forest(x_train[['distance','QDepDep','timeFromAcModel2','WindSpeedInMPS','depArr','distance_long','angle_sum','Pressure','TemperatureInCelsius','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Arr','AvgSpdLast10','timeFromAcModel1']], y_train, x_test[['distance','QDepDep','timeFromAcModel2','WindSpeedInMPS','depArr','distance_long','angle_sum','Pressure','TemperatureInCelsius','AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Arr','AvgSpdLast10','timeFromAcModel1']], y_test)

#Random Forest , which gives the average accuracy mean abdolute error and root mean squared error, from the results of the model with 30 different random_states
def Random_Forest2(x_train, y_train, x_test, y_test):
    #Fitting Random Forest Regression to the dataset
    list_acc = []
    list_mae =[]
    list_rmse = []
    for rand in range(1,31):
        regressor = RandomForestRegressor(random_state = rand, max_depth=20, max_features='auto',min_samples_leaf=7, n_estimators=200)
        #Best hyperparameters: max_depth=None, max_features='auto', min_samples_leaf=8,n_estimators=150
        #print 'Parameters:', get_params(deep=True)
        #    'bootstrap': [True],
        #    'max_depth': [80, 90, 100, 110], not this i'll try 20 and 40
        #    'max_features': [2, 3],
        #    'min_samples_leaf': [3, 4, 5],
        #    'min_samples_split': [8, 10, 12],
        #    'n_estimators': [100, 200, 300, 1000]
        #best parameters n_estimators= 200, random_state = 0,max_features= 3,min_samples_leaf= 3,min_samples_split=12
        regressor.fit(x_train,y_train)
        #Cross-validation Score
        Rcross=cross_val_score(regressor,x_train, y_train,cv=10)    #We input the object, the feature in this case ' depArr', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 10
#       print"The mean of the folds is", round(Rcross.mean(),2),"and the standard deviation is" ,round(Rcross.std(),2)
        y_pred=cross_val_predict(regressor,x_test, y_test,cv=10)
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
        list_acc.append(accuracy)
        list_mae.append(np.mean(errors))
        list_rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
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
#        for pair in feature_importances:
#            print('Variable: {:20} Importance: {}'.format(*pair)) 
    average_acc = sum(list_acc) / float(len(list_acc))
    average_mae = sum(list_mae) / float(len(list_mae))
    average_rmse = sum(list_rmse) / float(len(list_rmse))
    print 'the average accuracy of 30 seed is: ', round(average_acc,2), '%'
    print 'the average Mean Absolute Error of 30 seed is: ', round(average_mae,2)
    print 'the average accuracy of 30 seed is: ', round(average_rmse,2)
#Random_Forest(x_train[['QDepDep','depArr','distance','angle_sum','AvgSpdLast5Dep','timeFromAcModel1','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','timeFromAcModel2','aircraft_size','angle_error']], y_train, x_test[['QDepDep','depArr','distance','angle_sum','AvgSpdLast5Dep','timeFromAcModel1','distance_long','QDepArr','NDepDep','Pressure','TemperatureInCelsius','WindSpeedInMPS','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','timeFromAcModel2','aircraft_size','angle_error']], y_test)

#Variable Selection with Variance Threshold
from sklearn.feature_selection import VarianceThreshold
def Var_Thershold(pred2_train, pred2_test):
    # Arbitrarily set threshold to 0.1
    sel = VarianceThreshold(threshold = 0.1)
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Multilayer Perceptron
from sklearn.neural_network import MLPRegressor
def Multilayer_Perceptron(pred2_train, tar2_train, pred2_test, tar2_test):  #import training and test data
    clf = MLPRegressor(random_state=11,activation = 'identity', hidden_layer_sizes = 100, learning_rate_init = 0.001, max_iter = 200)
    clf.fit(pred2_train, tar2_train) 
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
#Multilayer_Perceptron(pred2_train, tar2_train, pred2_test, tar2_test)

#Multilayer Perceptron , which gives the average accuracy mean abdolute error and root mean squared error, from the results of the model with 30 different random_states
def MLP2(pred2_train, tar2_train, pred2_test, tar2_test):   #import training and test data
    list_acc = []
    list_mae =[]
    list_rmse = []
    for rand in range(1,31):
        clf = MLPRegressor(random_state=rand,activation = 'identity', hidden_layer_sizes = 100, learning_rate_init = 0.001, max_iter = 200)
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
Multilayer_Perceptron(pred2_train, tar2_train, pred2_test, tar2_test)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Modelling process for aircrafts B744
x_data=df_2.drop('TaxiTime',axis=1)
y_data=df_2['TaxiTime']
x_train2,x_test2,y_train2,y_test2= train_test_split(x_data, y_data, test_size=0.3, random_state=1)

predvar= df_2[['depArr','distance','angle_error','distance_long','angle_sum','QDepDep',
'QDepArr','QArrDep','QArrArr','NDepDep','NDepArr','NArrDep','NArrArr','Pressure',
'VisibilityInMeters','TemperatureInCelsius','WindSpeedInMPS','isRain','isSnow','isDrizzle','isFog','isMist',
'AvgSpdLast5Dep','AvgSpdLast5Arr','AvgSpdLast5','AvgSpdLast10Dep','AvgSpdLast10Arr','AvgSpdLast10','timeFromAcModel1','timeFromAcModel2']]
predictors=predvar.copy()
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
predictors['timeFromAcModel1']=preprocessing.scale(predictors['timeFromAcModel1'].astype('float64'))
predictors['timeFromAcModel2']=preprocessing.scale(predictors['timeFromAcModel2'].astype('float64'))

# split data into train and test sets
pred_train2,pred_test2,tar_train2,tar_test2= train_test_split(predictors, y_data, 
                                                              test_size=.3, random_state=123)

#Normalize the data
normalized_df=(df_2-df_2.mean())/df_2.std()

normalized_df = normalized_df.drop('isMist', 1)
normalized_df = normalized_df.drop('isSnow', 1)
normalized_df = normalized_df.drop('isFog', 1)
normalized_df = normalized_df.drop('isDrizzle', 1)
x_data2=normalized_df.drop('TaxiTime',axis=1)
pred2_train2,pred2_test2,tar2_train2,tar2_test2= train_test_split(x_data2, y_data, 
                                                              test_size=.3, random_state=123)
#print pred2_train2
#lasso_reg(pred_train2, tar_train2, pred_test2, tar_test2)
#Mult_Linear(x_train2,y_train2,x_test2,y_test2)
#Mult_Linear2(x_train2,y_train2,x_test2,y_test2)
#PolReg(x_train2, y_train2, x_test2, y_test2,2)
#Pol_degree(x_train2,y_train2,x_test2,y_test2)
'''
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000], 'normalize': [True,False]}]
RR=Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=10)  #Create a ridge grid search objec
Grid1.fit(x_train2[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_train2)  #Fit the model
BestRR=Grid1.best_estimator_    #The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR 
print "The best parameter values", BestRR
scores = Grid1.cv_results_
for param,mean_val,mean_test in zip(scores['params'],scores['mean_test_score'],scores['mean_train_score']):
    print param, "R^2 on train data: ", mean_val, "R^2 on test data: ", mean_test
print"By testing our model the R^2 on the test data: ", BestRR.score(x_test2[['timeFromAcModel1', 'Pressure', 'angle_error', 'WindSpeedInMPS','depArr','QDepArr','QDepDep']],y_test2)
'''
#Opt_Ridge_Pol_Reg(x_train2,y_train2,x_test2,y_test2, 1, True)
#Random_Forest(x_train2,y_train2,x_test2,y_test2)
#Multilayer_Perceptron(pred2_train2, tar2_train2, pred2_test2, tar2_test2)
#Var_Thershold(pred2_train2, pred2_test2)
