#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, show
import unittest
from pandas.util.testing import assert_frame_equal


def load_file(location, x_variable="x"):
        try:
            data = pd.read_csv(location)

            for column in data:
                if data[column].dtype != float:
                    data.drop(
                        [column], axis=1, inplace=True
                    )  # drop all columns that contain non-float values

            if data.isnull().any().any():
                print("Data file contains empty cells")

            if x_variable not in data.columns:
                print("X variable not found in data file")

            if len(data.columns) < 2:
                print("No Y values found in data file")

        except Exception as e:
            print(e)
            quit()
        else:
            return data

class TestAssignment(unittest.TestCase):
    """ class for running unittests """

    def setUp(self):
        
        data = pd.read_csv('test.csv')
        assert len(set(data.columns) - set(['x', 'y'])) == 0
 
        
        
def showgraphsingle(title,a,b):
    """
    This is the function which shows the graph for the given two series
    The input A will be used as X axis, the input B will be used plotted as Y axis
    The input title is used as the title in for the shown chart
    """
    output_notebook()
    p = figure(title=title,plot_width = 900, plot_height = 200)
    p.circle(a, b,size = 5, color = "red", alpha = 0.5)
    show(p) 
    
def showgraphcomparisson(title,a,b,c,legend1,legend2):
    """
    This is the function which shows the graph for the given three series
    The input A will be used as X axis, the input B and C will be used plotted as Y axis
    The input title is used as the title in for the shown chart
    Legend1 and legend2 will be used as the legend for respective serie
    """
    output_notebook()
    p = figure(title=title, width=900, height=400)
    p.circle(a,b,size = 5, color="red",legend_label=legend1)
    p.line(a,c, color="blue",legend_label=legend2)
    show(p)
    


class train_ideal:
    
    """
    This is a class created to store all the ideal functions for the respective train functions along with its difference
    """
    def __init__(self, x, y1, y2, y3, y4, y1_delta, y2_delta, y3_delta, y4_delta):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y1_delta = y1_delta
        self.y2_delta = y2_delta
        self.y3_delta = y3_delta
        self.y4_delta = y4_delta

class test_included(train_ideal):
    """
    This is an inherited child class which only has the combination of train and test functions 
    """
    def __init__(self, x, y1, y2, y3, y4, y1_delta, y2_delta, y3_delta, y4_delta, y):
        self.y = y
        train_ideal.__init__(self, x, y1, y2, y3, y4, y1_delta, y2_delta, y3_delta, y4_delta) 
            
class Error(Exception):
    """Base class for other exceptions"""
    pass


class allfine(Error):
    """Everything is perfect"""
    pass


class notfine(Error):
    """This is not correct"""
    pass
   


# importing all the raw data from csv file
test = load_file("test.csv")
train = load_file("train.csv")
ideal = load_file("ideal.csv")

#visialising all the data train data 
showgraphsingle("Train data Y1",train["x"], train["y1"])
showgraphsingle("Train data Y2",train["x"], train["y2"])
showgraphsingle("Train data Y3",train["x"], train["y3"])
showgraphsingle("Train data Y4",train["x"], train["y4"])


#Looking for suitable ideal function for all the train function using Least-Square method

a1 = pd.DataFrame( columns =[ "y1_ideal", "y2_ideal", "y3_ideal", "y4_ideal"])
for idx,column in enumerate(ideal):
        train['construct1'] = (train['y1']-ideal[column])**2
        train['construct2'] = (train['y2']-ideal[column])**2
        train['construct3'] = (train['y3']-ideal[column])**2
        train['construct4'] = (train['y4']-ideal[column])**2
        s = pd.DataFrame([[ train['construct1'].sum(), train['construct2'].sum(),train['construct3'].sum(),train['construct4'].sum()]], columns = ["y1_ideal", "y2_ideal", "y3_ideal", "y4_ideal"])
        a1 = a1.append(s, ignore_index = True) 
        train = train.drop(['construct1','construct2','construct3','construct4'] , axis =1)


minvalueIndexLabel = a1.idxmin()
for index,value in minvalueIndexLabel.items():
    train[index] = ideal["y" + str(value)]

#visualising the graphs with Ideal and train data set
showgraphcomparisson("Train data Y1 vs Ideal Y1", train["x"],train["y1"],train["y1_ideal"],"Y1 train","Y1 ideal")
showgraphcomparisson("Train data Y2 vs Ideal Y2", train["x"],train["y2"],train["y2_ideal"],"Y2 train","Y2 ideal")
showgraphcomparisson("Train data Y3 vs Ideal Y3", train["x"],train["y3"],train["y3_ideal"],"Y3 train","Y3 ideal")
showgraphcomparisson("Train data Y4 vs Ideal Y4", train["x"],train["y4"],train["y4_ideal"],"Y4 train","Y4 ideal")

#visualising the Test data set
showgraphsingle("Test Data",test["x"], test["y"])
    
#left joining the test data and train data with ideal function
train = train.merge(test, on='x', how='left')

#calculating the delta between the Train and ideal function
train["y1_delta"] = abs(train["y1_ideal"]-train["y1"])
train["y2_delta"] = abs(train["y2_ideal"]-train["y2"])
train["y3_delta"] = abs(train["y3_ideal"]-train["y3"])
train["y4_delta"] = abs(train["y4_ideal"]-train["y4"])


#adding the processed data as objects
all_ideal = train_ideal(train["x"], train["y1_ideal"], train["y2_ideal"], train["y3_ideal"], train["y4_ideal"], train["y1_delta"], train["y2_delta"], train["y3_delta"], train["y4_delta"])
train.dropna(subset = ["y"], inplace=True)
just_test = test_included(train["x"], train["y1_ideal"], train["y2_ideal"], train["y3_ideal"], train["y4_ideal"], train["y1_delta"], train["y2_delta"], train["y3_delta"], train["y4_delta"], train["y"])



ideal_fuction = []
delta = []
actual_delta = []
ideal_val = []


"""
From the objects which were created earlier finding suitable
ideal function which has least difference between the test and ideal
"""

i = 0
for index in just_test.x:
    temp1 = abs(just_test.y.iloc[i] - just_test.y1.iloc[i])
    temp2 = abs(just_test.y.iloc[i] - just_test.y2.iloc[i])
    temp3 = abs(just_test.y.iloc[i] - just_test.y3.iloc[i])
    temp4 = abs(just_test.y.iloc[i] - just_test.y4.iloc[i])
    
    t = [['y1', temp1, just_test.y1.iloc[i],just_test.y1_delta.iloc[i]], ['y2', temp2, just_test.y2.iloc[i],just_test.y2_delta.iloc[i]], ['y3', temp3, just_test.y3.iloc[i],just_test.y3_delta.iloc[i]], ['y4', temp4, just_test.y4.iloc[i],just_test.y4_delta.iloc[i]]]
    differ = pd.DataFrame(t, columns = ['ideal_fuction', 'difference', 'ideal_value', 'actual_delta'])
    
    differ = differ.set_index('ideal_fuction')
    y = differ['difference'].idxmin()
    w = differ.loc[differ['difference'].idxmin(),"actual_delta"]
    q = differ.loc[differ['difference'].idxmin(),"ideal_value"]
    ideal_fuction.append(y)
    delta.append(w)
    ideal_val.append(q)
    i=i+1

x_list = just_test.x.tolist()
y_list = just_test.y.tolist()
    
#creating final dataset with the test function and its respective train function
final = pd.DataFrame(list(zip(x_list, y_list, delta, ideal_fuction, ideal_val)),columns =['X (test func)','Y (test func)','Delta Y (train func)', 'No. of ideal func', 'Ideal Val'])

#processing final data to filter add additional columns
final['max_delta'] = final["Delta Y (train func)"] * math.sqrt(2)
final['Delta Y (test func)'] = abs(final["Y (test func)"] - final['Ideal Val'])
final['Condition'] = final['Delta Y (test func)']<final['max_delta']

#visualising the Max delta vs the actual delta the function got
output_notebook()
p = figure(title="Max Delta vs Delta Y (test func)", width=900, height=400)
p.star(final["X (test func)"],final["max_delta"],size = 5, color="blue",legend_label="Max delta")
p.circle(final["X (test func)"],final["Delta Y (test func)"],size = 5, color="red",legend_label="Delta Y (test func)")
show(p)


#removing the rows which doesn't meet the criteria of not having less delta than max delta
final.drop(final[ (final['Condition'] == False)].index, inplace = True)


#User defined exception to check and confirm if the data in the final table meets the criteria
try:
    if (final["Condition"] == True).all(axis=0):
        raise allfine
    elif (final["Condition"] == False).all(axis=0):
        raise notfine
except allfine:
    print("Everything is perfect")
    print()
except notfine:
    print("This is not correct")
    print()


#visualising the Max delta vs Delta Y    
output_notebook()
p = figure(title="Max Delta vs Delta Y (test func)", width=900, height=400)
p.line(final["X (test func)"],final["max_delta"], color="blue",legend_label="Max delta")
p.line(final["X (test func)"],final["Delta Y (test func)"], color="red",legend_label="Delta Y (test func)")
show(p)

final = final.reset_index()
#dropping the unwanted columns from the Dataframe
final = final.drop(['Delta Y (train func)',  "max_delta", "Delta Y (test func)", "Condition", "index"] , axis =1)


#comparing the final result in graph
showgraphcomparisson("Train data Y4 vs Ideal Y4", final["X (test func)"],final["Y (test func)"],final["Ideal Val"],"Y (test func)","Ideal Val")

#showing the final expected data
print(final)

if __name__ == '__main__':
    unittest.main()


# In[ ]:




