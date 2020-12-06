import pandas as pd
import numpy as np
import streamlit as st
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

cleaned_df = pd.read_csv('cleaned_dataframe.csv', index_col = 'index')

st.sidebar.title('Variable Configuration')


#generate a list of option by getting all headers.
prediciton_options_all_possible_variables = []
for col in cleaned_df.columns:
    prediciton_options_all_possible_variables.append(col)

prediciton_options_selection = st.sidebar.multiselect(
    'What are the variables you want to analyze?',
    prediciton_options_all_possible_variables, default = 'FreeCashFlow')

#use a dictionary to store expected growth rate associated with each variable.
prediciton_option_predicted_growth_rate = {}

for item in prediciton_options_selection:
    #we divided the result by 100 so that the value store is in %
    prediciton_option_predicted_growth_rate[item] = st.sidebar.number_input(label = str(item) + " in %", value = 5.00, step = 0.01) / 100

# We randomly split our dataset into training set and validation set.
# Then we will have some validation data for model predicton performance testing.
np.random.seed(100)
prediction_list_of_all_columns = np.arange(0, len(cleaned_df))
prediction_selected_validation = np.sort(np.random.choice(a=len(cleaned_df), size=len(cleaned_df)//4))
prediction_selected_training = np.setdiff1d(prediction_list_of_all_columns, prediction_selected_validation)



#def functions that return a regression model fitting into our selected variables and a r2_score when using validation date
def prediction_regression_model(list_of_selected_variables):
    train = cleaned_df.iloc[prediction_selected_training][:]
    train_x = train.loc[:, list(list_of_selected_variables)]
    train_y = train['Close']
    model = linear_model.LinearRegression()
    model.fit(train_x,train_y)
    return model

def prediction_regression_r2(tested_df, tested_columns, tested_model, tested_variables = list(prediciton_option_predicted_growth_rate)):
    test = tested_df.iloc[tested_columns]
    test_x = test[tested_variables]
    test_y = test['Close']
    pred = tested_model.predict(test_x)
    return r2_score(test_y, pred)
#we have defined two functions that return model and accuracy of the model.

#As the regression model can not take a dictionary as an input, we transfer the dictionary with the name of variables and
#the expected growth rate to a format acceptable by the regression model.
def prediction_option_to_df(input_dic):
    local_df = pd.DataFrame(index = ['expected_value'], columns = list(input_dic))
    for variable in list(input_dic):
        local_df.iloc[0][str(variable)] = cleaned_df.iloc[0][variable] * (1 + input_dic[variable])
    return local_df
#we've done the function of transfering inputs into an acceptable format.

#Now, we want to define a function to return the predicted stock price given the inputs.
#this function is not necessary as the actual method only takes one step, but defining such a function
#imcreases the readbility of the code.
def prediction_predicted_value(model, df_input):
    return float(model.predict(df_input))

model = prediction_regression_model(prediciton_option_predicted_growth_rate)
prediction_input = prediction_option_to_df(prediciton_option_predicted_growth_rate)
expected_value = prediction_predicted_value(model, prediction_input)

#since we want to draw several scatter plot to show the performance of the model, we define a function to help
#us do so
def scatter_plot(tested_df, tested_indexes, model, title, tested_variables = list(prediciton_option_predicted_growth_rate)):
    test = tested_df.iloc[tested_indexes]
    test_x = test[tested_variables]
    test_y = test['Close']
    pred = model.predict(test_x)
    figure_test = plt.figure(figsize=(6, 3), dpi=100)
    figure_test.suptitle(title, fontsize=16)
    plt.scatter(range(len(test_y)),test_y[::-1], color='black', label = 'Actual')
    plt.scatter(range(len(pred)),pred[::-1], color='blue', label = 'Predict')
    plt.legend()
    return figure_test




st.sidebar.title('Expected Price: ' + str(expected_value))

st.title('Performance of the Selected Model')

st.pyplot(fig = scatter_plot(tested_df = cleaned_df,
                             tested_indexes = prediction_selected_training,
                             model = model,
                             title = 'Performance on train data'))

st.write('\nR square on train data: ', prediction_regression_r2(tested_df = cleaned_df,
                                                                tested_columns = prediction_selected_training,
                                                                tested_model = model,))

st.pyplot(fig = scatter_plot(tested_df = cleaned_df,
                             tested_indexes = prediction_selected_validation,
                             model = model,
                             title = 'Performance on validation data'))

st.write('\nR square on validation data: ', prediction_regression_r2(tested_df = cleaned_df,
                                                                     tested_columns = prediction_selected_validation,
                                                                     tested_model = model,))
