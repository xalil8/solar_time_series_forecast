#l2 and l1 regularizations also can be use with SGD regressor model by using penalty parameter set to "l2" or "l1"



sgd_reg = SGDRegressor(penalty="l2")

sgd_reg.fit(x_train,y_train)

sgd_train = sgd_reg.predict(x_train)
sgd_test = sgd_reg.predict(x_test)

metrics(y_train,sgd_train,y_test,sgd_test)

#############################################
#elastic_net is both mixture of lasso and ridge regresion, it can be adjust by using l1_ratio

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter= 5000)

elastic_net.fit(x_train,y_train)

elastic_train = elastic_net.predict(x_train)
elastic_test = elastic_net.predict(x_test)

metrics(y_train,elastic_train,y_test,elastic_test)


###########################################################

#kneigbor model, give best test result for data 
KN = KNeighborsRegressor()
KN.fit(x_train, y_train)

KN_train_pred = KN.predict(x_train)
KN_test_pred = KN.predict(x_test)

metrics(y_train,KN_train_pred,y_test,KN_test_pred)




##############################################################
ridge_train =[]
ridge_test = []
lasso_train = []
lasso_test = []
for i in errors:
    ridge_train.append(i["ridge"][0])
    ridge_test.append(i["ridge"][1])
    lasso_train.append(i["lasso"][0])
    lasso_test.append(i["lasso"][1])

aylar = ["January","February","March","April","May","June","July","August","September","October","November","December"]

seri1  = pd.Series(ridge_train) 
seri2  = pd.Series(ridge_test) 
seri3  = pd.Series(lasso_train) 
seri4  = pd.Series(lasso_test) 
seri5 = pd.Series(aylar)

frame = {"months":seri5,"ridge_train":ridge_train, "ridge_test":ridge_test,"lasso_train":lasso_train,"lasso_test":lasso_test}
result = pd.DataFrame(frame)


###################################################################

data = result

fig = go.Figure()
fig.add_trace(go.Bar(x=train_errors["months"],y=train_errors["ridge"],name='Ridge Model Train Results', marker_color='indianred'))

fig.add_trace(go.Bar(x=train_errors["months"], y=test_errors["ridge"], name='Ridge Model Test Results',marker_color='lightsalmon'))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45,title="(-1,1) normalization, 3 encoding")
fig.show()



######################################################################


def calculate_ridge(x_train,x_test,y_train,y_test):
    model = Ridge(alpha=1) 
    model.fit(x_train,y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)



def calculate_lasso(x_train,x_test,y_train,y_test):
    model = Lasso(alpha=1) 
    model.fit(x_train,y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

    
def calculate_linear_reg(x_train,x_test,y_train,y_test):

    model = LinearRegression()
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

def calculate_xgb(x_train,x_test,y_train,y_test):
    model = XGBRegressor()
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

def calculate_kneighbors(x_train,x_test,y_train,y_test):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

def calculate_decisiontree(x_train,x_test,y_train,y_test):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

def calculate_randomforest(x_train,x_test,y_train,y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return mean_squared_error(y_train, train_prediction), mean_squared_error(y_test, test_prediction)

