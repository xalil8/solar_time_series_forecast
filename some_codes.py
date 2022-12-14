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