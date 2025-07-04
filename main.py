from Readfile import Read_file
import pandas as pd
import joblib
import numpy as np
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
def main():
    path = r"data/car.csv"
    data = Read_file(path).read()
    # report = ProfileReport(data,title ="Profiling Report")
    # report.to_file("report.html")
    target = "mpg"
    data['horsepower'] = data['horsepower'].replace('?', np.nan).astype(float)
    y = data[target]
    x = data.drop([target,"car name"],axis=1)
    chuanhoa = [ 'displacement', 'horsepower', 'weight','acceleration', 'model year']
    num_features = Pipeline(steps=[
        ("impute",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
    ("num", num_features, chuanhoa)
])
    X_processed = preprocessor.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    # cft = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    # models,predictions = cft.fit(x_train, x_test, y_train, y_test)
    # print(pd.DataFrame(models).sort_values(by='R-Squared', ascending=False))
    # RandomForestRegressor                        0.91       0.91  2.14        0.24
    # HistGradientBoostingRegressor                0.90       0.91  2.23        0.22
    # GradientBoostingRegressor                    0.90       0.91  2.24        0.13
    params = {
        "n_estimators":[100,200,300],
        "criterion":["squared_error", "absolute_error", "friedman_mse", "poisson"],
        'min_samples_split': [2, 5]
    }
    # print(x_train,y_train)
    # gsv = GridSearchCV(
    #     estimator=RandomForestRegressor(random_state=42),
    #     param_grid=params,
    #     scoring='neg_mean_absolute_error',
    #     cv=4,
    #     verbose=2)
    # gsv.fit(x_train,y_train)
    # print(gsv.best_estimator_)
    # print(gsv.best_params_)
    # print(gsv.best_score_)
    model = RandomForestRegressor(criterion="absolute_error",min_samples_split=2,n_estimators=300,random_state=42)
    model.fit(x_train,y_train)
    # y_pred=model.predict(x_test)
    # print(y_pred)
    # for i,j in zip(y_pred,y_test):
    #     print(f"ket qua:{i} - thuc te:{j}")
    # y_test_class = y_test.round().astype(int)
    # y_pred_class = y_pred.round().astype(int)

    # print(classification_report(y_test_class, y_pred_class))
    joblib.dump(model, 'model_rf.pkl')

# Lưu cả preprocessor (pipeline chuẩn hóa)
    joblib.dump(preprocessor, 'preprocessor.pkl')

main()
