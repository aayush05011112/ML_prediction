import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





data=pd.read_csv("housing.csv")
# print(data.head())
# # print(data.info)
# # print(data.describe)

x=pd.DataFrame(data.drop("MEDV",axis=1))
# print(x)
y=pd.DataFrame(data["MEDV"])
# print(y)


for column in x.columns:
    plt.scatter(data[column], data["MEDV"])
    plt.xlabel(column)
    plt.ylabel("Median Value (MEDV)")
    plt.title(f"{column} vs MEDV")
    # plt.show()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(x_train,y_train)

# print(model.coef_)
# print(model.intercept_)
def prediction(input_feature):
    pred=model.predict(input_feature)
    return pred

predict=prediction(x_test)


from sklearn import metrics

print(f"RMSE:{np.sqrt(metrics.mean_squared_error(y_test,predict))}")

def predict_from_user_input():
    print("Please enter values for the following features:")
    

    features = {}
    for column in x.columns:
        while True:
            try:
                value = float(input(f"{column}: "))
                features[column] = [value]  # Store as list to create DataFrame later
                break
            except ValueError:
                print("Please enter a valid number.")
    

    input_df = pd.DataFrame(features)
    
   
    pred = prediction(input_df)
    print(f"\nPredicted Median Value (MEDV): ${pred[0][0]:.2f}")


predictions = prediction(x_test)
from sklearn import metrics
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}")


while True:
    user_choice = input("\nWould you like to make a prediction? (yes/no): ").lower()
    if user_choice == 'yes':
        predict_from_user_input()
    elif user_choice == 'no':
        print("Goodbye!")
        break
    else:
        print("Please enter 'yes' or 'no'.")
    