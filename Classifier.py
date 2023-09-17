#All rights reserved to the Hasnain Riaz and Dataset taken from Kaggle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import matplotlib.pyplot as plt

#reads the data file
data = pd.read_csv("Pubg_Stats.csv")

#Excludes the Sr.no and the player name from the data and put the labels(final output) into y
X = data.drop(data.columns[:2].append(pd.Index(["Rank"])), axis=1)
y = data["Rank"]

#plotting the data with respect to only first three columns
# for feature in X.columns[:3]:
#     print(f"This is {feature}")
#     plt.scatter(X[feature], y, alpha=1)
#     plt.xlabel(feature)
#     plt.ylabel("class")
#     plt.title(f"Scatter Plot of {feature} vs class")
#     plt.show()

#splitting the data into training and testing set randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"First Row{X_train.iloc[1,:]}")
# print(f"This is the X_train {X_train.shape}");
# print(f"This is the X_test {X_test.shape}");
# print(f"This is the y_train {y_train.shape}");
# print(f"This is the y_test {y_test.shape}");

#creating a logistic model Regression Object
model=LogisticRegression(multi_class='multinomial',solver='lbfgs',C=2)
#fitting the model(Tunning the parameters) 
model.fit(X_train,y_train);

#making the prediction on the test set
prediction= model.predict(X_test)


print(f"The test data set is {X_test.iloc[0,:]} and the prediction about it is {prediction[0]} and the actual value is {y_test.iloc[0]}" )
# print(X_test);
# print(y_test);

#measures the accuracy of model in terms test dataset
accuracy=accuracy_score(y_test,prediction);
print(f"Accuracy {accuracy*100}");


#classification report with respects to the test dataset
print("Classification Report:")
print(classification_report(y_test, prediction))

#Confusion matrix w.r.t test dataset
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction))


# Multiclass Problems: For multiclass logistic regression, which involves multiple classes, some of the mentioned optimization algorithms like 'lbfgs' and 'sag' are better suited because they can handle the multiclass case directly ratger than bacth gradient descent.






