#1 import required libraries to perform initial tasks
import pandas as pd
import matplotlib.pyplot as plt
#Pandas library is used to manipulate and analyse data that is displayed in tables, CSVs, etc.
#Matplotlib is used manipulate and redistribute the data as an interactive visualization

#2 import your DataFrame and assign to variable [1], then display DataFrame
health = pd.read_csv("diabetes.csv") 
health
#Note: The DataFrame 'Health' [1] that I have used in this example consists of the outcome, in binary form, that I want to test using scikit-learn, 
#therefore standardisation and re-labelling the DataFrame is not required
#0 = female patients over 21 diagnosed with diabetes
#1 = female patients over 21 with no diabetes diagnosis

import seaborn as sns
#Seaborn is used to create statistical graphs
sns.countplot(health["Outcome"]) 
plt.ylabel("Female Patients") 
plt.xlabel("No Diagnosis vs. Diagnosed") 
plt.title("Diabetes Database") 
plt.show()

print(health["Outcome"].value_counts())

#3 importing the train_test_split function from scikit-learn to begin our training and testing of the data
# as the data already contains the binary outcome, I am separating and reformatting the data from the 'Outcome' column and assigning value to X and y
# next, the data is divided to reflect what is to be trained and what is to be tested, while setting the test size
#In my example I am only testing 5% of the data, due to the small input (768 rows). The general rule is 20% test, 80% train, or 30%/70% respectively.

from sklearn.model_selection import train_test_split X = health.drop("Outcome", axis=1)
y = health["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, rando m_state=0)
X_train

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

#4 I will explore three example machine learning algorithms to test and train the data, then make predictions, and test the accuracy of each algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_s core
dtc = DecisionTreeClassifier() 
dtc.fit(X_train, y_train) 
pred_dtc = dtc.predict(X_test)
print(classification_report(y_test, pred_dtc))
 
from sklearn.metrics import plot_confusion_matrix 
print(confusion_matrix(y_test, pred_dtc))
plot_confusion_matrix(dtc, X_test, y_test) 
plt.show()
print("Overall Accuracy:", accuracy_score(pred_dtc, y_test))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200) 
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc)) 
print(confusion_matrix(y_test, pred_rfc))
print("") #blank line input to increase readability of result 
print("Overall Accuracy:", accuracy_score(pred_rfc, y_test))

from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000) 
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
print(classification_report(y_test, pred_mlpc)) 
print(confusion_matrix(y_test, pred_mlpc))
print("") #blank line input to increase readability of result 
print("Overall Accuracy:", accuracy_score(pred_mlpc, y_test))

from sklearn.neighbors import KNeighborsClassifier
nnc = KNeighborsClassifier() 
nnc.fit(X_train, y_train) 
pred_nnc = nnc.predict(X_test)
print(classification_report(y_test, pred_nnc)) 
print(confusion_matrix(y_test, pred_nnc))
print("") #blank line input to increase readability of result 
print("Overall Accuracy:", accuracy_score(pred_nnc, y_test))

#5Once you have performed various training and testing algorithms, determine which algorithm had best overall accuracy score, 
#then perform a single test with a random array to determine the outcome.
#In this instance, the Random Forest Classifier has performed
 
X_new_tester = [[4, 127, 67, 0, 45, 29.9, 0.327, 38]] 
X_new_tester = sc.transform(X_new_tester) 
pred_X_new_tester = rfc.predict(X_new_tester) 
pred_X_new_tester
#Determined that this array, based on the data inside the array, would be a patient with diabetes
