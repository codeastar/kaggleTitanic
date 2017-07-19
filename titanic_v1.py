import pandas as pd
import numpy as np

df = pd.read_csv("Development/Playground/Titanic/train.csv")

#get the test sample size
print(df.shape)

#check null value in the sample
print(df.apply(lambda x: sum(x.isnull()),axis=0))

#survival counts, 1=survived, 0=died
print(df['Survived'].value_counts(ascending=True))

#Sex and survived relationship
sex_stats = pd.pivot_table(df, values='Survived', index=['Sex'], aggfunc=lambda x: x.mean())
print(sex_stats)

#Class and survived relationship
class_stats = pd.pivot_table(df, values='Survived', index=['Pclass'], aggfunc=lambda x: x.mean())
print(class_stats)

#Class, sex and survived relationship
class_sex_stats = pd.pivot_table(df, values='Survived', index=['Pclass', 'Sex'], aggfunc=lambda x: x.mean())
print(class_sex_stats)

#Class, sex and survived in graph
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df)

plt.show()

#familySize in graph
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

sns.barplot(x="FamilySize", y="Survived", data=df)

plt.show()

#put FamilySize into Family group
bins = (-1, 1, 2, 3, 12)
group_names = [1,2,3,4]
categories = pd.cut(df['FamilySize'], bins, labels=group_names)
df['FamilyGroup'] = categories

#count for familyGroup and size
print(df['FamilyGroup'].value_counts(ascending=True))
print(df['FamilySize'].value_counts(ascending=True))

#add isAlone column 
bins = (-1, 1, 12)
group_names = [1,0]
categories = pd.cut(df['FamilySize'], bins, labels=group_names)
df['isAlone'] = categories

#isAlone in graph
sns.barplot(x="isAlone", y="Survived", data=df);

# convert Sex into categorical value 1 for male and 0 for female
df["Sex"] = df["Sex"].map({"male": 1, "female":0})

#check age correlation with others
age_heat = sns.heatmap(df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)

plt.show()

#fill up NaN age according class / sibsp / parch
index_NaN_age = list(df["Age"][df["Age"].isnull()].index)

for i in index_NaN_age :
    age_mean = df["Age"].mean()
    age_std = df["Age"].std()
    age_pred_w_spc = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].mean()
    age_pred_wo_spc = np.random.randint(age_mean - age_std, age_mean + age_std)
    
    if not np.isnan(age_pred_w_spc) :
        df['Age'].iloc[i] = age_pred_w_spc
    else :
        df['Age'].iloc[i] = age_pred_wo_spc    

#check age distribution
age_dist = sns.distplot(df['Age'], label="Skewness : %.2f"%(df["Age"].skew()))
age_dist.legend(loc="best")

plt.show()

#separate age into 4 groups 
bins = (-1, 16, 30, 50, 100)
group_names = [1,2,3,4]
categories = pd.cut(df['Age'], bins, labels=group_names)
df['AgeGroup'] = categories

print(df['AgeGroup'].value_counts(ascending=True))

#age groups in a graph
sns.barplot(x="AgeGroup", y="Survived", data=df);

#check fare distribution
fare_dist = sns.distplot(df['Fare'], label="Skewness : %.2f"%(df["Fare"].skew()))
fare_dist .legend(loc="best")

plt.show()

#logarithm the fare
df['Fare_log'] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

fare_dist_w_log = sns.distplot(df['Fare_log'], label="Skewness : %.2f"%(df["Fare_log"].skew()))
fare_dist_w_log.legend(loc="best")

plt.show()

#cut into group according to its description
df['Fare_log'].describe()

bins = (-1, 2, 2.67, 3.43, 10)
group_names = [1,2,3,4]
categories = pd.cut(df['Fare_log'], bins, labels=group_names)
df['FareGroup'] = categories

print(df['FareGroup'].value_counts(ascending=True))

#fill up the missing 2 Embarked
df['Embarked'] = df['Embarked'].fillna('S')
#map into value
df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#map sex into int
df['Sex'] = df['Sex'].map( {'male': 1, 'female':0} ).astype(int)

#converse categories into int
df['FamilyGroup'] = df['FamilyGroup'].astype(int)
df['AgeGroup'] = df['AgeGroup'].astype(int)
df['FareGroup'] = df['FareGroup'].astype(int)

#prepare training dataframe
X_learning = df.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Survived', 'Ticket', 'Fare_log', 'FamilySize', 'PassengerId'], axis=1)
Y_learning = df['Survived']

#get the train and test set
from sklearn.model_selection import train_test_split

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_learning, Y_learning, test_size=num_test, random_state=31)

#test using SVC
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train,y_train)
predictions = svc.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy rate:  %f" % (accuracy_score(y_test, predictions)))

#test using RainForest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)

#using Xgboost
import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
#gbm = xgb.XGBClassifier()
gbm.fit(X_train,y_train)
predictions = gbm.predict(X_test)
