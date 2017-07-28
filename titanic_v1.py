import pandas as pd
import numpy as np

#replace file path
df = pd.read_csv("Development/Playground/Titanic/train.csv")

#data structure checking

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


# start prediction

import pandas as pd
import numpy as np

train_df = pd.read_csv("Development/Playground/Titanic/train.csv")
test_df = pd.read_csv("Development/Playground/Titanic/test.csv")

def setFamilyGroup(df): 
    #set people into parentchild and spousesib groups
    df['withP']=0
    df['withS']=0
    
    df.loc[train_df['SibSp'] > 0, 'withS'] = 1
    df.loc[train_df['Parch'] > 0, 'withP'] = 1
    
    #handle family group
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    bins = (-1, 1, 2, 3, 12)
    group_names = [1,2,3,4]
    categories = pd.cut(df['FamilySize'], bins, labels=group_names)
    df['FamilyGroup'] = categories
    
def setAgeGroup(df): 
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
        
    #separate age into 6 groups 
    bins = (-1, 15, 23, 33, 43, 53, 100)
    group_names = [1,2,3,4,5,6]
    categories = pd.cut(df['Age'], bins, labels=group_names)
    df['AgeGroup'] = categories
    
def setFareGroup(df):
  #fill the missing fare with median
  df["Fare"] = df["Fare"].fillna(df["Fare"].median())

  df['Fare_log'] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    
  bins = (-1, 2, 2.68, 3.44, 10)
  group_names = [1,2,3,4]
  categories = pd.cut(df['Fare_log'], bins, labels=group_names)
  df['FareGroup'] = categories
    
def setTitle(df):
   df['Title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
   df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
   df['Title'] = df['Title'].replace('Mlle', 'Miss')
   df['Title'] = df['Title'].replace('Ms', 'Miss')
   df['Title'] = df['Title'].replace('Mme', 'Mrs')
   title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
   df['Title'] = df['Title'].map(title_mapping)    


def getmanipulatedDF(train_df, test_df):
  dfs = [train_df, test_df]
   
  for df in dfs:
    df["Sex"] = df["Sex"].map({"male": 1, "female":0})
    
    setFamilyGroup(df) 
    
    setAgeGroup(df)
    
    setFareGroup(df)
    
    #fill up the missing 2 Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    #map into value
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    #converse categories into int
    df['FamilyGroup'] = df['FamilyGroup'].astype(int)
    df['AgeGroup'] = df['AgeGroup'].astype(int)
    df['FareGroup'] = df['FareGroup'].astype(int)
    
    setTitle(df)    
    
  return dfs[0], dfs[1]


train_df, test_df = getmanipulatedDF(train_df, test_df)

#fare_log, survived check    
facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare_log',shade= True)
facet.set(xlim=(0, train_df['Fare_log'].max()))
facet.add_legend()

plt.show()    

X_learning = train_df.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Survived', 'Ticket', 'PassengerId'], axis=1)
Y_learning = train_df['Survived']
 
X_test = test_df.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Ticket', 'PassengerId'], axis=1)    

#use Kfold validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

random_state = 19

models = []
models.append(("RFC", RandomForestClassifier(random_state=random_state)) )
models.append(("ETC", ExtraTreesClassifier(random_state=random_state)) )
models.append(("ADA", AdaBoostClassifier(random_state=random_state)) )
models.append(("GBC", GradientBoostingClassifier(random_state=random_state)) )
models.append(("SVC", SVC(random_state=random_state)) )
models.append(("LoR", LogisticRegression(random_state=random_state)) )
models.append(("LDA", LinearDiscriminantAnalysis()) )
models.append(("QDA", QuadraticDiscriminantAnalysis()) )
models.append(("DTC", DecisionTreeClassifier(random_state=random_state)) )
models.append(("XGB", xgb.XGBClassifier()) )

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10)

k_names=[]
k_means=[]
k_stds=[]

for name, model in models:
     #cross validation among models, score based on accuracy
     cv_results = model_selection.cross_val_score(model, X_learning, Y_learning, scoring='accuracy', cv=kfold )
     print("\n"+name)    
     #print("Results: "+str(cv_results))
     print("Mean: " + str(cv_results.mean()))
     print("Standard Deviation: " + str(cv_results.std()))
     k_names.append(name)
     k_means.append(cv_results.mean())
     k_stds.append(cv_results.std())

#display the result       
kfc_df = pd.DataFrame({"CrossValMeans":k_means,"CrossValerrors": k_stds,"Algorithm":k_names})  

sns.barplot("CrossValMeans","Algorithm",data = kfc_df, orient = "h",**{'xerr':k_stds})

#Using XGBoost
xgbc = xgb.XGBClassifier()
xgbc.fit(X_learning,Y_learning)
predictions = xgbc.predict(X_test)

output = pd.DataFrame({ 'PassengerId' : test_df['PassengerId'], 'Survived': predictions })

#replace file path
output.to_csv("Development/Playground/Titanic/out_xgb.csv", index = False)

#0.78469 = XGB 
