import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

train_df= pd.read_csv("Titanic_train.csv")
test_df=pd.read_csv("Titanic_test.csv")


st.dataframe(train_df) 
st.write(train_df.describe())
st.text("Null Values in Each column")
st.write(train_df.isnull().sum())

# from above, we can say that age column have missing value so we can fill by taking mean or median of them, and
#drop cabin colmn because it has so many null values it will not affect our final result 
train_df["Age"].fillna(train_df["Age"].median(skipna=True),inplace=True)
train_df["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
train_df=train_df.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)
st.text("Null Values in Each column")
st.write(train_df.isnull().sum())

train_df["Sex"]= train_df['Sex'].map( {'male': 0, 'female': 1} )
train_df["Embarked"]= train_df['Embarked'].map( {'C': 0, 'S': 1,'Q':2} )
final_train=train_df

#test data EDA 
st.text("Test Data EDA")
st.dataframe(test_df)
st.write(test_df.head())
st.write(test_df.isnull().sum())
test_df["Age"]=test_df['Age'].fillna(test_df['Age'].median(skipna=True))
test_df["Fare"]=test_df['Fare'].fillna(test_df['Fare'].median(skipna=True))
test_df=test_df.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)


st.write(test_df.isnull().sum())
test_df["Sex"]= test_df['Sex'].map( {'male': 0, 'female': 1} )
test_df["Embarked"]= test_df['Embarked'].map( {'C': 0, 'S': 1,'Q':2} )
final_test=test_df 




fig1, ax1 = plt.subplots(figsize=(15,8))
ax1= sns.kdeplot(final_train['Age'][final_train.Survived==1],color="darkturquoise",shade=True)
sns.kdeplot(final_train['Age'][final_train.Survived==0],color='lightcoral',shade=True)
plt.legend(['Survived','Died'])
plt.title('Dendity plot of Age for Surviving Population and Deceased Population')
ax1.set(xlabel='Age')
plt.xlim(-10,85)
st.pyplot(fig1)

fig2,ax2=plt.subplots(figsize=(15,10))
ax2=sns.barplot(x='Pclass', y='Survived', data=train_df, color="darkturquoise")
st.pyplot(fig2)


fig3,ax3=plt.subplots(figsize=(15,10))
ax3=sns.barplot(x='Sex',y='Survived',data=train_df,color='aquamarine')
st.pyplot(fig3)

#model training
X=final_train.drop('Survived',axis=1)
y=final_train['Survived']

model = LogisticRegression(max_iter=100)
model.fit(X,y) 

pred=model.predict(final_test)
st.write(pred)

#prediction
sex=st.selectbox("Sex",['male','female'])
pclass=st.selectbox("Pclass",[1,2,3])
age=st.slider("Age",1,100,25)
sibsp=st.number_input("SibSp",0,10,0)
parch= st.number_input("Parch",0,10,0)
fare= st.slider("Fare",0.0,600.0,50.0)
embarked= st.selectbox("Embarked",['S','C','Q'])
input_data=pd.DataFrame({
    'Pclass':[pclass],
    'Sex':[0 if sex=='male' else 1],
    'Age':[age],
    'SibSp':[sibsp],
    'Parch':[parch],
    "Fare": [fare],
    "Embarked":[0 if embarked =='C'  else 1 if embarked=='S' else 2]

})

if(st.button("Predicted survived")):
    prediction=model.predict(input_data)[0]
    st.markdown(f"**Predicted Survival:{'Yes' if prediction==1 else 'No'}**")