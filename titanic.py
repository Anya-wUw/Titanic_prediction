import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('train.csv')
#print(data.columns) #названия колонок
#print(data)

#----Подготовка данных-----------
data = data.drop(["PassengerId", "Name", "Ticket"], axis=1) #axis=1 удаление колонок а не строк
#print(data)

y = data['Survived']
data = data.drop(["Survived"], axis=1)
#print(data)

# print(data.columns)
# print(data.dtypes) #типы данных

#--------проверяем в каких колонках есть пустые данные-------
print(data.columns[data.isna().any()].tolist())
# plt.scatter(data["Age"], data["Parch"])

#---------заполняем пропуски------
#средний возраст
print(data["Age"].mean(), data["Age"].median())
#print((data["Age"].median()).dtype)
#заполняем пропуски средним возрастом
data["Age"] = data["Age"].fillna(28)

#заполняем пропуски S тк в выборке их больше всего
print(data["Embarked"].value_counts())
data["Embarked"] = data["Embarked"].fillna('S')

#узнаем сколько пропущ объектов - 687 из 891
print(data['Cabin'].isna().sum(), len(data))
#тк пропусков больше половины - удалим значения
data = data.drop("Cabin", axis=1)

#проверим что пропусков вообще не осталось
#выдаст ошибку если есть
assert not data.isnull().values.any()

print(data.columns)
print(data.dtypes) #типы данных

print(data['Sex'].describe())
#заменим муж-0 жен-1
data["Sex"] = data["Sex"].astype('category').cat.codes
#print(data)

print(data["Embarked"].describe())
#бинаризация данных (get_dummies)
#создадим 3(тк 3 варианта ответа) колонки: 1)проверим S(0/1), 2)C(0/1) 3)Q(0/1) 
data = pd.get_dummies(data, columns=["Embarked"])
#print(data)

#разбиваем данные на train и val
train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.3) #обучаем модель на 70%

#обучение модели (fit)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_y)

predicted = knn.predict(val_data) #оправляет входные параметры
# print(predicted) #предикт
# print(np.array(val_y)) #правил ответы

print(accuracy_score(predicted, val_y)) #совпадение

#подбор гиперпараметра k, для более точного ответа
val_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_y)
    predicted = knn.predict(val_data)
    acc_score = accuracy_score(predicted, val_y)

    val_scores.append(acc_score)
print(val_scores)

plt.plot(list(range(1,21)), val_scores)
plt.xticks(list(range(1,21)))
plt.xlabel("кол-во соседей")
plt.ylabel("accuracy_score")
#plt.show()

#Получение ответов для тестового датасета
test_data = pd.read_csv("test.csv")
#print(test_data)

#Предобработка данных как и в треин наборе
test_data = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
#Заполняем пропуски
test_data["Age"] = test_data["Age"].fillna(28)
test_data["Embarked"] = test_data["Embarked"].fillna('S')
#Категории в бинарность(муж-0, жен-1)
test_data["Sex"] = test_data["Sex"].astype('category').cat.codes
#Разделяем данные в бимнарный вид S,C,Q
test_data=pd.get_dummies(test_data, columns=["Embarked"])
print(test_data)

#проверяем пропуски
print(test_data.isna().any())
#Fare -  True ---> пропуск есть
test_data["Fare"] = test_data["Fare"].fillna(train_data["Fare"].median())

#обучим тренировочные данные
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, y)

#print(test_data.columns)
#test_data.append = [1, 1, 19.0, 0, 1, 9.0, 0, 1, 0]
test_predicted = knn.predict(test_data)
#print(test_predicted)

test_predicted = pd.DataFrame({"Survived":test_predicted})
test_predicted["PassengerId"] = list(range(892, 892+len(test_data)))

test_predicted.to_csv("test_predicted.csv")
print(test_predicted)