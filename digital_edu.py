# ГИПОТЕЗА: люди, учащиеся в универе, с наибольшей вероятностью купят курс
#Импорт нужных библиотек
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')

#Очистка данных
def sex_apply(sex):
    if sex == 2:
        return 0 #мужчины
    return 1 #женщины
df['sex'] = df['sex'].apply(sex_apply)


def edu_status_apply(edu_status):
    if edu_status == "Undergraduate applicant":
        return 0
    elif (edu_status == "Student (Master's) "
        or edu_status == "Student (Bachelor's)" 
        or edu_status == "Student (Specialist)"):
        return 1 #учащиеся
    elif  (edu_status == "Alumnus (Master's)" 
        or  edu_status == "Alumnus (Bachelor's)" 
        or edu_status == "Alumnus (Specialist)"):
        return 2 #окончившие
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)


def  occu_type_apply(occupation_type):
    if occupation_type == "university":
        return 1
    else:
        return 0
df['occupation_type'] = df['occupation_type'].apply(occu_type_apply)


def langsCleaner(lang):
    return lang.count(';')+1
df['langs'] = df['langs'].apply(langsCleaner)

#Удаление ненужных данных
df.drop(['id', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'education_form',
'relation', 'last_seen', 'occupation_name', 'life_main', 'people_main', 'city', 'bdate',
'career_start', 'career_end'], axis = 1, inplace = True)

#Создание модели

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


x = df.drop('result', axis = 1)
y = df['result']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(train_x, train_y)

pred_y = classifier.predict(test_x)
print('Точность теста:', round(accuracy_score(test_y, pred_y)*100, 2))