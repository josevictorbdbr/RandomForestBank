import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

#Abrir o arquivo de dados
dados = pd.read_csv('bank-full.csv', sep = ';')
#Separar atributos e classe
dados_atributos = dados.drop(columns=['y'])
dados_classe = dados['y']


#Converter o texto para numeros
dados_atributos = pd.get_dummies(
    dados_atributos,
    drop_first=True
)

#Balancear os dados
resampler = SMOTE(random_state=42) 

atributos_b, classes_b = \
    resampler.fit_resample(
    dados_atributos,
    dados_classe
)

print('#### FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO ###')
class_count = Counter(classes_b)
print(class_count)

#segmentar os dados em dados para treinamento e dados para teste
atributos_train, atributos_teste, \
    classe_train,classe_test = train_test_split(
            atributos_b,
            classes_b, 
            test_size=0.3,
            random_state=42
        )


#HIPERPARAMETRIZAÇÃO DA RANDOM FOREST
#Definir os domínios para os hiperparâmetros
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

#criar a grade de valores
rf_grid={
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split':min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier()
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=5,
    cv=3,
    verbose=2,
    n_jobs=1
)
rf_hyperparameters.fit(atributos_train, classe_train)

#Mostrar o resultado da hiperparametrização
print('Melhores parametros:')
pprint(rf_hyperparameters.best_params_)

#Recuperar melhor modelo
melhor_rf = rf_hyperparameters.best_estimator_

predicoes = melhor_rf.predict(
    atributos_teste
)

#Calcular Acuracia
acuracia = accuracy_score(
    classe_test,
    predicoes
)
print("Acuracia: ", acuracia)


#Matriz de confusão
cm = confusion_matrix(
    classe_test,
    predicoes
)

print(cm)

tn,fp,fn,tp = confusion_matrix(
    classe_test,
    predicoes
).ravel()


especificidade = tn/(tn+fp)
sensibilidade = tp/(tp+fn)

print("Especificidade:", especificidade)
print("Sensibilidade:", sensibilidade)
