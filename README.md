# Random Forest Bank

Este projeto utiliza o algoritmo Random Forest para classificar clientes de uma campanha de marketing bancário, prevendo se o cliente irá aderir ou não a uma oferta oferecida pelo banco.

**Dataset:** bank-full.csv

Dataset Bank Marketing:
https://archive.ics.uci.edu/dataset/222/bank+marketing

## Como Rodar

1. Dentro da pasta, criar e ativar o ambiente virtual.
2. Instalar as dependências presentes no arquivo **requirements.txt**.
3. Rodar o arquivo **classificador.py** para:
   - Ler o dataset.
   - Converter atributos categóricos em atributos numéricos.
   - Balancear as classes utilizando SMOTE.
   - Treinar o modelo Random Forest.
   - Realizar a busca dos melhores hiperparâmetros.
   - Avaliar o desempenho do modelo.

## Métricas Geradas

Ao final da execução são exibidos:

- Frequência das classes após o balanceamento.
- Melhores hiperparâmetros encontrados.
- Acurácia.
- Matriz de confusão.
- Especificidade.
- Sensibilidade.
