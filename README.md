# Modelo de Classificação de Falhas em Máquinas

Este projeto implementa um modelo de classificação de falhas em máquinas numa linha de produção.

## Abordagens Utilizadas

Foram treinados e avaliados diferentes tipos de classificação:
- **Classificação Binária**
- **Classificação Multiclasse**
- **Classificação Multilabel**

## Gerenciamento de Experimentos

O [MLflow](https://mlflow.org/) foi utilizado para salvar os artefatos gerados durante os experimentos, como modelos, métricas e parâmetros.

## Priorização de Métricas

A métrica **Recall** foi priorizada devido ao contexto industrial, onde falhas podem gerar grande impacto nos negócios da empresa. Um alto Recall pode resultar em um aumento de Falsos Positivos, levando a paralisações para investigação e conserto das máquinas. Quando a falha não é confirmada, a parada se converte em manutenção preventiva. Considerou-se que o custo dessa manutenção preventiva é proporcionalmente muito menor do que o custo de uma falha real não detectada.

## Desbalanceamento de Classes

O dataset apresentou um alto desbalanceamento entre as classes. Destaca-se que a classe **Falha Aleatória** possui apenas uma amostra, sendo considerada uma falha rara.

Diversas estratégias foram aplicadas para lidar com o desbalanceamento:
- Ajuste dos hiperparâmetros dos modelos
- Ajuste do peso das classes
- Amostragem com **SMOTE**

  ## Análise Exploratória

  O dataset de treinamento foi analisado e várias inconsistencias foram encontradas e solucionadas.
  Foram identificados:
  - falta de padronização nas colunas que indicam o ocorrência ou não de falhas
  - alto de desbalanceamento da classe que indica se houve ou nao falha e também entre as classes que indicam o tipo de falha
  - valores fora dos limites de distribuição dos dados, sendo considerados outliers

## Modelos Treinados

Foram treinados modelos de classificação binária, multiclasse e multilabel.
Dessa forma de acordo com o objetivo, pode-se classificar uma nova amostra de dados se indica falha ou não (binária), que classe foi classificada (multiclasse) ou quais classes foram identificadas, podem ser uma ou mais falhas (multilabel)

# Classe Binária

Foram selecionados 5 modelos:
- KNeighborsClassifier
- LogisticRegression
- RandomForestClassifier
- GradientBoosting
- XGBClassifier

# Estratégias

Afim de ajustar o modelo ao desbalanceamento de classes, foram aplicadas diferentes estratégias, sendo o modelo avaliado na sequência:
 
-sem smote - classes desbalanceadas
- com smote - classes iguais
- scale_pos_weight - ajuste do peso das classes
- sample_weights - ajuste do peso das classes
- class_weigt: 'balanced'

# Avaliação

No contexto analisado, foi considerado que o custo de uma máquina parada por falha irá gerar um enorme impacto financeiro, portanto adotou como objetivo uma métrica Recall alta. Um valor alto de Recall, significa que o modelo tem uma maior sensibilidade e portanto pode gerar Falsos Positivos. Pode indicar falha onde não existe, levando a uma paralisação para manutenção e conserto. O custo dessa paralisação em situações onde a falha não foi confirmada (Falso Positivo) é bem menor, sendo considerada uma manutenção preventiva.

# Multiclasse

No dataset uma coluna chamada falha_maquina, indica a ocorrência ou não de falhas, que seria especifica pelas 5 colunas de tipos de falhas.
Porém ao filtrar o dataset por essa coluna, nem todas amostras indicavam um dos 5 tipos de falhas.
Após os tratamentos dos dados. Identificou que algumas amostras mostram mais de um tipo de falha, sendo a maioria de amostras indicando apenas 1 tipo de falha, um pequeno número de amostras com 2 tipos de falhas e apenas 1 amostra indicam 3 falhas.
Essa amostra única foi considerada rara e como tem apenas uma amostra, não seria possível utlizar no treinamento e teste, pois deveriam ter amostras no conjunto de treinamento e teste. Sendo portanto, removida do dataset usado no treinamento.

# Multilabel

Por último, como haviam amostras indicando mais de um tipo de falha nas amostras, o que caracteriza um multilabel, foi feito um treinamento utilizado modelos de classificação apropriados para essa ocorrência.

- MultiOutputClassifier
- ClassifierChain

 # Estratégia

 Para esses modelos foi usado o modelo RandomForestClassifier como auxiliar.

 # Resultados

 Cada classe de falha foi identifica e as métricas precision, recall, f1-score apresentadas.
 O melhor modelo entre os dois pode ser definido comparando as mesmas métricas obtidas entre os modelos.
 Um modelo pode ser o melhor para uma classe segundo uma métrica e não ser para outra classe.
 A quantidade de amostras de cada tipo de falha, impacta na performance da modelo e na escolha da métrica de avaliação.

 # ClassifierChain
 
<img width="1769" height="1189" alt="image" src="https://github.com/user-attachments/assets/ec35a5c7-69d1-4478-b9f2-4063f6922fc7" />

# MultiOutputClassifier

<img width="1800" height="1200" alt="multi_confusion_matrices_multilabel" src="https://github.com/user-attachments/assets/c04b2830-1714-4cc7-b573-fa9412e49474" />



 


