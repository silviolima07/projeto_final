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


     

---
