"""

Modelo de classificação de falhas em maquinas numa linha de producao.
Foram treinados e avaliados:
- Classificação Binaria
- Classificação Multiclasse
- Classificação Multilabel

O MLFlow foi usado para salvar os artefatos gerados durante os experimentos.
A métrica Recall foi priorizada devido ao contexto onde as falhas podem ocorrer e gerar um grande impacto nos negocios da empresa.
Um alto Recall pode gerar Falsos Positivos, o que implica numa paralisação para investigação e conserto da maquina.
Se a falha não for confirmada, a paralisação se converte numa manutenção preventiva.
Foi considerado que o custo dessa paralisação é proporcionalmente bem menor do que se a falha fosse real.

O dataset apresentou um alto desbalanceamento de classes.
A classe Falha Alestória tem apenas uma amostra no dataset, sendo assim considerado um erro raro.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import mlflow
from mlflow.models import infer_signature
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore")

def plot_histograms(df, column, category_col, bins, custom_labels=None):
    df_plot = df.copy()

    if custom_labels and category_col in custom_labels:
        label_list = custom_labels[category_col]
        unique_values = sorted(df_plot[category_col].dropna().unique())

        label_map = {val: label_list[i] for i, val in enumerate(unique_values) if i < len(label_list)}
        df_plot[category_col] = df_plot[category_col].map(label_map)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df_plot,
        x=column,
        hue=category_col,
        kde=True,
        bins=bins,
        palette='tab10',
        element='step',
        fill=True,
    )
    plt.title(f'Histograma de {column} por {category_col}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.show()
    print( ' ')

def plot_stacked_barplot(df, column, category_col=None, custom_labels=None):
    df_plot = df.copy()

    if custom_labels and column in custom_labels:
        label_list = custom_labels[column]
        unique_values = sorted(df_plot[column].dropna().unique())
        label_map = {val: label_list[i] for i, val in enumerate(unique_values) if i < len(label_list)}
        df_plot[column] = df_plot[column].map(label_map)

    if category_col and custom_labels and category_col in custom_labels:
        label_list = custom_labels[category_col]
        unique_values = sorted(df_plot[category_col].dropna().unique())
        label_map = {val: label_list[i] for i, val in enumerate(unique_values) if i < len(label_list)}
        df_plot[category_col] = df_plot[category_col].map(label_map)

    count_data = pd.crosstab(df_plot[column], df_plot[category_col])

    count_data.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='tab10')

    plt.title(f'Barras Empilhadas de {column}' + (f' por {category_col}' if category_col else ''))
    plt.xlabel(column)
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print('')

def plot_boxplots_category(df, column, category_col, custom_labels=None):
    df_plot = df.copy()
    if custom_labels and category_col in custom_labels:
        label_list = custom_labels[category_col]
        unique_values = sorted(df_plot[category_col].dropna().unique())

        label_map = {val: label_list[i] for i, val in enumerate(unique_values) if i < len(label_list)}
        df_plot[category_col] = df_plot[category_col].map(label_map)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_plot, x=category_col, y=column, hue=category_col, palette='tab10')
    plt.title(f'Boxplot de {column} por {category_col}')
    plt.xlabel(category_col)
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()
    print( ' ')

def plot_correlation_matrix(df):
    corr_df = df.select_dtypes(include=['float', 'int']).columns
    # Ensure the target column is included if it exists
    if 'classe_binaria' in df.columns:
        corr_df = corr_df.tolist()
        if 'classe_binaria' not in corr_df:
            corr_df.append('classe_binaria')
        corr_df = df[corr_df]
    else:
        corr_df = df[corr_df]

    plt.figure(figsize=(14, 12))
    correlation_matrix = corr_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.show()

def remove_outliers_iqr(df, column, category_col=None):
    if category_col:
        filtered_df = pd.DataFrame()
        for category in df[category_col].unique():
            subset = df[df[category_col] == category]
            Q1 = subset[column].quantile(0.25)
            Q3 = subset[column].quantile(0.75)
            IQR = Q3 - Q1
            mask = (subset[column] >= Q1 - 1.5 * IQR) & (subset[column] <= Q3 + 1.5 * IQR)
            filtered_df = pd.concat([filtered_df, subset[mask]])
        return filtered_df
    else:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
        return df[mask]

df = pd.read_csv('../bootcamp_train.csv')

df

"""# Identificar valores nulos"""

df.isna().sum()

"""# Identificar o tipo dos dados"""

df.info()

"""# Estatistica descritiva"""

df.describe().T

"""# Frequencias das classes

- Colunas com alta cardinalidade não expressam um padrao e não ajudam no treinamento do modelo e portanto são removidas. As colunas id e id_produto se encaixam nessa situacao.
"""

df.nunique()

df.head()

"""# As colunas referentes as falhas apresentaram diversos valores alem de sim e nao. Isso foi identificado e corrigido.

# coluna falha_maquina
"""

df.falha_maquina.value_counts()

df.falha_maquina.unique()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    'Não': 'nao', 'N': 'nao', '0': 'nao', 'não': 'nao',
    'Sim': 'sim', '1': 'sim', 'y': 'sim', 'sim': 'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['falha_maquina_map'] = df['falha_maquina'].map(mapeamento)

"""# Dataset desbalanceado"""

df.falha_maquina_map.value_counts()

# Checar se todas ocorrencias foram ajustadas
34598 + 662 # Total de linhas no dataset

"""# Multiclasse - 5 tipos de falhas

# coluna FDF (Falha Desgaste Ferramenta)
"""

df['FDF (Falha Desgaste Ferramenta)'].value_counts()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    'False': 'nao', 'N': 'nao', '0': 'nao',
    'True': 'sim', '1': 'sim', '-': 'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['FDF (Falha Desgaste Ferramenta)_map']= df['FDF (Falha Desgaste Ferramenta)'].map(mapeamento)

"""# Dataset desbalanceado"""

df['FDF (Falha Desgaste Ferramenta)_map'].value_counts()

# Checar se todas ocorrencias foram ajustadas
35119 + 141 # Total de linhas no dataset

"""# coluna FDC (Falha Dissipacao Calor)"""

df['FDC (Falha Dissipacao Calor)'].value_counts()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    'False' : 'nao', 'nao': 'nao', '0': 'nao',
    'True': 'sim', 'y': 'sim', '1': 'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['FDC (Falha Dissipacao Calor)_map']= df['FDC (Falha Dissipacao Calor)'].map(mapeamento)

"""# Dataset desbalanceado"""

df['FDC (Falha Dissipacao Calor)_map'].value_counts()

# Checar se todas ocorrencias foram ajustadas
35035 + 225 # Total de linhas no dataset

"""# coluna FP (Falha Potencia)"""

df['FP (Falha Potencia)'].value_counts()

df['FP (Falha Potencia)'].unique()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    'Não': 'nao', 'não': 'nao', 'N':'nao', '0':'nao',
    'Sim': 'sim', 'sim': 'sim', '1': 'sim', 'y':'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['FP (Falha Potencia)_map']= df['FP (Falha Potencia)'].map(mapeamento)

"""# Dataset desbalanceado"""

df['FP (Falha Potencia)_map'].value_counts()

# Checar se todas ocorrencias foram ajustadas
35134 + 126 # Total de linhas no dataset

"""#  coluna FTE (Falha Tensao Excessiva)"""

df['FTE (Falha Tensao Excessiva)'].value_counts()

df['FTE (Falha Tensao Excessiva)'].unique()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    False: 'nao',
    True: 'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['FTE (Falha Tensao Excessiva)_map']= df['FTE (Falha Tensao Excessiva)'].map(mapeamento)

"""# Dataset desbalanceado"""

df['FTE (Falha Tensao Excessiva)_map'].value_counts()

# Checar se todas ocorrencias foram ajustadas
35090 + 170 # Total de linhas no dataset

"""# coluna FA (Falha Aleatoria)"""

df['FA (Falha Aleatoria)'].value_counts()

df['FA (Falha Aleatoria)'].unique()

# Mapear os valores para 'sim' e 'nao'
mapeamento = {
    'Não': 'nao', 'não': 'nao', '-':'nao', '0':'nao',
    'Sim': 'sim', 'sim': 'sim', '1': 'sim'
}

# Aplicar o mapeamento na coluna falha_maquina
df['FA (Falha Aleatoria)_map']= df['FA (Falha Aleatoria)'].map(mapeamento)

"""# Dataset desbalanceado"""

df['FA (Falha Aleatoria)_map'].value_counts()

# Checar se todas ocorrencias foram ajustadas
35186 + 74 # Total de linhas no dataset

"""# Colunas de indicadores de falha ajustadas para sim e nao"""

falhas = ['falha_maquina_map',
 'FDF (Falha Desgaste Ferramenta)_map',
 'FDC (Falha Dissipacao Calor)_map',
 'FP (Falha Potencia)_map',
 'FTE (Falha Tensao Excessiva)_map',
 'FA (Falha Aleatoria)_map']

"""# Percentual de exemplos com sim e nao"""

for coluna in falhas:
    print("Coluna (%):\n", df[coluna].value_counts(True) * 100, "\n-------------")

"""# Classificacao Binaria

# As colunas referentes a falhas apresentam alta taxa de desbalanceamento.
# A tarefa de classificação pode ser feita em duas etapas.
# Treinar um modelo com um conjunto de dados que indica se existe falha ou nao. Dessa forma, podemos aplicar esse modelo para  identificar se um novo conjunto de dados indica falha ou não.
# Treinar um segundo modelo com apenas com dados indicando os 5 tipos de falhas. Dessa forma se um novo conjunto de dados foi classificado com falha pelo primeiro modelo, o segundo modelo pode classificar o tipo de falha.

# Será feito o treinamento e avaliação para classe binária apenas a coluna falha_maquina_map, que foi ajustada a partir da coluna falha_maquina original.
"""

df2 = df [['id', 'id_produto', 'tipo', 'temperatura_ar', 'temperatura_processo',
       'umidade_relativa', 'velocidade_rotacional', 'torque',
       'desgaste_da_ferramenta', 'falha_maquina_map']].copy()

"""# Ajustar colunas com valores ausentes"""

colunas = df2.columns.tolist()
colunas

"""# Identificar o percentual de dados ausentes"""

for col in colunas:
    total_linhas = df.shape[0]
    if df2[col].isna().sum() > 0:
      print("Colunas:",col)
      print(df2[col].isna().sum(), 'dados ausentes/nulos de  ',total_linhas)
      print('Taxa: ',round(df2[col].isna().sum()/total_linhas * 100,2), "% ")
      print(" ")

"""# Aplicar o fillna nas colunas e preencher com o valor da mediana.
# Foi usada a mediana pois nao é impactada por outliers.
"""

df3 = df2.copy()
print("Aplicar a mediana da coluna\n")
for col in colunas:
    total_linhas = df.shape[0]
    if df2[col].isna().sum() > 0:
      print("Colunas:",col)
      df3[col].fillna(df3[col].median(), inplace=True)

df3.head()

df3.head()

"""# coluna falha_maquina_map, a classe binaria, sera convertida em numerica, para se ajustar o valor esperado por alguns modelos que serão treinados"""

le = LabelEncoder()
df3['falha_maquina_map'] = le.fit_transform(df3['falha_maquina_map'])

df3.head()

df_binario = df3.copy()
df_binario = df_binario.rename(columns={'falha_maquina_map': 'classe_binaria'})

df_binario.head()

df_binario.shape

"""# Visualizar a distribuicao de cada coluna"""

df_binario.hist();

numerical = df_binario.select_dtypes(include=np.number).columns.tolist()
categorical = df_binario.columns.difference(numerical).to_list()

print('Colunas numericas:', numerical)
print('Colunas categoricas:', categorical)

bins = int(np.sqrt(len(df_binario)))
for column in numerical:
  if column != 'classe_binaria' and column != 'id' and column != 'id_produto':
    plot_histograms(df_binario, column, 'classe_binaria', bins)

for column in categorical:
    if column != 'classe_binaria' and column != 'id_produto':
        plot_stacked_barplot(df_binario, column, 'classe_binaria')

"""# Boxplot
- identificar valores outliers, alem do limites aceitáveis.
- Limite definido pelos quartis gerados na distribuição dos dados.
"""

for column in numerical:
  if column != 'classe_binaria':
    plot_boxplots_category(df_binario, column, 'classe_binaria')

"""# Identificar correlação"""

# Select only numerical columns for correlation calculation
numerical_df = df_binario.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Get the correlation of each numerical column with 'classe_binaria'
correlation_with_classe_binaria = correlation_matrix['classe_binaria']

# Display the correlations
display(correlation_with_classe_binaria)

plot_correlation_matrix(df_binario)

"""# DATASET DESBALANCEADO

# Checar distribuição das classes
"""

df_binario.classe_binaria.value_counts()

# Checar se todas ocorrencias foram ajustadas
34598 + 662 # Total de linhas no dataset

"""# Identificar e remover Outliers"""

print("Antes:", df_binario.shape)

# Criar uma cópia do DataFrame para não modificar o original diretamente
df_cleaned = df_binario.copy()

# Identificar as colunas numéricas, excluindo 'GradeClass' pois será a coluna categoria
numerical_cols_for_outliers = df_cleaned.select_dtypes(include=np.number).columns.tolist()
if 'classe_binaria' in numerical_cols_for_outliers:
    numerical_cols_for_outliers.remove('classe_binaria')

print(f"Colunas numéricas para detecção de outliers por classe_binaria:\n{numerical_cols_for_outliers}")

# Aplicar a remoção de outliers para cada coluna numérica, considerando GradeClass
for col in numerical_cols_for_outliers:
    initial_rows = len(df_cleaned)
    df_cleaned = remove_outliers_iqr(df_cleaned, col, 'classe_binaria')
    removed_rows = initial_rows - len(df_cleaned)
    print(f"Após remover outliers em '{col}' por classe_binaria: {len(df_cleaned)} linhas restantes ({removed_rows} removidas)")

print("\nDataFrame após remover outliers de todas as colunas numéricas por classe_binaria:")
display(df_cleaned.head())
print(f"\nShape final do DataFrame: {df_cleaned.shape}")

"""# Treinamento dos modelos

# Dividir conjunto das variaveis dependentes e independentes.
"""

X = df_cleaned.drop(['id', 'id_produto', 'classe_binaria'], axis=1).copy() # Independentes
y = df_cleaned.classe_binaria.copy() # Dependentes

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Colunas numericas:", numeric_cols, "\nColunas categoricas:", categorical_cols)

"""# Dividir em treino e teste"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, X_test.shape

"""# Baseline
- metricas
- precision_recall_curve
- matriz de confusao
- salvos pelo mlflow
"""

from sklearn.dummy import DummyClassifier
import mlflow
from mlflow.models import infer_signature
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


# Função para logar matriz de confusão
def log_confusion_matrix(y_true, y_pred, nome_modelo, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.tight_layout()
    path = f"report_binario\BASELINE\confusion_matrix_{nome_modelo}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()


# Função auxiliar para logar no MLflow
def avaliar_e_logar_baseline(model, nome_execucao, X_train, y_train, X_test, y_test, labels):
    with mlflow.start_run(run_name=nome_execucao):
        # Make predictions
        y_pred = model.predict(X_test)

        # Get predicted probabilities for precision-recall curve if available
        y_scores = None
        if hasattr(model, "predict_proba"):
             y_scores = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (class 1)

        # Calculate metrics
        report_dict = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], output_dict=True)

        # Log metrics
        mlflow.log_metric("accuracy", report_dict.get('accuracy', 0.0))

        # Log precision, recall, and f1-score for the positive class (binary average)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)

        # Safely access metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)


        # Log classification report as artifact
        df_report = pd.DataFrame(report_dict).transpose()
        csv_path = f"classification_report\BASELINE_binario_{nome_execucao}.csv"
        df_report.to_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred, nome_execucao, [str(l) for l in labels])

        # Generate and log precision-recall curve if probabilities are available
        if y_scores is not None:
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {nome_execucao}')
            plt.grid(True)
            pr_curve_path = f"report_binario\\BASELINE\\precision_recall_curve_{nome_execucao}.png"
            plt.savefig(pr_curve_path)
            mlflow.log_artifact(pr_curve_path)
            plt.close()


        print(f"\n### {nome_execucao} ###")
        print(classification_report(y_test, y_pred, target_names=[str(l) for l in labels]))

        mlflow.end_run()

# ================================
# Executar experimentos e salvar no MLflow
# ================================
mlflow.set_experiment("Baseline binario")

one = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = one.fit_transform(X_train)
X_test_encoded = one.transform(X_test)

labels = ['nao', 'sim'] # Assuming this is consistent with your LabelEncoder mapping

# Baseline 1: sempre prever a classe majoritária
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_encoded, y_train)
avaliar_e_logar_baseline(dummy, "DummyClassifier_Baseline", X_train_encoded, y_train, X_test_encoded, y_test, labels)


# Baseline 2: Logistic Regression sem ajuste
model = LogisticRegression(max_iter=1000)
model.fit(X_train_encoded, y_train)
avaliar_e_logar_baseline(model, "LogisticRegression_Baseline", X_train_encoded, y_train, X_test_encoded, y_test, labels)

#Counter(y_train)
# Counter({0: 27678, 1: 530})

"""# Treinamento e avaliacao com MLFlow

# Foram treinados 5 modelos.
# Os hiperparametros foram ajustados para o dataset desbalanceado.
- metricas
- precision_recall_curve
- matriz de confusao
- salvos pelo mlflow
# Estrategias:
- sem smote   - classes desbalanceadas
- com smote   - classes iguais
- scale_pos_weight - ajuste do peso das classes
- sample_weights   - ajuste do peso das classes
- class_weigt: 'balanced'
"""

# ================================
# Função para logar matriz de confusão
# ================================
def log_confusion_matrix(y_true, y_pred, nome_modelo, estrategia, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {nome_modelo}_{estrategia}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.tight_layout()
    # Use nome_modelo for the path
    path = f"report_binario\confusion_matrix_{nome_modelo}_{estrategia}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

# Função auxiliar para logar no MLflow
# --------------------------
# Added 'model' parameter to pass the actual fitted model object
def avaliar_e_logar(model_pipeline, nome_execucao, X_train, y_train, X_test, y_test, resultados, labels, apply_smote=False):
    with mlflow.start_run(run_name=nome_execucao):
        # Apply preprocessing
        print("Preprocessing...OneHot e MinMax")
        X_train_processed = model_pipeline.named_steps['preprocessing'].fit_transform(X_train)
        X_test_processed = model_pipeline.named_steps['preprocessing'].transform(X_test)

        classifier = model_pipeline.named_steps['classifier']
        print("Model:", classifier.__class__.__name__)

        # Determine training data based on apply_smote flag
        if apply_smote:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_fit, y_fit = smote.fit_resample(X_train_processed, y_train)
            print("SMOTE applied.")
            estrategia = 'Smote'
            print("Estrategia:", estrategia)
        else:
            X_fit, y_fit = X_train_processed, y_train.copy()
            estrategia = 'sem_smote'
            print("Estrategia:", estrategia)

            # Print specific hyperparameters for verification (when not using SMOTE)
            if isinstance(classifier, (XGBClassifier, RandomForestClassifier)):
                if hasattr(classifier, 'scale_pos_weight'):
                     print(f"\nXGBoost - Ajustar pesos das classes com scale_pos_weight: {classifier.get_params().get('scale_pos_weight')}")
                     # If using scale_pos_weight, the strategy is scale_pos_weight
                     estrategia = 'scale_pos_weight' if isinstance(classifier, XGBClassifier) else estrategia
                     print("Estrategia:",estrategia)
                if hasattr(classifier, 'class_weight') and classifier.get_params().get('class_weight') == 'balanced':
                     print(f"RandomForest class_weight: {classifier.get_params().get('class_weight')}")
                     # If using class_weight='balanced', the strategy is class_weight
                     estrategia = 'class_weight' if isinstance(classifier, RandomForestClassifier) else estrategia
                     print("Estrategia:",estrategia)


        # Treina o pipeline
        print("Training...")
        # Fit the classifier using the appropriate data (X_fit, y_fit)
        if classifier.__class__.__name__ == 'GradientBoostingClassifier' and not apply_smote :
             # Apply sample weights only when not using SMOTE for GradientBoosting
             classifier.fit(X_fit, y_fit, sample_weight=compute_sample_weight(class_weight='balanced', y=y_fit))
             print("GradientBoostingClassifier with sample_weights")
             estrategia = 'sample_weights' # Update strategy if sample_weights is used
             print("Estrategia:",estrategia)
        else:
             # Fit all other models, including GB when using SMOTE
             classifier.fit(X_fit, y_fit)




        # Infer signature AFTER fitting
        input_example = X_train[:5] # Use original X_train for input example
        input_example_processed = model_pipeline.named_steps['preprocessing'].transform(input_example) # Preprocess the input example
        signature = infer_signature(input_example_processed, model_pipeline.named_steps['classifier'].predict(input_example_processed))


        mlflow.sklearn.log_model(
            sk_model=model_pipeline.named_steps['classifier'], # Log only the classifier
            name="model", # Define an artifact path
            signature=signature,
            input_example=input_example_processed # Use processed input example for logging
        )


        # Make predictions using the fitted pipeline
        y_pred = model_pipeline.named_steps['classifier'].predict(X_test_processed)

        # Get predicted probabilities for precision-recall curve
        y_scores = None
        if hasattr(classifier, "predict_proba"):
             y_scores = classifier.predict_proba(X_test_processed)[:, 1] # Get probabilities for the positive class (class 1)


        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        # Calculate precision, recall, f1-score for the positive class (binary average)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)


        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        print("\nMetricas: accuracy, precision, recall e f1-score  salvas no mlflow")

        # Relatório detalhado (by class)
        # Use the labels passed to the function
        report_dict = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()


        print("\nMatriz de confusao salva no mlflow\n")
        # Use nome_execucao for the CSV path
        csv_path = f"classification_report\Binario_{nome_execucao}_{estrategia}.csv"
        df_report.to_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # Matriz de confusão
        # Pass nome_execucao as nome_modelo
        log_confusion_matrix(y_test, y_pred, nome_execucao, estrategia, [str(l) for l in labels])

        # Generate and log precision-recall curve if probabilities are available
        if y_scores is not None:
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {nome_execucao}_{estrategia}')
            plt.grid(True)
            pr_curve_path = f"report_binario\precision_recall_curve_{nome_execucao}_{estrategia}.png"
            plt.savefig(pr_curve_path)
            mlflow.log_artifact(pr_curve_path)
            plt.close()
            print(f"Precision_Recall_Curve salva no mlflow")


        # Guardar para ranking final (include per-class metrics)
        # Safely access metrics for class '1' for the result_entry
        class_1_metrics_for_results = report_dict.get('1', {})
        result_entry = {
            "Modelo": nome_execucao,
            "Accuracy": acc,
            # Use the directly calculated metrics for the results dataframe
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            'Estrategia': estrategia.upper()
        }


        resultados.append(result_entry)


        # Show in console
        print(f"\n### {nome_execucao} ###")
        print(classification_report(y_test, y_pred, target_names=[str(l) for l in labels]))

        mlflow.end_run()
# Calcule os pesos da amostra
# 'balanced' ajusta os pesos inversamente proporcionais à frequência das classes
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Compute the positive class weight
# Counter(y_train)
# Counter({0: 27678, 1: 530})
pos_class_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train) # 27678 / 530

#print("Pos class weight:",pos_class_weight)
# ================================
# Lista de modelos a testar
# ================================
modelos = {
    'KNN': KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=100),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42), # No smote class_weigt= 'None'
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', scale_pos_weight=pos_class_weight),
}

labels = le.classes_


# ================================
# Executar experimentos e salvar no MLflow
# ================================
mlflow.set_experiment("Classe binaria")

resultados = []
# ================================
# Executar experiments and log to MLflow
# ================================
n=1

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (if any, though none expected here)
)

for nome, modelo in modelos.items():
    print("Modelo", n, "de", len(modelos))
    n+=1

    print(f"\n### Executing {nome} ###")

    # Create a pipeline for the current model
    model_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', modelo) # Use the model instance from the dictionary
    ])

    # 1) Without balancing
    print("\nWithout balancing")
    # Pass the pipeline instance to the function
    #avaliar_e_logar(model_pipeline, f"{nome}_sem_balanceamento", X_train, y_train, X_test, y_test, resultados, labels, apply_smote=False)
    avaliar_e_logar(model_pipeline, f"{nome}", X_train, y_train, X_test, y_test, resultados, labels, apply_smote=False)


    # 2) With SMOTE (Skip SMOTE for XGBoost as it uses scale_pos_weight)
    if not isinstance(modelo, XGBClassifier):
        print("With SMOTE")
        # Special handling for RandomForest with SMOTE: set class_weight to None
        if isinstance(modelo, RandomForestClassifier):
             model_pipeline.named_steps['classifier'].set_params(class_weight=None)
             print("RandomForest class_weight set to None for SMOTE.")

        # Pass the pipeline instance and resampled data to the function
        #avaliar_e_logar(model_pipeline, f"{nome}_com_SMOTE", X_train, y_train, X_test, y_test, resultados, labels, apply_smote=True)
        avaliar_e_logar(model_pipeline, f"{nome}", X_train, y_train, X_test, y_test, resultados, labels, apply_smote=True)



    print("\n------------------------------\n")

# ================================
# Ranking final em Pandas
# ================================
df_resultados = pd.DataFrame(resultados).sort_values(by="F1-score", ascending=False)
print("\n🏆 Ranking dos modelos:")
print(df_resultados)

"""# A metrica definida como objetivo foi Recall.
# No contexto onde os dados foram coletados, uma linha de produção, o custo de uma maquina quebrada por alguma falha foi considerado alto.
# Um Recall alto, pode gerar falsos positivos, o que iria gerar uma paralização para manutenção, caso seja um alarme falso, isso seria menos impactante do que se tivesse que trocar alguma peça defeituosa, sendo a parada considerada uma manutenção preventiva. A metrica usada pode ser alterada para precision ou f1-score ao longo do tempo de acordo com as prioridades definidas pela empresa.
"""

df_resultados = pd.DataFrame(resultados).sort_values(by="Recall", ascending=False).reset_index()
print("\n🏆 Ranking dos modelos:")
colunas = ['Modelo', 'Recall','Precision', 'F1-score', 'Estrategia']
print(df_resultados[colunas])

best_model = df_resultados.loc[0, 'Modelo']
print("Melhor modelo:", best_model)

"""# Avaliacao com dados de teste"""

df_test = pd.read_csv('../bootcamp_test.csv')

df_test.head()

df_test.columns

df_test_processed = df_test.drop(['id','id_produto'], axis=1)

df_test_processed

pred = np.round(model_pipeline.predict_proba(df_test_processed),2)

print("Probabilidade:", pred)

"""# Best model from mlflow"""

# Best model
best_gb_sample_weights_row = df_resultados[
    (df_resultados['Modelo'] == 'GradientBoosting') &
    (df_resultados['Estrategia'] == 'SAMPLE_WEIGHTS')
].iloc[0]

# Get the run_id from the original runs DataFrame based on the model and strategy
# We need to search the original runs DataFrame to find the run_id that corresponds
# to the GradientBoosting model run with the 'SAMPLE_WEIGHTS' strategy.
# We can do this by filtering the original runs DataFrame by run name and potentially
# by looking at the logged metrics (like recall_1) to match the best model from df_resultados.

# Let's get all runs for the "Classe binaria" experiment again to ensure we have the latest.
experiment_name = "Classe binaria"
all_runs = mlflow.search_runs(experiment_names=[experiment_name])

# Find the run_id in all_runs that corresponds to the best_gb_sample_weights_row
# We can filter by run name and then find the run that has the matching metrics (e.g., recall_1)
# This is a bit indirect, but based on the available tags, filtering by run name and then finding the best one is the most feasible.
gb_runs = all_runs[all_runs['tags.mlflow.runName'] == 'GradientBoosting']

# From the gb_runs, find the one that corresponds to the best_gb_sample_weights_row
# We can match by the recall_1 metric, assuming recall_1 was logged for these runs.
best_run_id = gb_runs[gb_runs['metrics.recall'] == best_gb_sample_weights_row['Recall']].iloc[0].run_id


print(f"Run ID do melhor GradientBoosting com sample_weights: {best_run_id}")

# Load the best classifier model (since the full pipeline wasn't logged)
# The artifact path was set to "model" in the avaliar_e_logar function
loaded_classifier = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

# Recreate the preprocessor using the same settings as during training
# Make sure numeric_cols and categorical_cols are defined and correct
numeric_cols = df_test_processed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_test_processed.select_dtypes(exclude=np.number).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit the preprocessor on the training data to learn the transformations (important!)
# We need X_train from the previous steps. Make sure it's available in the environment.
# If X_train is not available, you would need to load it or re-run the previous steps.
# Assuming X_train is available:
preprocessor.fit(X_train)


# Apply the preprocessor to the test data
df_test_processed_transformed = preprocessor.transform(df_test_processed)


# Make predictions of probabilities on the preprocessed test data using the loaded classifier
# Get the probabilities for the positive class (class 1)
prediction_probabilities = loaded_classifier.predict_proba(df_test_processed_transformed)[:, 1]


# You can display the predictions
print("\nPrevisões de probabilidade de falha no dataset de teste:")
display(prediction_probabilities)

df_test.columns

# Add the prediction probabilities to the original test DataFrame
df_test['probabilidade_falha (%)'] = prediction_probabilities * 100
df_test.sort_values(by='probabilidade_falha (%)', ascending=False, inplace=True)

# Display the updated test DataFrame with the probabilities
print("DataFrame de teste com as probabilidades de falha:")
display(df_test[['id', 'id_produto', 'probabilidade_falha (%)']])

"""# Treinar modelo GradientBoostingClassifier e aplicar a estrategia"""

# Load the test data
df_test = pd.read_csv('../bootcamp_test.csv')

# Drop 'id' and 'id_produto' columns from the test data
df_test_processed = df_test.drop(['id','id_produto'], axis=1)

# Recreate the preprocessor using the same settings as during training
# Make sure numeric_cols and categorical_cols are defined and correct
numeric_cols = df_test_processed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_test_processed.columns.difference(numeric_cols).to_list()
if 'classe_binaria' in numeric_cols:
    numeric_cols.remove('classe_binaria')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit the preprocessor on the training data to learn the transformations
# Assuming X_train is available from previous steps
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the preprocessor to the test data
df_test_processed_transformed = preprocessor.transform(df_test_processed)

# Define the best model with the appropriate strategy
best_model_no_mlflow = GradientBoostingClassifier()

# Calculate sample weights for the training data
# Assuming y_train is available from previous steps
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train the best model using the preprocessed training data and sample weights
print("Treinando o modelo GradientBoosting com sample_weights...")
best_model_no_mlflow.fit(X_train_processed, y_train, sample_weight=sample_weights)
print("Treinamento concluído.")

# Make predictions of probabilities on the preprocessed test data
prediction_probabilities_no_mlflow = best_model_no_mlflow.predict_proba(df_test_processed_transformed)[:, 1]

# Add the prediction probabilities to the original test DataFrame
df_test['probabilidade_falha (%)'] = prediction_probabilities_no_mlflow * 100
df_test.sort_values(by='probabilidade_falha (%)', ascending=False, inplace=True)

# Display the updated test DataFrame with the probabilities
print("\nDataFrame de teste com as probabilidades de falha (sem MLflow):")
display(df_test[['id', 'id_produto', 'probabilidade_falha (%)']])

"""# Multiclasse"""

df.isna().sum()

df_falhas = df[['id', 'id_produto','falha_maquina_map', 'FDF (Falha Desgaste Ferramenta)_map',
                       'FDC (Falha Dissipacao Calor)_map',
                       'FP (Falha Potencia)_map',
                       'FTE (Falha Tensao Excessiva)_map',
                       'FA (Falha Aleatoria)_map']]
df_falhas_sim = df_falhas.loc[df_falhas['falha_maquina_map'] == 'sim']
df_falhas_sim.drop('falha_maquina_map', axis=1, inplace=True)

"""# Amostras filtradas pela coluna falha_maquina indicando que existe alguma falha."""

df_falhas_sim

# Select the columns representing individual failure types
individual_failures_sim_cols = ['id', 'id_produto', 'FDF (Falha Desgaste Ferramenta)_map',
                               'FDC (Falha Dissipacao Calor)_map',
                               'FP (Falha Potencia)_map',
                               'FTE (Falha Tensao Excessiva)_map',
                               'FA (Falha Aleatoria)_map']

# Count the number of 'sim' occurrences in each row for these columns
df_falhas_sim['sim_count'] = df_falhas_sim[individual_failures_sim_cols].apply(
    lambda row: (row == 'sim').sum(), axis=1
)

# Display rows where sim_count is greater than 1
multilabel_cases = df_falhas_sim[df_falhas_sim['sim_count'] > 1]

print("Casos com múltiplos tipos de falha (mais de um 'sim' por linha):")
display(multilabel_cases)

print("\nContagem total de 'sim' por linha:")
display(df_falhas_sim['sim_count'].value_counts())

df_falhas_sim.loc[df_falhas_sim['sim_count'] == 3]



"""# Nas amostras onde aparece mais de um 'sim' indicando um tipo de falha.
# A amostra sera atribuida a falha que apresentar o primeiro 'sim' na respectiva linha.
"""

# Filter df_falhas_sim to include only rows with at least one 'sim'
df_multiclasse_temp = df_falhas_sim[df_falhas_sim['sim_count'] > 0].copy()


# Define the individual failure columns
individual_failures_cols = ['FDF (Falha Desgaste Ferramenta)_map',
                           'FDC (Falha Dissipacao Calor)_map',
                           'FP (Falha Potencia)_map',
                           'FTE (Falha Tensao Excessiva)_map',
                           'FA (Falha Aleatoria)_map']

# Identificar o primeiro sim na linha da amostra
def get_failure_type(row):
    for col in individual_failures_cols:
        if row[col] == 'sim':
            return col  # Return the column name as the failure type
    return 'nao_especificada' # Should not happen if sim_count > 0, but as a fallback

df_multiclasse_temp['tipo_falha'] = df_multiclasse_temp.apply(get_failure_type, axis=1)


# Merge
df_multiclasse = df.loc[df_multiclasse_temp.index].copy()

# Add the 'tipo_falha' column from the temporary DataFrame to the final df_multiclasse DataFrame
df_multiclasse['tipo_falha'] = df_multiclasse_temp['tipo_falha']

# Proxima analise
df_multilabel = df_multiclasse.copy()

# Display the first few rows and the value counts of the new target column
print("DataFrame preparado para classificação multiclasse:")
display(df_multiclasse.head())

print("\nContagem dos tipos de falha para classificação multiclasse:")
display(df_multiclasse['tipo_falha'].value_counts())

df_multiclasse.columns

df_multiclasse.shape

colunas = ['tipo', 'temperatura_ar', 'temperatura_processo',
       'umidade_relativa', 'velocidade_rotacional', 'torque',
       'desgaste_da_ferramenta',
       'tipo_falha']

df_multiclasse = df_multiclasse[colunas]

df_multiclasse.head()

df_multiclasse.tipo_falha.value_counts()

"""# Remover a linha referente a falha FA(FalhaAleatoria) pois como a divisao em treino e teste sera estratificada, a ocorrencia unica devera ser removida."""

df_multiclasse = df_multiclasse.loc[df_multiclasse['tipo_falha'] != 'FA (Falha Aleatoria)_map']

df_multiclasse.tipo_falha.value_counts()

# Separate features (X_multi) and target (y_multi) for multiclass classification
X_multi = df_multiclasse.drop('tipo_falha', axis=1).copy()
y_multi = df_multiclasse['tipo_falha'].copy()

# Identify numerical and categorical columns in the multiclass features
numeric_cols_multi = X_multi.select_dtypes(include=np.number).columns.tolist()
categorical_cols_multi = X_multi.columns.difference(numeric_cols_multi).to_list()

print("Features para classificação multiclasse (X_multi):")
display(X_multi.head())
print("\nAlvo para classificação multiclasse (y_multi):")
display(y_multi.head())
print("\nColunas numéricas para multiclasse:", numeric_cols_multi)
print("Colunas categóricas para multiclasse:", categorical_cols_multi)
print("X multiclasse:", X_multi.shape)
print("y multiclasse:", y_multi.shape)

"""# Aplicar labelencoder em tipo_falha"""

le_multi = LabelEncoder()
df_multiclasse['tipo_falha'] = le_multi.fit_transform(df_multiclasse['tipo_falha'])

"""# Dividir em X e y"""

X_multi = df_multiclasse.drop('tipo_falha', axis=1).copy()
y_multi = df_multiclasse['tipo_falha'].copy()

# Split the multiclass data into training and testing sets using the cleaned data
# We are using X_multi_processed_cleaned and y_multi_cleaned
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

print("Dados multiclasse divididos em treino e teste (após pré-processamento com imputação e remoção da classe minoritária):")
print("X_multi_train shape:", X_multi_train.shape)
print("X_multi_test shape:", X_multi_test.shape)
print("y_multi_train shape:", y_multi_train.shape)
print("y_multi_test shape:", y_multi_test.shape)

print("\nDistribuição das classes em y_multi_train:")
display(y_multi_train.value_counts())
print("\nDistribuição das classes em y_multi_test:")
display(y_multi_test.value_counts())

"""# Baseline Multiclasse"""

from sklearn.dummy import DummyClassifier
import mlflow
from mlflow.models import infer_signature
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression # Import LogisticRegression


# Função para logar matriz de confusão
def log_confusion_matrix(y_true, y_pred, nome_modelo, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.tight_layout()
    path = f"report_multiclasse\BASELINE\confusion_matrix_{nome_modelo}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()


# Função auxiliar para logar no MLflow
def avaliar_e_logar_baseline(model, nome_execucao, X_train, y_train, X_test, y_test, labels):
    with mlflow.start_run(run_name=nome_execucao):
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        # Use weighted average for precision, recall, and f1-score in multiclass
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("weighted_precision", precision)
        mlflow.log_metric("weighted_recall", recall)
        mlflow.log_metric("weighted_f1-score", f1)


        # Log classification report as artifact
        report_dict = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report_dict).transpose()
        csv_path = f"report_multiclasse\BASELINE\classification_report_{nome_execucao}.csv"
        df_report.to_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # Log confusion matrix
        log_confusion_matrix(y_test, y_pred, nome_execucao, [str(l) for l in labels])

        print(f"\n### {nome_execucao} ###")
        print(classification_report(y_test, y_pred, target_names=[str(l) for l in labels], zero_division=0))

        mlflow.end_run()

# ================================
# Executar experimentos e salvar no MLflow
# ================================
mlflow.set_experiment("Baseline Multiclasse")

one = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = one.fit_transform(X_multi_train)
X_test_encoded = one.transform(X_multi_test)

labels = ['FDC (Falha Dissipacao Calor)',
       'FDF (Falha Desgaste Ferramenta)', 'FP (Falha Potencia)',
       'FTE (Falha Tensao Excessiva)']

# Baseline 1: sempre prever a classe majoritária
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_encoded, y_multi_train)
avaliar_e_logar_baseline(dummy, "DummyClassifier_Baseline", X_train_encoded, y_multi_train, X_test_encoded, y_multi_test, labels)


# Baseline 2: Logistic Regression sem ajuste (usando dados multiclasse)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_encoded, y_multi_train) # Use multiclass train data
avaliar_e_logar_baseline(model, "LogisticRegression_Baseline", X_train_encoded, y_multi_train, X_test_encoded, y_multi_test, labels) # Use multiclass data and labels

"""# Treinamento e Avaliacao Multiclasse
- KNeighborsClassifier
- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier
- XGBClassifier

# Estrategias
- Sem smote
- Com smote
- class_weight
- sample_weight

"""

# ================================
# Função para logar matriz de confusão
# ================================
def log_confusion_matrix(y_true, y_pred, nome_modelo, estrategia, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    # Use the string labels for display
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {nome_modelo}_{estrategia}")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.tight_layout()
    # Use nome_modelo for the path
    path = f"report_multiclasse\confusion_matrix_{nome_modelo}_{estrategia}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

# Função auxiliar para logar no MLflow
# --------------------------
# Modified for Multiclass Classification - Using encoded target variables for metrics calculation
def avaliar_e_logar(model_pipeline, nome_execucao, X_train, y_train, X_test, y_test, original_label_names, resultados, apply_smote=False):
    with mlflow.start_run(run_name=nome_execucao):
        # Apply preprocessing (X_train and X_test are original dataframes)
        # The pipeline now includes preprocessing
        classifier = model_pipeline.named_steps['classifier']
        print("Model:", classifier.__class__.__name__)

        # Determine training data based on apply_smote flag
        if apply_smote:
            print("Applying SMOTE...")
            # SMOTE for multiclass - use SMOTE or other oversampling methods
            # SMOTE expects encoded labels for multiclass
            smote = SMOTE(random_state=42)
            # Apply SMOTE to the preprocessed training data
            # Ajuste o pré-processador separadamente ANTES de usá-lo para SMOTE
            preprocessor = model_pipeline.named_steps['preprocess']
            X_train_processed = preprocessor.fit_transform(X_train)
            X_fit, y_fit_encoded = smote.fit_resample(X_train_processed, y_train)

            print("SMOTE applied.")
            estrategia = 'Smote'
            # Ajuste o modelo com os dados do SMOTE
            classifier = model_pipeline.named_steps['classifier']
            classifier.fit(X_fit, y_fit_encoded)
            # Agora transforme o X_test
            X_test_processed = preprocessor.transform(X_test)
            y_pred_encoded = classifier.predict(X_test_processed)
            print("Estrategia:", estrategia)
        else:
            print("Sem SMOTE.")
            # Use the preprocessed training data directly
            model_pipeline.fit(X_train, y_train)
            y_pred_encoded = model_pipeline.predict(X_test)
            estrategia = 'sem_smote'
            print("Estrategia:", estrategia)
            X_fit = model_pipeline.named_steps['preprocess'].transform(X_train)
            y_fit_encoded = y_train.copy()

            # Print specific hyperparameters for verification (when not using SMOTE)
            if isinstance(classifier, RandomForestClassifier) and hasattr(classifier, 'class_weight') and classifier.get_params().get('class_weight') == 'balanced':
                print(f"RandomForest class_weight: {classifier.get_params().get('class_weight')}")
                # If using class_weight='balanced', the strategy is class_weight
                estrategia = 'class_weight'
                print("Estrategia:",estrategia)


        # Treina o pipeline
        print("Training...")
        # Fit only the classifier part of the pipeline using the appropriate data (X_fit_processed, y_fit_encoded)
        # Apply sample weights only when not using SMOTE for GradientBoosting
        if classifier.__class__.__name__ == 'GradientBoostingClassifier' and not apply_smote :
            # compute_sample_weight for multiclass using encoded y_fit
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_fit_encoded)
            classifier.fit(X_fit, y_fit_encoded, sample_weight=sample_weights) # Fit with encoded y_fit and sample weights
            print("GradientBoostingClassifier with sample_weights")
            estrategia = 'sample_weights' # Update strategy if sample_weights is used
            print("Estrategia:",estrategia)

        else:
            # Fit all other models, including GB when using SMOTE
            # For LogisticRegression and potentially others, can use class_weight='balanced'
            if isinstance(classifier, LogisticRegression) and not apply_smote:
                classifier.set_params(class_weight='balanced')
                print("LogisticRegression with class_weight='balanced'")
                estrategia = 'class_weight'
                print("Estrategia:",estrategia)

            classifier.fit(X_fit, y_fit_encoded) # Fit with encoded y_fit


        # Make predictions using the fitted pipeline
        # Predict on the preprocessed test data
        X_test_processed = model_pipeline.named_steps['preprocess'].transform(X_test)
        y_pred_encoded = classifier.predict(X_test_processed)


        # Calculate metrics - Use weighted average for multiclass
        # Use the encoded test labels (y_test_encoded) and encoded predictions (y_pred_encoded) for metric calculations
        # Provide the numerical labels (0, 1, 2, 3) to the 'labels' parameter
        numerical_labels = sorted(y_test.unique())

        acc = accuracy_score(y_test, y_pred_encoded)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_encoded, average='weighted', zero_division=0, labels=numerical_labels)


        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("weighted_precision", precision)
        mlflow.log_metric("weighted_recall", recall)
        mlflow.log_metric("weighted_f1-score", f1)
        print("\nMetricas: accuracy, weighted precision, recall e f1-score  salvas no mlflow")

        # Relatório detalhado (by class)
        # Use the original_label_names for target_names in classification_report for readability
        # Ensure y_test_encoded and y_pred_encoded are used as the input to classification_report
        report_dict = classification_report(y_test, y_pred_encoded, target_names=original_label_names, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report_dict).transpose()


        print("\nMatriz de confusao salva no mlflow\n")
        # Use nome_execucao for the CSV path
        csv_path = f"classification_report\multiclasse_{nome_execucao}_{estrategia}.csv"
        df_report.to_csv(csv_path)
        mlflow.log_artifact(csv_path)

        # Matriz de confusão
        # Pass nome_execucao as nome_modelo and use encoded y_test_encoded and y_pred_encoded as input
        # Use original_label_names for the labels parameter in log_confusion_matrix for display
        log_confusion_matrix(y_test, y_pred_encoded, nome_execucao, estrategia, original_label_names)

        # Generate and log precision-recall curve if probabilities are available
        # Skipping binary PR curve for multiclass context


        # Guardar para ranking final (include per-class metrics if desired, but weighted is standard)
        result_entry = {
            "Modelo": nome_execucao,
            "Accuracy": acc,
            # Use the directly calculated weighted metrics for the results dataframe
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            'Estrategia': estrategia.upper()
        }

        # 🚀 AQUI ESTÁ A CHAVE DA MUDANÇA: Itere sobre as classes e adicione ao MLflow E ao dicionário de resultados
        # Use original_label_names to iterate through the classes
        for i, cls_name in enumerate(original_label_names):
            cleaned_cls_name = cls_name
            # Access metrics from the report_dict using the original class name (string)
            if cls_name in report_dict:
                metrics = report_dict[cls_name]
                # Log no MLflow
                mlflow.log_metric(f"precision_{cleaned_cls_name}", metrics['precision'])
                mlflow.log_metric(f"recall_{cleaned_cls_name}", metrics['recall'])
                mlflow.log_metric(f"f1-score_{cleaned_cls_name}", metrics['f1-score'])

                # Adicione ao dicionário de resultados
                result_entry[f"Precision_{cleaned_cls_name}"] = metrics['precision']
                result_entry[f"Recall_{cleaned_cls_name}"] = metrics['recall']
                result_entry[f"F1-score_{cleaned_cls_name}"] = metrics['f1-score']

        # Adicione o dicionário completo à lista de resultados
        resultados.append(result_entry)

        print("Metricas por classe salvas no MLflow.")
        # --- Fim da adição ---


        # Show in console
        # Use original_label_names for target_names in classification_report for display
        print(f"\n### {nome_execucao} ###")

        print(classification_report(y_test, y_pred_encoded, target_names=original_label_names, zero_division=0))

        mlflow.end_run()

# Calcule os pesos da amostra para balanceamento - multiclasse
# 'balanced' ajusta os pesos inversamente proporcionais à frequência das classes
# These sample weights will be used for models that support the sample_weight parameter
# Use the encoded training labels for sample weight calculation
sample_weights_multi = compute_sample_weight(class_weight='balanced', y=y_multi_train)

# Compute the positive class weight - this was for binary, not directly applicable here
# pos_class_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

# ================================
# Lista de modelos a testar - Multiclasse
# ================================
modelos_multi = {
    'KNN': KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000), # Increased max_iter for convergence
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42), # Use balanced class_weight for RF
    "GradientBoosting": GradientBoostingClassifier(), # Will use sample_weight
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), # scale_pos_weight not for multi
}

# Get the labels for the multiclass classification (the unique failure types)
# Use the labels from the LabelEncoder to ensure correct mapping
original_label_names = le_multi.classes_.tolist()
# Sort labels for consistent order in reports and matrices
original_label_names.sort()
labels = ['FDC_Falha_Dissipacao_Calor',
       'FDF_Falha_Desgaste_Ferramenta', 'FP_Falha_Potencia',
       'FTE_Falha_Tensao_Excessiva']


# ================================
# Executar experimentos e salvar no MLflow - Multiclasse
# ================================
mlflow.set_experiment("Classificacao Multiclasse")

resultados_multi = []
# ================================
# Executar experiments and log to MLflow - Multiclasse
# ================================
n=1

# The preprocessor_multi
preprocessor_multi = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]), numeric_cols_multi),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_multi)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

for nome, modelo in modelos_multi.items():
    print("Modelo", n, "de", len(modelos_multi))
    n+=1

    print(f"\n### Executing {nome} ###")

    # Create a pipeline containing the preprocessor and the classifier
    model_pipeline_multi = Pipeline(steps=[
        ('preprocess', preprocessor_multi), # Include the preprocessor
        ('classifier', modelo) # Use the model instance from the dictionary
    ])

    # 1) Without balancing strategy explicitly in model params (some models might have defaults)
    print("\nWithout explicit balancing strategy")
    # Pass the pipeline instance and the preprocessed/split multiclass data (encoded target)
    # Pass the original y_multi_test (string labels) as the new parameter
    # X_multi_train and X_multi_test are the original dataframes BEFORE preprocessing
    avaliar_e_logar(model_pipeline_multi, f"{nome}_sem_balanceamento", X_multi_train, y_multi_train, X_multi_test, y_multi_test,labels, resultados_multi, apply_smote=False)


    # 2) With SMOTE (Skip SMOTE for models where class_weight or sample_weight is preferred or more effective)
    # SMOTE can be applied to multiclass, but implementation needs care for minority classes.
    # Let's apply SMOTE for KNN and LogisticRegression as a strategy
    if nome in ['KNN', 'LogisticRegression']:
        print("With SMOTE")
        # Pass the pipeline instance and the preprocessed/split multiclass data (encoded target)
        # Pass the original y_multi_test (string labels) as the new parameter
        # X_multi_train and X_multi_test are the original dataframes BEFORE preprocessing
        avaliar_e_logar(model_pipeline_multi, f"{nome}_com_SMOTE", X_multi_train, y_multi_train, X_multi_test, y_multi_test, labels, resultados_multi, apply_smote=True)


    print("\n------------------------------\n")

# ================================
# Ranking final em Pandas - Multiclasse
# ================================
df_resultados_multi = pd.DataFrame(resultados_multi)
print("\n🏆 Ranking dos modelos multiclasse:")
print(df_resultados_multi)

df_resultados = pd.DataFrame(resultados).sort_values(by="Recall", ascending=False).reset_index()
print("\n🏆 Ranking dos modelos:")
colunas = ['Modelo', 'Recall','Precision', 'F1-score', 'Estrategia']
print(df_resultados[colunas])



"""# Multilabel

# Foram filtradas do dataset apenas as linhas onde a coluna falha_maquina indica ocorrencia de falha (sim).
"""

df_multilabel.head()

df_multilabel.shape

df_multilabel.columns

colunas = ['tipo', 'temperatura_ar', 'temperatura_processo',
       'umidade_relativa', 'velocidade_rotacional', 'torque',
       'desgaste_da_ferramenta',
       'FDF (Falha Desgaste Ferramenta)_map',
       'FDC (Falha Dissipacao Calor)_map', 'FP (Falha Potencia)_map',
       'FTE (Falha Tensao Excessiva)_map', 'FA (Falha Aleatoria)_map']

df_final =df_multilabel[colunas].copy()

df_final

df_final.rename(columns={
    'FDF (Falha Desgaste Ferramenta)_map': 'FDF_Falha_Desgaste_Ferramenta',
    'FDC (Falha Dissipacao Calor)_map': 'FDC_Falha_Dissipacao_Calor',
    'FP (Falha Potencia)_map': 'FP_Falha_Potencia',
    'FTE (Falha Tensao Excessiva)_map': 'FTE_Falha_Tensao_Excessiva',
    'FA (Falha Aleatoria)_map': 'FA_Falha_Aleatoria'
}, inplace=True)

"""# Identificado um alto desbalanceamento com relação a quantidade de ocorrencia dos 5 tipos de falhas."""

import pandas as pd

# Supondo que seu DataFrame é df
# Lista das colunas de falhas
colunas_falhas = [
    'FDF_Falha_Desgaste_Ferramenta',
    'FDC_Falha_Dissipacao_Calor',
    'FP_Falha_Potencia',
    'FTE_Falha_Tensao_Excessiva',
    'FA_Falha_Aleatoria'
]

for coluna in colunas_falhas:
    df_final[coluna] = df_final[coluna].map({'sim': 1, 'nao': 0})

# 1. Contagem de falhas por linha
df_final['numero_de_falhas'] = df_final[colunas_falhas].sum(axis=1)

# 2. Contagem individual de cada falha
contagem_individual = df_final[colunas_falhas].sum()

# 3. Contagem de quantas linhas têm 0, 1, 2, ... falhas
distribuicao_falhas = df_final['numero_de_falhas'].value_counts().sort_index()

print("Contagem individual de cada falha:")
print(contagem_individual)
print("\nDistribuição do número de falhas por linha:")
print(distribuicao_falhas)

"""# Foi identificado que a ocorrencia de 3 falhas em um equipamento tem apenas 1 amostra.
# O treinamento do modelo irá dividir de forma proporcional as amostras em train e test e isso seria impossível tendo apenas uma amostra.
# Portanto essa amostra será removida, sendo considerada um caso raro.
"""

df_final.loc[df_final.numero_de_falhas ==3]

# Exclua a amostra com 3 falhas da divisão
df_sem_casos_raros = df_final[df_final['numero_de_falhas'] < 3]

print("Dataset sem casos raros:", df_sem_casos_raros.shape)

# Separe as features (X) e os rótulos (y) do novo DataFrame
X = df_sem_casos_raros.drop(columns=['numero_de_falhas'])
y = df_sem_casos_raros[colunas_falhas]

# Use a coluna 'total_falha' para estratificar o restante dos dados
estratificacao = df_sem_casos_raros['numero_de_falhas']

# Faça a divisão estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=estratificacao
)

print("Amostra rara excluída com sucesso. Divisão estratificada realizada.")

"""# Baseline Multilabel
- MultiOutputClassifier
- DummyClaassifier
"""

# Baseline 3: Dummy Classifier do scikit-learn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier

dummy_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(
        DummyClassifier(strategy='stratified', random_state=42)  # ou 'most_frequent'
    ))
])

dummy_pipeline.fit(X_train, y_train)
y_pred_dummy = dummy_pipeline.predict(X_test)

"""#  Treinamento"""

# 1. Defina o pipeline de pré-processamento
# Use SimpleImputer para preencher dados ausentes
# Pipeline para colunas numéricas: imputação e escalonamento
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Pipeline para colunas numéricas: imputação e escalonamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para colunas categóricas: imputação e codificação One-Hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False para evitar matriz esparsa
])


# 2. Defina o pipeline final de modelagem
# Combine o preprocessor com o classificador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Definir os Modelos Finais dentro do Pipeline

# --- Modelo 1: MultiOutputClassifier (Seu original) ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])

# 3. Treine o modelo
model_pipeline.fit(X_train, y_train)

# 4. Faça previsões
y_pred = model_pipeline.predict(X_test)
print("Previsoes:\n",y_pred[:10])

"""# Avaliacao"""

# Lista das colunas de falha

classes = [
    'FDF_Falha_Desgaste_Ferramenta',
    'FDC_Falha_Dissipacao_Calor',
    'FP_Falha_Potencia',
    'FTE_Falha_Tensao_Excessiva',
    'FA_Falha_Aleatoria'
]

# --- Métricas de Avaliação ---

## 1. Métricas de Agregação (Geral)
# Estas métricas fornecem uma visão geral do desempenho do modelo.

# F1-Score: A média harmônica entre precisão e recall.
# 'weighted': Considera o suporte (número de amostras) de cada classe.
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Recall: A proporção de falhas reais que o modelo identificou.
# 'weighted': Média ponderada pela quantidade de amostras em cada classe.
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

# Precisão: A proporção de previsões de falha que estavam corretas.
# 'weighted': Média ponderada pela quantidade de amostras em cada classe.
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)

# Acurácia de Subconjunto Exato:
# É muito rigorosa. Só considera a previsão correta se todos os rótulos forem idênticos ao real.
# Útil para saber quão "perfeito" o modelo é, mas geralmente o valor é baixo.
accuracy_subset = accuracy_score(y_test, y_pred)


print("--- Métricas de Desempenho (Agregadas) ---")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print(f"Recall (Weighted): {recall_weighted:.4f}")
print(f"Precisão (Weighted): {precision_weighted:.4f}")
print(f"Acurácia de Subconjunto Exato: {accuracy_subset:.4f}")

print("\n" + "-"*45 + "\n")

## 2. Métricas por Classe
# Estas métricas são as mais importantes para o seu caso, pois mostram o desempenho
# em cada tipo de falha, especialmente as mais raras.

print("--- Métricas por Classe ---")
for i, col in enumerate(classes):
    # Selecionar as colunas de cada classe individualmente
    y_test_col = y_test.iloc[:, i]
    y_pred_col = y_pred[:, i]

    # Calcular as métricas
    f1 = f1_score(y_test_col, y_pred_col, zero_division=0)
    recall = recall_score(y_test_col, y_pred_col, zero_division=0)
    precision = precision_score(y_test_col, y_pred_col, zero_division=0)

    print(f"\nClasse: {col}")
    print(f"  - F1-Score: {f1:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precisão: {precision:.4f}")

print(classification_report(y_test, y_pred_dummy, target_names=classes, zero_division=0))
report_dict = classification_report(y_test, y_pred_dummy, target_names=classes, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()

# Use nome_execucao for the CSV path
csv_path = f"report_multilabel\BASELINE\MultiOutput_DummyClassifier.csv"
df_report.to_csv(csv_path)

"""# Treinamento final
-  MultiOutputClassifier
- ClassifierChain
- Estrategia: usar RandomForestClassifier
"""

def log_confusion_matrix_simple(y_true, y_pred, nome_modelo, estrategia, target_columns):
    """Versão ultra-simplificada que evita problemas de formato"""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Converter para arrays numpy
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    for i, label_name in enumerate(target_columns):
        try:
            # Extrair coluna individual
            y_true_single = y_true_np[:, i]
            y_pred_single = y_pred_np[:, i]

            cm = confusion_matrix(y_true_single, y_pred_single)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=['Não', 'Sim'],
                       yticklabels=['Não', 'Sim'])
            plt.title(f"Matriz de Confusão - {nome_modelo} {label_name}")
            plt.ylabel("Verdadeiro")
            plt.xlabel("Previsto")
            plt.tight_layout()

            path = f"report_multilabel\confusion_matrix_{nome_modelo}_{estrategia}_{label_name}.png"
            plt.savefig(path)
            mlflow.log_artifact(path)
            plt.close()

        except Exception as e:
            print(f"Erro em {label_name}: {e}")
            continue

def avaliar_e_logar(nome_execucao, estrategia, y_test, y_pred_multi_label, target_columns):
    with mlflow.start_run(run_name=nome_execucao):
      # Relatório detalhado (by class)
        # Use the original_label_names for target_names in classification_report for readability
        # Ensure y_test_encoded and y_pred_encoded are used as the input to classification_report

        log_confusion_matrix_simple(y_test, y_pred_multi_label, nome_execucao, estrategia, target_columns)
        print("\nMatriz de confusao salva no mlflow\n")

        print(f"\n### {nome_execucao} ###")
        #print(classification_report(y_test, y_pred_multi_label, target_names=target_columns, output_dict=True, zero_division=0))
    mlflow.end_run()

# Features e targets
feature_columns = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta', 'tipo'] # Incluindo 'tipo' que é string

target_columns = [
    'FDF_Falha_Desgaste_Ferramenta',
    'FDC_Falha_Dissipacao_Calor',
    'FP_Falha_Potencia',
    'FTE_Falha_Tensao_Excessiva',
    'FA_Falha_Aleatoria'
]

X = df_sem_casos_raros[feature_columns]
y = df_sem_casos_raros[target_columns]

# Dividir os dados
# Use a coluna 'total_falha' para estratificar o restante dos dados
estratificacao = df_sem_casos_raros['numero_de_falhas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=estratificacao)

# 2. Identificar as colunas numéricas e categóricas AUTOMATICAMENTE
# Isso é mais robusto do que listar manualmente
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns


# 3. Criar o Pré-processador (ColumnTransformer)
# Pipeline para colunas numéricas: imputação e escalonamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para colunas categóricas: imputação e codificação One-Hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False para evitar matriz esparsa
])

# Combinar os transformadores numéricos e categóricos
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features),
    ])

# 4. Definir os Modelos Finais dentro do Pipeline

# --- Modelo 1: MultiOutputClassifier (Seu original) ---
multi_output_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1))
])

# --- Modelo 2: ClassifierChain (Nova proposta) ---
# A ordem da cadeia: [FDF, FDC, FP, FTE, FA] -> índices [0, 1, 2, 3, 4]
chain_order = [0, 1, 2, 3, 4]

classifier_chain_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ClassifierChain(RandomForestClassifier(random_state=42),
                                   order=chain_order,
                                   random_state=42))
])


# ================================
# Executar experimentos e salvar no MLflow - Multilabel
# ================================
mlflow.set_experiment("Classificacao Multilabel")

print("=== Treinando e Avaliando MultiOutputClassifier ===")
multi_output_pipeline.fit(X_train, y_train)
y_pred_multi_label = multi_output_pipeline.predict(X_test)
avaliar_e_logar('MultiOutputClassifier', 'RandomForestClassifier', y_test, y_pred_multi_label, target_columns)
report_dict = classification_report(y_test, y_pred_multi_label, target_names=target_columns, output_dict=True, zero_division=0)
csv_path = "classification_report\multilabel_MultiOutputClassifier.csv"
# Save the DataFrame to a CSV file
df_report.to_csv(csv_path, index=True)
print(classification_report(y_test, y_pred_multi_label, target_names=target_columns, zero_division=0))



print("\n" + "="*60 + "\n")
print("=== Treinando e Avaliando ClassifierChain ===")
classifier_chain_pipeline.fit(X_train, y_train)
y_pred_chain = classifier_chain_pipeline.predict(X_test)
avaliar_e_logar('ClassifierChain', 'RandomForestClassifier', y_test, y_pred_chain, target_columns)
report_dict = classification_report(y_test, y_pred_chain, target_names=target_columns, output_dict = True, zero_division=0)
csv_path = "classification_report\multilabel_ClassifierChain.csv"
# Save the DataFrame to a CSV file
df_report.to_csv(csv_path, index=True)
print(classification_report(y_test, y_pred_chain, target_names=target_columns, zero_division=0))

"""# Matriz de Confusao"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Defina o modelo que você quer avaliar
y_pred = y_pred_chain  # Use as previsões do modelo escolhido

# Crie uma figura
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, coluna in enumerate(target_columns):
    # Verificar se há dados para esta label no teste
    if y_test[coluna].sum() == 0 and (y_pred[:, i].sum() == 0):
        # Caso especial: nenhum exemplo positivo no teste e nenhuma previsão positiva
        axes[i].text(0.5, 0.5, f'Nenhum dado de teste\npara {coluna}',
                    ha='center', va='center', fontsize=12)
        axes[i].set_title(f'Matriz de Confusão - {coluna}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        continue

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test[coluna], y_pred[:, i])

    # Garantir que a matriz seja 2x2 (pode ser 1x1 ou 1x2 em casos extremos)
    if cm.shape == (1, 1):
        # Expandir para 2x2 preenchendo com zeros
        cm_full = np.zeros((2, 2), dtype=int)
        cm_full[0, 0] = cm[0, 0]  # TN
    elif cm.shape == (1, 2):
        cm_full = np.zeros((2, 2), dtype=int)
        cm_full[0, :] = cm[0, :]  # TN e FP
    elif cm.shape == (2, 1):
        cm_full = np.zeros((2, 2), dtype=int)
        cm_full[:, 0] = cm[:, 0]  # TN e FN
    else:
        cm_full = cm

    # Calcular percentuais (evitando divisão por zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = np.zeros_like(cm_full, dtype=float)
        row_sums = cm_full.sum(axis=1)
        for j in range(cm_full.shape[0]):
            if row_sums[j] > 0:
                cm_percent[j, :] = cm_full[j, :].astype('float') / row_sums[j] * 100
            else:
                cm_percent[j, :] = 0

    # Plotar heatmap
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'])

    # Adicionar percentuais
    for j in range(cm_full.shape[0]):
        for k in range(cm_full.shape[1]):
            if cm_full[j, k] > 0:  # Só adiciona texto se houver valor
                axes[i].text(k + 0.5, j + 0.3, f'({cm_percent[j, k]:.1f}%)',
                           ha='center', va='center', color='red', fontsize=9)

    axes[i].set_title(f'ClassifierChain\nMatriz de Confusão - {coluna}')
    axes[i].set_xlabel('Previsao')
    axes[i].set_ylabel('Verdadeiro')

# Remover subplots vazios
for j in range(len(target_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('report_multilabel\chain_confusion_matrixes_multilabel.png')
plt.show()

# Analisar cada label individualmente
for i, coluna in enumerate(target_columns):
    print(f"\n=== Análise para {coluna} ===\n")

    try:
        cm = confusion_matrix(y_test[coluna], y_pred[:, i])
        print("Matriz de Confusão:")
        print(cm)

        # Extrair TN, FP, FN, TP (cuidado com matrizes não-2x2)
        if cm.size == 1:
            tn = cm[0, 0]
            fp, fn, tp = 0, 0, 0
        elif cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            print("Matriz com formato inesperado")
            continue

        print(f"\nVerdadeiros Negativos (TN): {tn}")
        print(f"Falsos Positivos (FP): {fp}")
        print(f"Falsos Negativos (FN): {fn}")
        print(f"Verdadeiros Positivos (TP): {tp}")

    except Exception as e:
        print(f"Erro ao calcular matriz para {coluna}: {e}")

# Crie uma figura
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, coluna in enumerate(target_columns):
    try:
        # Tentar criar a matriz de confusão normalmente
        cm = confusion_matrix(y_test[coluna], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não', 'Sim'])
        disp.plot(ax=axes[i], values_format='d', colorbar=False)
        axes[i].set_title(f'Chain\nMatriz de Confusão - {coluna}')
    except Exception as e:
        # Se der erro, mostrar mensagem informativa
        #axes[i].text(0.5, 0.5, f'Erro: Sem dados',
                    #ha='center', va='top', fontsize=10)
        axes[i].set_title(f'{coluna} - Erro na matriz / Sem dados')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

# Remover subplots vazios
for j in range(len(target_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
