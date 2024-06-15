import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Carregar os dados
data = pd.read_csv('text_data.csv')

# Verificar a distribuição dos dados
print(data['category'].value_counts())

# Separar os dados em recursos (X) e rótulos (y)
X = data['text']
y = data['category']

# Dividir os dados em conjuntos de treinamento e teste, garantindo representatividade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Converter textos em vetores de contagem de palavras
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Treinar o modelo
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Fazer previsões
y_pred = clf.predict(X_test_counts)

# Calcular a precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Novo texto para classificar
new_text = ["Eu estou muito feliz hoje"]
new_text_counts = vectorizer.transform(new_text)

# Fazer a previsão
prediction = clf.predict(new_text_counts)
print(f'Categoria prevista: {prediction[0]}')
