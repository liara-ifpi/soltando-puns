import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.corpus import stopwords


# nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

def load_data(file_path):
    return pd.read_json(file_path, lines=True)
    

def create_x_y(df):
    x = df['text']
    y = df['label']
    return x, y


df_train = load_data('dataset/train.jsonl')
x_train, y_train = create_x_y(df_train)

df_val = load_data('dataset/validation.jsonl')
x_val, y_val = create_x_y(df_val)

df_test = load_data('dataset/test.jsonl')
x_test, y_test = create_x_y(df_test)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words)
x_train_vectorized = vectorizer.fit_transform(x_train)
x_val_vectorized = vectorizer.transform(x_val)
x_test_vectorized = vectorizer.transform(x_test)

rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=15, random_state=40)
lr_model = LogisticRegression(random_state=40, max_iter=100)
svm_model = SVC(probability=True, random_state=40)

voting_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('lr', lr_model),
    ('svm', svm_model)
], voting='soft', n_jobs=30)

voting_model.fit(x_train_vectorized, y_train)

y_test_pred = voting_model.predict(x_test_vectorized)

print(classification_report(y_test, y_test_pred))
