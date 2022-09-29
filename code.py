from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

with open('training.json',encoding='utf-8') as file:
    lines = file.readlines()
lines = iter(lines)

N = int(next(lines))

X_train,classes = [],[]
for _ in range(N):
    line = next(lines)
    d = eval(line)
    X_train.append(d['question'])
    classes.append(d['topic'].strip())

classes_to_ix = {c:i for i,c in enumerate(set(classes))}
ix_to_classes = {i:c for c,i in classes_to_ix.items()}
y_train = [classes_to_ix[c] for c in classes]

vectorizer = CountVectorizer(max_df=0.5,stop_words="english")
X_train = vectorizer.fit_transform(X_train)

naive = MultinomialNB(alpha=0.3)
naive.fit(X_train, y_train)

P = int(input())
X_test = []
for _ in range(P):
    line = input()
    d = eval(line)
    X_test.append(d['question'])
    
X_test = vectorizer.transform(X_test)
prediction = [ix_to_classes[i] for i in naive.predict(X_test)]
for i in prediction:
    print(i)
