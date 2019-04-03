import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

essays = []
native_languages = []
with open('essays/essays/essays/data/text/index.csv', mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	for row in csv_reader:
		native_languages.append(row["Language"])
		with open(f'preprocessed_with_stopwords/{row["Filename"]}', mode='r') as essay:
			essays.append(essay.read())

matrix = CountVectorizer(max_features=5000, ngram_range=(1, 3), binary=True)
x = matrix.fit_transform(essays).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, native_languages)

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy = " + str(accuracy))