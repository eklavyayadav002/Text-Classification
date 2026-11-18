from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Educational text-classification demo using two newsgroup categories,
# not a real spam dataset or production spam filter.
data = fetch_20newsgroups(subset="all",
    categories=["rec.sport.baseball", "talk.politics.misc"],
    remove=("headers", "footers", "quotes"), random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("nb", MultinomialNB())
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Text Classification Demo")
print("Classes:", data.target_names)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=data.target_names))
