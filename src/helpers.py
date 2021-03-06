import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import nltk
import seaborn as sns
import re
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

stop_words = set(stopwords.words('english'))  # Sets English stopwords to avoid in analysis
pd.set_option('display.max_colwidth', None)  # Sets an option in pandas


def plot_pos_neg_in_training(training):
    print("Plotting Positive and Negative tweets and training data...")

    training['label'].value_counts().plot.bar(color='red',
                                              figsize=(6, 4),
                                              title="Positive vs. Negative in Training Data",
                                              xlabel="Pos. or Neg.",
                                              ylabel='Frequency')
    plt.savefig('../output/Positive vs. Negative in Training Data.png')
    plt.clf()

    print("Plotting complete!")


def plot_lengths_pos_neg(training, testing):
    print("Plotting relative lengths of Positive and Negative tweets in training data...")

    training['tweet'].str.len().plot.hist(color='blue', figsize=(6, 4))
    testing['tweet'].str.len().plot.hist(color='green', figsize=(6, 4))

    plt.xlabel("Tweet Length")
    plt.ylabel("Frequency")
    plt.title("Relative Lengths of Positive and Negative Tweets")
    plt.savefig("../output/Tweet Length Analysis.png")
    plt.clf()

    print("Plotting complete!")


def make_word_clouds(training):
    print("Making words clouds for Positive and Negative words in training data...")

    normal_words = ' '.join([text for text in training['tweet'][training['label'] == 0]])
    cleaned_normal_words = clean_data_for_word_clouds(normal_words)

    word_cloud = WordCloud(width=800,
                           height=500,
                           random_state=0,
                           max_font_size=110,
                           background_color='white')
    word_cloud.generate(cleaned_normal_words)

    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.title('Positive Words')
    plt.savefig("../output/Positive Wordcloud.png")
    plt.clf()

    negative_words = ' '.join([text for text in training['tweet'][training['label'] == 1]])
    cleaned_negative_words = clean_data_for_word_clouds(negative_words)
    word_cloud.generate(cleaned_negative_words)

    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.title('Negative Words')
    plt.savefig("../output/Negative Wordcloud.png")
    plt.clf()

    print("Plotting complete!")


def extract_hashtags(x) -> list:
    hashtags = []

    for i in x:
        ht = re.findall(r'#(\w+)', i)
        hashtags.append(ht)

    return hashtags


def hashtag_plot(sentiment_name, sentiments):
    if sentiment_name == "Positive":
        sentiment = sentiments[0]
    else:
        sentiment = sentiments[1]

    tool = nltk.FreqDist(sentiment)
    df = pd.DataFrame({'Hashtag': list(tool.keys()), 'Count': list(tool.values())})

    df = df.nlargest(columns='Count', n=20)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=df, x='Hashtag', y="Count")
    ax.set(ylabel="Count", title=f"Top 20 Most Frequent {sentiment_name} Hashtags")
    plt.savefig(f"../output/Top 20 Most Frequent {sentiment_name} Hashtags.png")
    plt.clf()


def hashtags(training):
    print("Plotting 20 most frequent Positive and Negative hashtags...")

    hashtags_pos = extract_hashtags(training['tweet'][training['label'] == 0])
    hashtags_pos = [item for sublist in hashtags_pos for item in sublist]  # Un-nests the list
    # print(hashtags_pos[:5])

    hashtags_neg = extract_hashtags(training['tweet'][training['label'] == 1])
    hashtags_neg = [item for sublist in hashtags_neg for item in sublist]  # Un-nests the list
    # print(hashtags_neg[:5])

    # Top 20 most frequent Positive hashtags
    hashtag_plot("Positive", [hashtags_pos, hashtags_neg])

    # Top 20 most frequent Negative hashtags
    hashtag_plot("Negative", [hashtags_pos, hashtags_neg])

    print("Plotting complete!")


def clean_data_for_word_clouds(words: str):
    word_array = words.split(' ')
    to_return = ''
    ps = PorterStemmer()

    for word in word_array:
        # If word is not a hashtag or mention, nor is it a stop word, nor is it blank
        if not re.match("^[#@].+", word) and word not in stop_words and word != '':

            word = re.sub("[^a-zA-Z]", '', word)  # Remove special characters
            word = word.lower()
            word = ps.stem(word)  # Find root word of word
            to_return += word + ' '  # Add it to to_return

    return to_return


def clean_train_data(training_row, training):
    print("Cleaning training data...")

    train_corpus = []

    for i in range(0, training_row):
        review = re.sub('[^a-zA-Z]', ' ', training['tweet'][i])
        review = review.lower()
        review = review.split()

        # Performs stemming on words. i.e. finds root word for each word
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)  # Joins words with a space
        train_corpus.append(review)

    print("Cleaning complete!")
    return train_corpus


def clean_test_date(testing_row, testing):
    print("Cleaning testing data...")
    test_corpus = []

    for i in range(0, testing_row):
        review = re.sub('[^a-zA-Z]', ' ', testing['tweet'][0])
        review = review.lower()
        review = review.split()

        # Performs stemming on words. i.e. finds root word for each word
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)  # Joins words with a space
        test_corpus.append(review)

    print("Cleaning complete!")
    return test_corpus


def vectorize_date(train_corpus, test_corpus, training):
    print("Vectorizing data...")

    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(train_corpus).toarray()
    y = training.iloc[:, 1]
    X_test = cv.fit_transform(test_corpus).toarray()

    print("Vectorizing complete!")
    return X, y, X_test


def generate_model(model_name, X_train, y_train, y_valid, X_valid):
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier()

    print(f"Generating {model_name} model...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)  # Make prediction based on validation data

    print(f"{model_name} training accuracy: ", model.score(X_train, y_train))
    print(f"{model_name} validation accuracy: ", model.score(X_valid, y_valid))
    print(f"Generating {model_name} complete!")

    # Confusion matrix
    plot_matrix(model_name, y_valid, y_pred)


def plot_matrix(model_name, y_valid, y_pred):
    print(f"Plotting {model_name} confusion matrix...")

    cm = confusion_matrix(y_valid, y_pred)
    cmfig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f'../output/{model_name} Confusion Matrix.png')
    plt.clf()

    print("Plotting complete!")


def read_data():
    print("Reading data...")

    training = pd.read_csv('../input/train_E6oV3lV.csv')
    testing = pd.read_csv('../input/test_tweets_anuFYb8.csv')

    print("Reading complete!")
    return training, testing


def read_sentiments(training):
    print("Reading sentiments...")

    negative_sentiments = training[training["label"] == 0]
    positive_sentiments = training[training['label'] == 1]

    print("Reading complete!")
    return positive_sentiments, negative_sentiments
