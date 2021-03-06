import warnings
from sklearn.model_selection import train_test_split
import helpers

warnings.filterwarnings('ignore', category=DeprecationWarning)  # Sets to ignore deprecation warnings


def main():
    training, testing = helpers.read_data()
    training_row = training.shape[0]
    testing_row = testing.shape[0]

    # print(training.shape)

    # Extracts all positive and negative tweets from the training data
    positive_sentiments, negative_sentiments = helpers.read_sentiments(training)

    # Plot the relative numbers of positive and negative sentiment in training data
    helpers.plot_pos_neg_in_training(training)

    # Plot relative lengths of Positive and Negative tweets
    helpers.plot_lengths_pos_neg(training, testing)

    # Hashtags
    helpers.hashtags(training)

    # Clean entire training set
    train_corpus = helpers.clean_train_data(training_row, training)

    # Word clouds
    helpers.make_word_clouds(training)

    # Clean entire testing set
    test_corpus = helpers.clean_test_date(testing_row, testing)

    # Vectorize data
    X, y, X_test = helpers.vectorize_date(train_corpus, test_corpus, training)

    # Split training into training amd validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

    # Decision Tree
    helpers.generate_model("Decision Tree", X_train, y_train, y_valid, X_valid)

    # Random Forest
    helpers.generate_model("Random Forest", X_train, y_train, y_valid, X_valid)


if __name__ == "__main__":
    main()
