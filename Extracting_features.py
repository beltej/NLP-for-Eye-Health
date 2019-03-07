import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from textblob import TextBlob, Word
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import word_tokenize


def read_excel_file_to_dataframe(file_path):

    read_df = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
    return read_df


def remove_usernames(tweet):

    usernames_removed_tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    return usernames_removed_tweet


def remove_url(tweet):

    url_removed_tweets = re.sub('https?://[A-Za-z0-9./]+', '', tweet)
    return url_removed_tweets


def remove_dots(tweet):
    dots = re.compile(r'\.{3,}')
    removed_dots_word = dots.sub('', tweet)
    return removed_dots_word

# def remove_punctuation(tweet):
#     punctuations = "?:!.,;#=()/"
#     punctuation_removed_tweets = re.sub("?:!.,;#=()/", '', tweet)
#     return url_removed_tweets

def write_to_excel(read_df,file_path):
    read_df.to_excel(file_path)
    del [read_df]

def calculate_polarity(read_df, i):
    tweet = read_df['Contents'][i]

    tweet = TextBlob(tweet)
    pol = tweet.sentiment.polarity
    return pol

def get_matching_keywords(tweet,stemmed_keywords_allergic,stemmed_keywords_infectious):

    ps = PorterStemmer()
    count_of_matching_keywords_allergic = 0
    count_of_matching_keywords_infectious = 0

    for word in word_tokenize(tweet):
        # textblob_word = TextBlob(word)
        # spell_checked_word = textblob_word.correct()   #correcting spelling of word

        lower_case_word = word.lower()  # convert to lower case

        stemmed_word = ps.stem(lower_case_word)

        # sum(stemmed_word == word for word in stemmed_keywords)
        count_of_matching_keywords_allergic += sum(stemmed_word == word for word in stemmed_keywords_allergic)
        count_of_matching_keywords_infectious += sum(stemmed_word == word for word in stemmed_keywords_infectious)

    return count_of_matching_keywords_allergic,count_of_matching_keywords_infectious



def conversion_to_numbers(read_df):
    print("Column headings:")
    print(read_df.columns)
    print(len(read_df))             # total number of records
    print(len(read_df[read_df.debate=='agree']))  # number of records where both human annotators agree

    for i in read_df.index:
        # print(read_df['debate'][i])

        if read_df['debate'][i]== 'agree':

            if read_df['human 1'][i]== 'allergic':
                read_df['category'][i] = 0
            else:
                read_df['category'][i] = 1

        else:
            read_df.drop(i)
            continue


    return read_df

def get_tweet_length(tweet):
    return len(tweet)


def main():

    file_path = "/Users/tejasvibelsare/Documents/training.xlsx"
    read_df = read_excel_file_to_dataframe(file_path)

    print(read_df.columns)
    # tweet=read_tweets(read_df)

    ps = PorterStemmer()

    keywords_allergic = ['allergies']
    stemmed_keywords_allergic = []

    for keyword in keywords_allergic:
        stemmed_keywords_allergic.append(ps.stem(keyword))

    keywords_infectious = []
    stemmed_keywords_infectious = ['infectious','pink','pinkeye']

    for keyword in keywords_infectious:
        stemmed_keywords_infectious.append(ps.stem(keyword))


    for i in read_df.index:

        tweet = read_df['Contents'][i]

        tweet = remove_usernames(tweet)

        tweet= remove_url(tweet)

        tweet = remove_dots(tweet)


        read_df['input_for_count_vect'][i] = tweet

        pol= calculate_polarity(read_df,i)
        read_df['polarity'][i]= pol

        count_of_matching_keywords_allergic,count_of_matching_keywords_infectious = get_matching_keywords(tweet,stemmed_keywords_allergic,stemmed_keywords_infectious)
        read_df['matching_number_of_keywords_allergic'][i] = count_of_matching_keywords_allergic
        read_df['matching_number_of_keywords_infectious'][i] = count_of_matching_keywords_infectious

        tweet_length = get_tweet_length(read_df['Contents'][i])
        read_df['length_of_tweet'][i]=tweet_length


    print(read_df['polarity'])

    # write_to_excel(read_df,file_path)

    read_df = conversion_to_numbers(read_df)

    write_to_excel(read_df, file_path)



if __name__ == "__main__":
    # calling main function
    main()
