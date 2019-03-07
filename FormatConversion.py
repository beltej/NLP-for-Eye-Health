import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from textblob import *
import emoji

def read_excel_file_to_dataframe(file_path):

    read_df = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
    return read_df


def remove_usernames(tweet):

    usernames_removed_tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    return usernames_removed_tweet


def remove_url(tweet):

    url_removed_tweets = re.sub('https?://[A-Za-z0-9./]+', '', tweet)
    return url_removed_tweets


def tokenize_tweet(tweet):

    word_tokens = word_tokenize(tweet)
    return word_tokens


def remove_dots(tweet):
    dots = re.compile(r'\.{3,}')
    removed_dots_word = dots.sub('', tweet)
    return removed_dots_word


def processing_words(word_tokens):

    ps = PorterStemmer()
    tokenized_tweet = []
    stop_words = set(stopwords.words('english'))
    punctuations = "?:!.,;#=()/"

    for word in word_tokens:
        if word not in stop_words and word not in punctuations:

            textblob_word = TextBlob(word)
            spell_checked_word = textblob_word.correct()        # spell check

            lower_case_word = spell_checked_word.lower()        # convert to lower case

            stemmed_word = ps.stem(lower_case_word)             # perform word stemming

            tokenized_tweet.append(stemmed_word)                # append processed word

    return tokenized_tweet


def write_to_excel(read_df,file_path):
    read_df.to_excel(file_path)
    del [read_df]

# print("Column headings:")
# print(read_df.columns)
# read_df.drop([,'GUID','Brightview'],axis=1,inplace=True)


# for i in read_df.index:
    # read_df['aftr_stopword_removal'][i]= read_df['Contents'][i]
    # print i
    # print(read_df['Contents'][i])
    # row_df = pd.DataFrame([[read_df['Contents'][i]],read_df['human 1'][i], read_df['human 2'][i]], columns = ['Contents', 'human 1', 'human 2'])
    # row_df = pd.DataFrame([read_df[headers][i]])
    # print(row_df['Contents'])
    # tweet = read_df['Contents'][i]
    # removed_usernames_tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # removing @ and username mentioned
    # removed_url_tweets = re.sub('https?://[A-Za-z0-9./]+', '', removed_usernames_tweet)
    # word_tokens = word_tokenize(removed_url_tweets)
    # tokenized_tweet = []
    # tokenized_tweet = [word for word in word_tokens if not word in stop_words]
    # for word in word_tokens:
    #     if word not in stop_words and word not in punctuations:
    #         textblob_word = TextBlob(word)
    #         spell_checked_word = textblob_word.correct()    #checking spelling of the word
    #
    #         lower_case_word = spell_checked_word.lower()
            #removed_emoji = (emoji_pattern.sub(r'', word))  # no emoji
            # emoji.get_emoji_regexp().sub(r'', word.encode('utf8'))
            # re.sub(r'@[A-Za-z0-9]+', '', word)   # removing @ and username mentioned
            # stemmed_word = ps.stem(lower_case_word)    # for word stemming
            # tokenized_tweet.append(stemmed_word)
    # read_df['Preprocessed_contents'][i]= tokenized_tweet
    # print(tokenized_tweet)
    # write_df.columns
    # row_df = pd.DataFrame(read_df['Contents'][i], read_df['human 1'][i], read_df['human 2'][i])
    # write_df.append(row_df)
    # print(regexp_tokenize(tweet, pattern='\w+|\$[\d\.]+|\S+'))


# read_df.to_excel(file_path)
# del[read_df]

# contents_H1_H2 =
# writer = pd.ExcelWriter(file_path )
# write_df.to_excel()

def main():

    file_path = "/Users/tejasvibelsare/Documents/training.xlsx"
    read_df = read_excel_file_to_dataframe(file_path)

    # tweet=read_tweets(read_df)

    for i in read_df.index:

        tweet = read_df['Contents'][i]

        tweet = remove_usernames(tweet)

        tweet= remove_url(tweet)

        tweet = remove_dots(tweet)

        word_tokens = tokenize_tweet(tweet)

        tokenized_tweet = processing_words(word_tokens)

        read_df['Preprocessed_contents'][i] = tokenized_tweet

    write_to_excel(read_df,file_path)


if __name__ == "__main__":
    # calling main function
    main()