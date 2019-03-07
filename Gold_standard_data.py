import pandas as pd
import re

def read_excel_file_to_dataframe(file_path):

    read_df = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
    return read_df


def main():

    file_path = "/Users/tejasvibelsare/Documents/training.xlsx"
    read_df = read_excel_file_to_dataframe(file_path)
    regex = re.compile('[/]')

    for i in read_df.index:

        if read_df['debate'][i] != 'agree':
            read_df.drop(index = i,inplace=True)

        # getting KeyError: 0 while trying to read element with /
        # if (regex.search(read_df['human 1'][i])):
            # if read_df['human 1'][i] != 'alergic' and read_df['human 1'][i] != 'infectious':
            # read_df.drop(index = i, inplace=True)

    read_df.to_csv("/Users/tejasvibelsare/Documents/Gold_Standard_Data.csv",encoding = 'utf-8')




if __name__ == "__main__":
    # calling main function
    main()