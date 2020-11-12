# Add code here...
import pandas as pd

def main():
    path = '/home/duc-hoang/Documents/Junior-Year/Py4DS/LAB/Lab 04/dataset/spam_original.csv'
    df = pd.read_csv(path)

    # show original dataset
    print('>> original df: ')
    print(df.head())
    print(f'>> data shape: {df.shape}')
    
    # show df info
    print('>> Data info: ')
    print(df.info())

    # remove space char in column names
    df.columns = df.columns.str.replace(' ', '')

    # Missing values percentage
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_data = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    print('missing values percentage: ')
    print(missing_value_data)
    
    
if __name__ == "__main__":
    main()