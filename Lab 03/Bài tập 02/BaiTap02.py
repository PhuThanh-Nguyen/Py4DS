import pandas as pd

def main():
	data = pd.read_csv('../dataset/AB_NYC_2019.csv')
	print('>> Before drop missing values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	data = data.dropna()
	print('>> After drop missing values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	print('>> Before drop duplicate values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
	print('>> After drop duplicate values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
if __name__ == '__main__':
	main()
