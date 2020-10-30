import pandas as pd

def main():
	# Read data
	data = pd.read_csv('../dataset/AB_NYC_2019.csv')
	
	# Remove missing values, duplicates and outliers
	
	# Remove missing values:
	print('>> Before drop missing values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	data = data.dropna()
	print('>> After drop missing values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	# Remove duplicate values
	print('>> Before drop duplicate values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
	print('>> After drop duplicate values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	# Remove outliers
	Q1 = data.quantile(0.25)
	Q3 = data.quantile(0.75)
	IQR = Q3 - Q1
	
	print('>> Before drop outliers: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())

	data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
	
	print('>> After drop outliers: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
if __name__ == '__main__':
	main()
