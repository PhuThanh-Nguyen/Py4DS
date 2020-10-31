l(X_train_standardScaler,X_test_standardScaler,y_train,y_test)
	print(f'\nMean Squared Error using Normalizer: {MSE}')
	print(f'R2 Score using Normalizer: {R2}')
	print(f'Mean Absolute Error using Normalizer: {MAE}')
	print(f'Mean Squareroot Error using Normalizer: {MSRE}')
if __name__ == '__main__':
	main()
