import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def main():
	data = pd.read_csv('../dataset/xAPI-Edu-Data.csv')
	object_feature = 'gender'
	int_feature = 'Discussion'
	# Nhận xét biểu đồ: Nhận thấy số lượng dữ liệu 'M' chiếm hơn 60% nên
	# dataset có xu hướng chệch nhiều sang 'M' (giới tính nam) hơn là 'F' (giới tính nữ)
	ax = data[object_feature].value_counts(normalize = True).plot(kind = 'bar')
	plt.show()
	# Nhận xét biểu đồ:
	# +) Phân vị thứ 3 của dữ liệu 'M' khoảng 60%, còn phân vị dữ liệu 'F' hơn 70%
	# --> Đa phần các nữ sinh tham gia bàn luận nhiều hơn so với các nam sinh
	# +) Boxplot của giới tính Nam có phân vị thứ 2 gần với phân vị thứ nhất -> Dữ liệu của nam có thể lệch phải
	# +) Boxplot của giới tính Nữ có phân vị thứ 2 gần với phân vị thứ nhất, nhưng không quá nhiều -> Dữ liệu của nữ  khá đều
	sns.boxplot(x = object_feature, y = int_feature, data = data)
	plt.show()
	# Nhạn xét biểu đồ:
	# +) Biều đồ phân phối của nam lệch phải, đa đỉnh
	# +) Biều đồ phân phối của nữ  lệch phải nhưng không nhiều, đa đỉnh
	FacetGrid = sns.FacetGrid(data, hue = object_feature, height = 6)
	FacetGrid.map(sns.kdeplot, int_feature, shade = True)
	FacetGrid.set(xlim = (0, data[int_feature].max()))
	FacetGrid.add_legend()
	plt.show()

if __name__ == '__main__':
	main()
