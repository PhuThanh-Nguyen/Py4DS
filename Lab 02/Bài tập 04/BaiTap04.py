import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def main():
	data = pd.read_csv('../dataset/HappinessReport2020.csv')
	"""==========================================================================="""
	'''
	Nhận xét biểu đồ:
	+) Số lượng dữ liệu cho khu vực Châu Phi hạ Sahara, trong khi các khu vực ở các khu vực khác khá ít (Khu vực Mỹ Latinh và Caribean)
	 đến rất ít (khu vực Bắc Mỹ) -> Dữ liệu bị chệch nhiều sang khu vực Sahara
	'''
	ax = data['Regional indicator'].value_counts(normalize = True).plot(kind = 'bar')
	plt.show()
	"""==========================================================================="""
	'''
	Nhận xét biểu đồ:
	+) Biểu đồ thanh khá đồng đều, tuổi thọ trung bình cao nhất trong khu vực Tây Âu (72.9), thấp nhất trong khu vực Châu Phi (55.5) tuổi
	'''
	region = data.groupby('Regional indicator')[['Healthy life expectancy']].agg('mean').reset_index()
	ax = region.plot(x = 'Regional indicator', y = 'Healthy life expectancy', kind = 'bar')
	plt.show()
	"""==========================================================================="""
	'''
	Nhận xét biểu đồ:
	+) Các khu vực Tây Âu, khu vực nhóm gồm Bắc Mỹ, New Zealand và Australia có tuổi thọ trung bình cao, không phân tán quá nhiều. Khu 		vực Tây Âu lệch phải trong khi nhóm Bắc Mỹ, New Zealand và Australia lệch trái
	+) Khu vực Đông Á có tuổi thọ trung bình cao, tuy nhiên phân tán mạnh
	+) Khu vực Sahara hạ Châu Phi có tuổi trung bình thấp, không bị lệch nhiều
	'''
	ax = sns.boxplot(x = 'Regional indicator', y = 'Healthy life expectancy', data = data)
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
	plt.show()
	"""==========================================================================="""
	print(data[['Healthy life expectancy', 'Social support']].corr())
	'''
	Nhận xét: Hệ số tương quan của tình trạng sức khỏe và sự hỗ trợ từ gia đình là khoảng 0.75, thể hiện sự phụ thuộc tuyến tính tăng khá 		mạnh giữa hai đặc trưng này -> Gia đình càng gắn bó thì sức khỏe của thành viên cũng tốt hơn
	Biểu đồ dưới thể hiện mối quan hệ này
	'''
	sns.regplot(x = 'Healthy life expectancy', y = 'Social support', data = data)
	plt.show()
	"""==========================================================================="""
	print(data[['Logged GDP per capita', 'Healthy life expectancy']].corr())
	'''
	Nhận xét: Hệ số tương quan của tình trạng sức khỏe và GDP của quốc gia là khoảng 0.85, thể hiện sự phụ thuộc tuyến tính tăng khá 		mạnh giữa hai đặc trưng này -> Quốc gia càng phát triển về kinh tế thì người dân được chăm sóc sức khỏe tốt hơn
	Biểu đồ dưới thể hiện mối quan hệ này
	'''
	sns.regplot(x = 'Logged GDP per capita', y = 'Healthy life expectancy', data = data)
	plt.show()
	"""==========================================================================="""

if __name__ == '__main__':
	main()
