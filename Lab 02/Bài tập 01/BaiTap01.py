# Add code here
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    df = pd.read_csv("/home/duc-hoang/Documents/Junior-Year/Py4DS/Py4DS/Lab 02/dataset/xAPI-Edu-Data.csv")
    df.rename(index=str, columns={'gender':'Gender',
                                'NationalITy': 'Nationality',
                                'raisedhands': 'RaisedHands',
                                'VisITedResources': 'VisitedResources'},
                                inplace=True)

    for i in range(1,17):
        print(df.iloc[:,i].value_counts())
        print("*"*20)
    """==========================================================================="""
    sns.pairplot(df, hue="Class")
    plt.show()
    """
    Kiểm tra độ tương quan giữa các features, thông qua mật độ phân bố của các data point. 
    Nếu các lớp dữ liệu phân bố trùng lặp lên nhau thì cho thấy độ tương quan giữa 2 features là cao, ta có thể thực hiện feature selection.
    """

    """==========================================================================="""
    # extract features by visualizing these coefficient of correlations with each other
    plt.figure(figsize=(14,12))
    sns.heatmap(df.corr(), linewidths=.1, cmap="YlGnBu", annot=True)
    plt.yticks(rotation=0)
    plt.show()
    """
    Các features có các hệ số tương quan với nhau, nếu cặp features có hệ số cao (> 0.8), ta có thể thực hiện việc bỏ 1 trong 2 feature.
    Các features có hệ số tương quan thấp, có thể sử dụng cùng với nhau.
    """

    """==========================================================================="""
    # Exploring 'Class' label
    P_Static = sns.countplot(x="Class", data=df, linewidth=2, edgecolor=sns.color_palette("dark"))
    plt.show()
    """
    So sánh các lớp trong label. Nếu có sự chênh lệch quá cao giữa các lớp, chứng tỏ data bị mất cân bằng (Imbalance).
    Ngược lại đồ thị này cho thấy sự chênh lệch giữa các lớp không quá cao, nên data có thể được xem là cân bằng (balance)
    """

    # counting each type in class 'label'
    df.Class.value_counts(normalize=True).plot(kind="bar")
    plt.show()
    """
    Thay vì so sánh các lớp dựa trên số lượng, ta có thể so sánh theo tỷ lệ phần trăm
    """
    """==========================================================================="""

    Raised_hand = sns.boxplot(x="Class", y="RaisedHands", data=df,order=['H','M','L'])
    plt.show()
    """
    Box-plot cho thấy số "RaisedHands" của từng lớp phân bố nhiều nhất tại mỗi khoảng xác định.
    Gồm các đại lượng thống kê như phân vị mức 25%, 50%, 75%, outlier. 
    Qua box-plot có thể đánh giá tổng quan như sau: 
    - Lớp H (High): có số lần "RaisedHands" tập trung trong khoảng 60 < RaisedHands < 85 --> Học sinh giỏi có số lần phát biểu nhiều
    - Lớp M (middle): có số lần "RaisedHands" tập trung trong khoảng 25 < RaisedHands < 75 --> Học sinh trung bình có số lần phát biểu ở mức vừa
    - Lớp L (low): có số lần "RaisedHands" tập trung trong khoảng 10 < RaisedHands < 23 --> Học sinh yếu có số lần phát biểu ở mức thấp
    """
    """==========================================================================="""

    Facegrid = sns.FacetGrid(df, hue='Class', size=6) # Initalize and grid
    Facegrid.map(sns.kdeplot, 'RaisedHands', shade=True) # Apply a plotting function to
    Facegrid.set(xlim=(0, df['RaisedHands'].max())) # Set attributes on each subplot
    Facegrid.add_legend()
    plt.show()

    sns.countplot(x='ParentschoolSatisfaction', data = df, hue = 'Class', palette = 'bright')
    plt.show()

    labels = df.ParentschoolSatisfaction.value_counts()
    colors = ["blue", "green"]
    explode = [0, 0]
    sizes = df.ParentschoolSatisfaction.value_counts().values

    plt.figure(figsize = (7, 7))
    plt.pie(sizes, explode = explode,labels = labels.index, colors = colors, autopct = '%1.1f%%')
    plt.title("Parent school Satisfaction in Data", fontsize = 15)
    plt.legend()
    plt.show()

if __name__ == '__main__':
	main()
