# Add code here
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def main():
    data = pd.read_csv('/home/duc-hoang/Documents/Junior-Year/Py4DS/Py4DS/Lab 02/dataset/creditcard.csv')

    data.Class.value_counts()
    #================================================================================
    """
        Tỉ lệ phần trăm của "fraud transaction" là 0.172%
    """
    #================================================================================
    P_Static = sns.countplot(x="Class", data=data, linewidth=2, edgecolor=sns.color_palette("dark"))
    plt.show()
    #================================================================================
    """
    Nhìn vào đồ thị ta thấy được sự dữ liệu có xu hướng "unbalance" giữa các class.
    """
    #================================================================================
    avg_amount = data.groupby("Class")[['Amount']].agg("mean").reset_index()
    sns.barplot(x='Class', y= "Amount", data = avg_amount)
    plt.show()
    #================================================================================
    """
        Dễ thấy được các "fraud transactions" thường có lượng giao dịch có giá trị "đột biến" cao. 
        => Trung bình của các "fraud transactions" cao hơn giao dịch bình thường.
    """
    #================================================================================

    Amount = sns.boxplot(x="Class", y="Amount", data=data)
    Amount.set(ylim=(data['Amount'].min(),300))
    plt.show()
    """
    - Dựa vào box-plot ta thấy rằng các "fraud transactions" gồm 25% có lượng giao dịch ở mức 2$, 75% giao dịch ở mức 120$. 
    - Hơn nữa độ biến động của các "fraud transactions" là cao hơn so với giao dịch thông thường.
    - Median của lớp 1 lệch với Q3 khá nhiều -> Dữ liệu bị lệch phải. 
    """
    Facegrid = sns.FacetGrid(data, hue='Class', size=6) # Initalize and grid
    Facegrid.map(sns.kdeplot, 'Time', shade=True) # Apply a plotting function to
    Facegrid.add_legend()
    plt.show()
    """
    Tại thời điểm khoảng 100000 giây (tương ứng xấp xỉ 27 giờ kể từ khi khảo sát) ta thấy rằng "fraud transactions" tăng đột biến so với "normal transactions"
    => Giả sử mốc khảo sát là 0h thì ta có thể suy đoán "fraud transactions" xảy ra đột biến vào lúc 3h sáng ngày thứ 2.
    Tại thời điểm 130000 giây (tương ứng xấp xỉ 36 giờ kể từ khi khảo sát) ta thấy "fraud transactions" giảm đi, và "normal transactions" tăng lên lại.
    => Có thể suy đoán khoảng 12h trưa ngày thứ 2 "normal transactions" tăng đỉnh điểm trở lại.
""" 
if __name__ == "__main__":
    main()
