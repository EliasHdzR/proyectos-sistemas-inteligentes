import matplotlib.pyplot as plt
import pandas as pd

sales_data = {
    'Week 1': pd.Series([5000, 5900, 6500, 3500, 4000, 5300, 7900]),
    'Week 2': pd.Series([4000, 3000, 5000, 5500, 3000, 4300, 5900]),
    'Week 3': pd.Series([4000, 5800, 300, 2500, 3000, 5300, 6000]),
    'Day': pd.Series(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
}

sales_df = pd.DataFrame(sales_data)
print(sales_df)

sales_df.plot(kind='line', color=['red', 'blue', 'brown'], x='Day', y=['Week 1', 'Week 2', 'Week 3'])
plt.title('Mela Sales Report')
plt.xlabel('Days')
plt.ylabel('Sales in Rs')
plt.show()

sales_data2 = {
    'Week 1': pd.Series([5000, 5900, 6500, 3500, 4000, 5300, 7900]),
    'Week 2': pd.Series([4000, 3000, 5000, 5500, 3000, 4300, 5900]),
    'Day': pd.Series(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
}

sales_df_2 = pd.DataFrame(sales_data2)
sales_df_2.plot(kind='bar', color=['red', 'blue'], x='Day', y=['Week 1', 'Week 2'])
plt.title('Mela Sales Report 2')
plt.xlabel('Days')
plt.ylabel('Sales in Rs')
plt.show()