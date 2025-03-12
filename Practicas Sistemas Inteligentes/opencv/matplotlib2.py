import matplotlib.pyplot as plt
import pandas as pd

height=[121.9,124.5,129.5,134.6,139.7,147.3,152.4,157.5,162.6]
weight=[19.7,21.3,23.5,25.9,28.5,32.1,35.7,39.6,43.2]

df=pd.DataFrame({"height":height,"weight":weight})

#Set xlabel for the plot
plt.xlabel('Weight in kg')
#Set ylabel for the plot
plt.ylabel('Height in cm')
#Set chart title:
plt.title('Average weight with respect to average height')

#plot using marker '-*' and line colour as green
plt.plot(df.weight,df.height,marker='8',markersize=10,color='green',linewidth=2, linestyle='dashdot')
plt.show()