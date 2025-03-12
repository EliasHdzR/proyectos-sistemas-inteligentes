import matplotlib.pyplot as plt

#list storing date in string format
date=["25/12","26/12","27/12","28/12"]

#list storing temperature values
temp=[8.5,10.5,6.8,15.5]

#create a figure plotting temp versus date
plt.plot(date, temp)

plt.xlabel("Date") #add the Label on x-axis
plt.ylabel("Temperature") #add the Label on y-axis
plt.title("Date wise Temperature") #add the title to the chart
plt.grid(True) #add gridlines to the background
plt.yticks(temp)

#show the figure
plt.show()

#save the figure
#plt.savefig("graficas/x.png")