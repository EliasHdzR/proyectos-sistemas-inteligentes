import pandas as pd
import numpy as np

dFrameEmt = pd.DataFrame()
print(dFrameEmt)
print(dFrameEmt.empty,"\n")

#array1 = np.array([[10,20,30],[40,50,60],[70,80,90]])
array1 = np.array([10,20,30])
array2 = np.array([1,2,3])
array3 = np.array([1,2,3,4])

dFrame = pd.DataFrame([array1,array2,array3], columns=["A","B","C","D"])
print(dFrame)
print(dFrame.empty,"\n")

listDict = [{'a':10, 'b':20}, {'a':5,'b':10, 'c':20}]
dFrame2 = pd.DataFrame(listDict)
print(dFrame2,"\n")

dictForest = {'State': ['Assam', 'Delhi','Kerala'],'GArea': [78438, 1483, 38852] ,'VDF' : [2797, 6.72,1663]}
dFrame3 = pd.DataFrame(dictForest)
print(dFrame3,"\n")

dFrameForest1 = pd.DataFrame(dictForest,columns = ['State','VDF', 'GArea'])
print(dFrameForest1,"\n")

seriesA = pd.Series([1,2,3,4,5], index = ['a', 'b', 'c', 'd', 'e'])
seriesB = pd.Series ([1000,2000,-1000,-5000,1000], index = ['a', 'b', 'c', 'd', 'e'])
seriesC = pd.Series([10,20,-10,-50,100], index = ['z', 'y', 'a', 'c', 'e'])

dFrame4 = pd.DataFrame([seriesA, seriesB, seriesC])
dFrame4 = dFrame4.fillna(0)
print(dFrame4,"\n")

ResultSheet= {
    'Arnab': pd.Series([90, 91, 97], index=['Maths','Science','Hindi']),
    'Ramit': pd.Series([92, 81, 96], index=['Maths','Science','Hindi']),
    'Samridhi': pd.Series([89, 91, 88], index=['Maths','Science','Hindi']),
    'Riya': pd.Series([81, 71, 67], index=['Maths','Science','Hindi']),
    'Mallika': pd.Series([94, 95, 99], index=['Maths','Science','Hindi'])
}
ResultDF = pd.DataFrame(ResultSheet)
ResultDF['Marco'] = [90,90,90]
ResultDF['Fide'] = [80,80,80]
ResultDF['Riya'] = [70,70,70]

ResultDF.loc['SI'] = [70,70,70,70,70,70,70]
ResultDF.loc[0] = 0
ResultDF.loc['Promedio'] = ResultDF.mean()

ResultDF = ResultDF.drop(0, axis = 0)
ResultDF = ResultDF.drop(['Marco','Fide'], axis = 1)

print(ResultDF,"\n")
print(ResultDF.index)
print(ResultDF.columns)
print(ResultDF.columns.tolist(),"\n")

L = ResultDF.values.tolist() # numpy.ndarray to list
print("Lista:", L)
print("Indice 1:", L[1])
print("Filas, Columnas:", ResultDF.shape)
print("Tamaño:",ResultDF.size,"\n")

for i in L:
    print(i)

print(ResultDF.to_csv("csvs/Resultados.csv", sep = ','))
print(ResultDF.to_json("csvs/Resultados.json"))

ResultDFNuevo = pd.read_csv("csvs/Resultados.csv")
print(ResultDFNuevo)

"""#Serie = pd.Series([10,20,30])
#Serie = pd.Series(["Marco","Fidencio","Luis"], index=[2,1,0])
Serie = pd.Series([50,90,70], index=["Marco","Fide","Luis"])
#Serie = pd.Series([["Marco","ITI"],"Fidencio","Luis"])
#Serie = pd.Series([True, False, True])
#Serie = pd.Series([True, "False", 2])


#print(Serie)
#print(type(Serie))
#print(dir(Serie)) # imprime todos los metodos posibles de una clase

#for i in range (len(Serie)):
    #print(Serie.iloc[i])

#print("Valor de marco", Serie["Marco"])

array1 = np.array([1,2,3,4])
serie1 = pd.Series(array1)
#print(serie1)

dict1 = {'India': 'NewDelhi', 'UK': 'London', 'Japan': 'Tokyo', 'Mexico': 'Victoria'}
#print(dict1) #Display the dictionary
series2 = pd.Series(dict1)

#print("Valor", series2["Mexico"]) #Display the series
#print("Valor", series2.iloc[0])
#print(series2[["Mexico","Mexico","India"]])
#print(series2["India":"Japan"])
#print(series2[::-1])

seriesAlph = pd.Series(np.arange(10,16,1),index = ['a', 'b', 'c', 'd', 'e', 'f'])
seriesAlph.name = 'Calificaciones'
seriesAlph.index.name = 'Letras'
#print(seriesAlph)
#print(seriesAlph['c':'f'])
#print(seriesAlph.values) # devuelve un numpy array
#print(seriesAlph.size)
#print(seriesAlph.empty)

seriesTenTwenty = pd.Series(np.arange(10,20,1))
seriesTenTwenty2 = pd.Series(np.arange(10,25,1))
#print(seriesTenTwenty)
#print(seriesTenTwenty2)

#print(seriesTenTwenty + seriesTenTwenty2)

#print(seriesTenTwenty.head(3))
#print(seriesTenTwenty.tail(3))
#print(seriesTenTwenty.count())
print(seriesTenTwenty.add(seriesTenTwenty2, fill_value=0))"""