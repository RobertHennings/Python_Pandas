#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:56:08 2021

@author: Robert_Hennings
"""


#Python_Pandas
#Kapitel 1.4) ################################################################

import pandas as pd
#Datensatz importieren
df = pd.read_table("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/gapminder.tsv",sep='\t')

#Einen Überblick übder die Daten erlangen
df

df.isnull()

df.count()

#ersten paar Zeilen ansehen
df.head(10)

#letzten Zeilen anzeigen
df.tail(10)

#Deskriptive Übersicht anzeigen lassen
df.info()
#Die Dimensionen des Datensatzes lassen sich mit shape anzeigen
df.shape
#() werden weggelassen da es sich nicht um eine Funktion mit Argumenten handelt
#Es kann auf einzelne Attribute zugegriffen werden 
df.shape[0]
df.shape[1]
#Die einzelnen Attribute anzeigen lassen
df.columns

df.describe()

#Auf einzelne Einträge kann zugegriffen werden
#iloc greift über Zeilen und Spalten auf die Einträge in den eckigen Klammern zu
#Alle Zeilen der ersten Spalte anzeigen lassen
df.describe().iloc[:,1]

#Gesondert kann der Index als Spalte angezeigt werden
df.index

#Überprüfung des Datentyps des Dataframes
df.dtypes

#Kapitel 1.5) Einzelne Spalten und Zeilen auswählen
#Verschiedenes Anzeigen von Spalten auf 3 Möglichkeiten

#Variante 1) Mit Hilfe des Namens
df["country"]

#Nur ein paar Einträge von Country 
df["country"][0:5]

#Alternative mit df.Spaltenname
df.country[0:5]

#Per Namen lassen sich auch mehrere Spalten auswählen was die Lesbarkeit erleichtert

df[["country","continent","year"]]

#Variante 2) Mit der Index Position
#Die erste, dritte und vierte Spalte auswählen
df.iloc[:,[0,2,3]]

#Variante 3) Mit Hilfe eines Range Objekts
#Jeweils die zweite Spalte soll ausgegeben werden
small_range = list(range(0,df.shape[1],2))
small_range

df.iloc[:,small_range]


#Kapitel 1.5.2) ###############################################################
#Verschiedenes Anzeigen von Zeilen auf drei Varianten

#Variante 1) Auswahl durch Index Label
#Aufrufen des Index
print(df.index)
#Normalerweise wird von Pandas immer automatisch eine Indexierung vorgenommen
#außer bei Zeitreihen wo der Index Datetime Objekte darstellt

#Variante 1) Aufruf von Zeilen über die Indexierung 
df.iloc[df.index==2,:]

#Zum Aufruf von Zeilen wird .loc genutzt
df.loc[2]

#Datentypen überprüfen, denn diese sind verschieden
print(type(df.iloc[df.index==2,:]))

print(type(df.loc[2]))

#Die ersten 100 Zeilen
df.loc[0:99]

#Achtung df.loc[-1] funktioniert nicht
df.loc[-1]

#Daher andere Möglichkeit
df.loc[df.shape[0]-1]

#Oder mit .tail()
df.tail()

#Auswahl vereinzelter Zeilen
df.loc[[0,2,4]]

#Genauso kann dies über  .iloc passieren

df.iloc[[0,2,4],[0,3]]


#Kapitel Nr. 1.6) ############################################################
#Gruppierungen im Dataframe
#Übersicht über den Datensatz

df.head()

#Weitere analytische Fragen sind z.B:
#1) Was war die durchschnittliche Lebenserwartung pro Jahr, ebenso die Population und das GDP?
#2) Wie sehen die Angaben bei einer Aggregation auf Kontinenten aus?
#3) wie viele Länder gäbe es pro Kontinent?

#Gruppierungen durch groupby
df.columns
df.groupby("year")["lifeExp"].mean()

#Weitere Beispiele dazu
df.head()

#Maximale Lebenserwartung je Jahr
df.groupby("year")["lifeExp"].max()
#Oder auch minimale
df.groupby("year")["lifeExp"].min()

#Daraus neuen Datensatz erzeugen mit minimaler und maximaler durchschnittlicher Lebenserwartung pro Jahr

df2 = pd.DataFrame(df.groupby("year")["lifeExp"].min()).merge(df.groupby("year")["lifeExp"].max(),left_on="year",right_on="year")
df2.columns 
#Anpassen der Spaltennamen
df2.head()
df2.columns = ["Minimale Lebenserwartung","Maximale Lebenserwartung"]
df2.columns

#Grafische Darstellung
import matplotlib.pyplot as plt

plt.plot(df2.iloc[:,0],"blue",label="Max Life Exp")
plt.plot(df2.iloc[:,1],"red",label="Min Life Exp")
plt.xlabel("Years")
plt.ylabel("Life Expectation")
plt.legend(loc=2)
plt.show()


#Weitere groupby Befehle
df.head()

#Pro Jahr, je Land die Population
df.groupby("year")["pop"]

#nein es braucht immer noch eine zusätzliche Aggregatfunktion dazu

#Minimale Bevölkerung je Land
df.groupby("country")["pop"].min()

#Gruppierungen mit mehr als einer Variable können ebenfalls ausgeführt werden
#Die Spalten müssen erneut als Liste mitgegebeben werden

#Beispiel:
df.groupby(["year","continent"])[["lifeExp","gdpPercap"]].mean()

#Angaben sind nach Jahr und nach Kontinent gruppiert

#Gruppierte Frequenzen
#Es sollen unterschiedliche Frequenzen der Daten je Kategorie angegeben werden
#Wie oft taucht ein bestimmter Wert auf?
#Pandas Methoden dazu: 1) nunique() oder 2) value counts

#Variante 1) .nunique()
#Wie viele Länder gibt es pro Kontinent?
df.groupby("continent")["country"].nunique()

#.nunique() auf Lnderbasis
df.iloc[:,0].nunique()
#142 verschiedene Länder

#Auf Kontinentbasis
df.iloc[:,1].nunique()
#5 verschiedene Kontinente

#Variante 2) .value_counts
df.groupby("continent")["country"].value_counts()
#.value_counts() gibt nähere Informationen
df.head()
#Auf Länderbasis
df.iloc[:,0].value_counts()
#Auf Kontinentbasis
df.iloc[:,1].value_counts()


#Kapitel Nr. 1.7) ############################################################
#Einfache graphische Darstellungen mit Python (ohne import von Matplotlib)
#Nähere Analyse der durchscnittlichen Lebenserwartung pro Jahr
df_life = df.groupby("year")["lifeExp"].mean()
df_life.head()

#Plotten
df_life.plot()

#mit weiteren Angaben
df_life.plot(xlabel="Year",ylabel="Life Expectation",kind="line")
df_life.plot(xlabel="Year",ylabel="Life Expectation",kind="hist")

#Alle Varianten anzeigen lassen für kind

Variante = ["line","bar","hist","box","kde","density","area","pie","scatter","hexbin"]

for kind in Variante:
    df_life.plot(xlabel="Year",ylabel="Life Expectation",kind=kind)
    
    

#Kapitel Nr.2) ################################################################
#Es soll sich nachfolgend näher mit dem Datentyp des Pandas Package beschäftigt werden
#Zu sehen ist, dass die Spalten eines Dataframes "Serien" sind
type(df.iloc[:,0])
    
#Beispielhaft soll ein kleiner Datensatz erstellt werden

s = pd.Series(["Banane",1])
print(s)

#Der Index wird wieder automatisch erzeugt

s.index

#Dieser kann bei der Erstellung aber auch manuell zugewiesen weren

s = pd.Series(["erster Eintrag","zweiter Eintrag"],index=["Nummer 1","Nummer 2"])
print(s)


#Die übliche Methode kleine manuelle Dataframes zu erstellen ist durch Dictionaries
#Beispiel:
scientists = pd.DataFrame({
    "Name":["Rosaline Franklin","William Gosset"],
    "Occupation": ["Chemist","Statistican"],
    "Born": ["1920-07-25","1876-06-13"],
    "Died": ["1958-04-16","1937-10-16"],
    "Age": ["37","16"]})
scientists

#Der Key ist dann jeweils der Spaltenname und die Values sind die einzelnen Zeileneinträge
#Neuzuweisung des Index
scientists.index = ["First Scientist","Second Scientist"]
#Überprüfung
scientists

#Jedoch ist zu sehen dass die Reihenfolge nich garantiert sein muss

#Soll eine bestimmte Ordnung eingehalten werden, kann dies angegeben werden

scientists = pd.DataFrame({
    "Name":["Rosaline Franklin","William Gosset"],
    "Occupation": ["Chemist","Statistican"],
    "Born": ["1920-07-25","1876-06-13"],
    "Died": ["1958-04-16","1937-10-16"],
    "Age": ["37","16"]},
    
    index = ["Rosaline Franklin", "William Gosset"],
    columns = ["Occupation","Age","Born","Died"])
scientists 

# Kapitel Nr. 2.5) ############################################################
#Die "Series" im Detail
#Werden Zeilen mit .loc ausgewählt so erhält man ein "Series" Objekt 
first_row = scientists.loc["William Gosset"]

type(first_row)

#Die Ausgabe erfolgt nicht als Zeile sondern transponiert als Vektor
#Der Index stellt die Spaltennamen dar, die erste Spalte die ursprünglichen Zeileneinträge
#Spaltennamen
first_row.index
#Werte der ersten Spalte
first_row.values

#Alternativ könne die Indexwerte als Spalten auch mit .keys abgerufen werden

first_row.keys()

#Kapitel Nr. 2.5.1) ###########################################################
#Vergleich von pd.Sereis und ndarrays
#Es soll die Spalte age extrahiert werden

ages = scientists["Age"]
ages 
ages = pd.Series(list(range(0,10)))
#ages hat nun zahlreiche Methoden und Operationen

#Deskriptiv
#Minimum
ages.min()
#Maximum
ages.max()
#Mittelwert
ages.mean()

#Erweitert muss die Series auch als Dictionary Typ
ages.append(pd.Series({"Einstein": "23"}))
ages
#Korrelation und Kovarianz

ages.values.corr(other = ages.values)

ages.values.cov(other = ages.values)

#Beschreibung
ages.describe()
#Transponieren
ages.transpose()

#Zum Frame wandeln
ages_frame = ages.to_frame()

#Sortieren
ages.sort_values()

#Werte ersetzen
ages.replace({"William Gosset":"33"})
ages

ages.drop_duplicates()

scientists.get("Age")

scientists["Age"].hist()
ages.hist()

ages.sample()

#Kapitel Nr. 2.5.2) Boolean subsetting Series #################################
#Dafür wird ein neuer Datensatz benötigt
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/scientists.csv")

df.head()

ages = df["Age"]
ages 

#Einfache deskriptive Statistiekn
ages.mean()

ages.max()
ages.min()
ages.std()
ages.var()
ages.corr(other = ages)
ages.cov(other = ages)

#Gesamte einfachere Beschreibung
ages.describe()

#Nun sollen angaben nach bestimmten Bedingungen ausgegeben werden
#Alle die älter als der Durchnschnitt sind
ages[ages >ages.mean()]

#In anderer Sicht
print(type(ages>ages.mean()))

#Die Series kann auch mit boolschen Werten geteiot werden
#Es sollen die Indexzahleneinträge von 0,1,4,5 ausgegeben werden

manual_bool = [True,True,False,False,True,True,False,False]

ages.shape

ages[manual_bool]


#Kapitel Nr. 2.5.3.1 ##########################################################
#Sind die Vektoren gleich lang können Operationen auf beiden ausgeführt werden 
ages+ages

ages*ages

ages-ages

#100 auf jedes alter
ages+100

ages*2

#Vektoren mit unterschiedlichen Längen "Broadcasting"
#in Pandas wird der kürzere Vektor nicht weiderholt sondern mit NaN (not a Number) aufgefüllt

#Beispiel:
ages + pd.Series([1,100])



#Jedoch hängt das Verhalten vom type ab

import numpy as np
ages+np.array([1,100])
       
#Das praktische an Pandas ist, dass sich automatisch an gleichen Indexxierungen orientiert wird

ages

reversed_ages = ages.sort_index(ascending = False)

reversed_ages

#Es wird automatisch anhand des Index addiert
ages *2

ages+reversed_ages

#Kapitel Nr. 2.6) ############################################################
#DataFrame
#Mit bolschen Ausdrücken kann auch der DataFrame selektiert werden

df.head()
#Alle die älter sind als das mittlere alter

df[df["Age"]>df["Age"].mean()]

df.index

df[df.Age==61]
#Alternative Ausgabe

df[[True,True,False,False,True,True,False,True]]

#Kapitel Nr. 2.6.2) ###########################################################
#Operationen werden automatisch auf Gleichheit überprüft und vektorisiert
first_half,second_half = df[:4], df[4:]

first_half

second_half

first_half+second_half

df*2

#Kapitel Nr. 2.7) #############################################################
#Veränderungen am DataFrame und der pd.Series
#Die Spalten Born und Died sind an sich Objekte die verändert werden können

df["Born"].dtype
df["Died"].dtype

#Diese Typen sollen zu einem Datetimeobjekt gewandelt werden um entsprechende Operationen anwenden zu können
#Datetime Format: YYYY-MM-DD
#"Y-%m-%d"
born_datetime = pd.to_datetime(df["Born"],format="%Y-%m-%d")
born_datetime

#Ebenfalls selbiges für die Died Spalte

died_datetime = pd.to_datetime(df["Died"],format="%Y-%m-%d")
died_datetime


#Beide Spalten werden dem originalen Datensatz hinzugefügt
df["Born_dt"] = born_datetime
df["Died_dt"] = died_datetime


df.head()

#Direktes Ändern einer Spalte ohne Zwischenspeicherung und Hilfsvariablen

import random
random.seed(42)

random.shuffle(df["Age"])

#Subtrahieren der Datetimespalten ergibt die genauen Tage als Lebenszeit

df["Lebenszeit_Tage"] = df["Died_dt"]-df["Born_dt"]

df["Lebenszeit_Tage"]

#Diese Tage können nun einfach in Jahre umgewandelt werden

df["Age_dt"] = df["Lebenszeit_Tage"].astype("datetime")

#Kapitel Nr. 2.8) #############################################################
#Exportieren und Importieren von Daten

#Pickle 
#abspeichern kann mit Hilfe; Daten.to_pickle("Dateipfad.pickle") geschehen

names = df["Name"]
names

names.to_pickle("/Users/Robert_Hennings/Downloads/Names.pickle")


#Das ganze kann auch auf den ganzen DataFrame angewendet werden

df.to_pickle("/Users/Robert_Hennings/Downloads/Scientists.pickle")
#Sollen die Daten wieder eingelesen werden

scientists = pd.read_pickle("/Users/Robert_Hennings/Downloads/Scientists.pickle")

scientists

scientists == df


#CSV
#Selbige Anwendungen können mittels .to_csv()
#und mit pd.read_csv() getan werden

#TSV 
#Dafür muss eine spezielle Separation angewendet werden

#TSV = Tab Separated Values
scientists.to_csv(, sep="\t")

#Achtung: Beim Abspeichern kann die Indexspalte gelöscht werden da beim einlesen weider eien neue generiert wird
scientists.head()

#Excel
#Ebenso können die Methoden für Excel angewandt werden
#Hier kann noch der spezifische Sheet Name mitgegeben werden

scientists.to_excel("/Users/Robert_Hennings/Downloads/Scientists.xlsx",sheet_name = "Scientists",index=False)


#Kapitel Nr.3) ################################################################
#Daten graphisch darstellen
#Anscombe Dataset
#Enthält 4 Datensätze wobei jeder 2 kontinuierliche Variablen enthält
#Jeder Datensatz hat den gleichen Durchschnitt, Varianz und Korrelation und Regressionsgerade
#Graphisch sind die Unterschiede jedoch schnell zu sehen

import seaborn as sns
anscombe = sns.load_dataset("anscombe")

#Erste Analyse
anscombe.shape

anscombe.columns

#Unterschiedliche Datensätze
print(anscombe["dataset"].nunique())
anscombe["dataset"].value_counts()

anscombe.head()
#Nun werden sich deskriptive Statistiken je Datensatz angesehen
#Der Durchschnitt ist in allen Datensätzen gleich
anscombe.groupby("dataset")[["x","y"]].mean()

anscombe.groupby("dataset").var()
anscombe.groupby("dataset").std()

def Deskriptive_Angaben(df,group):
    print("---------Mean------------")
    print(df.groupby(str(group)).mean())
    print("---------Var------------")
    print(df.groupby(str(group)).var())
    print("---------Std------------")
    print(df.groupby(str(group)).std())


Deskriptive_Angaben(anscombe,"dataset")

anscombe.info()
anscombe.describe()
#Kapitel Nr. 3.4) #############################################################
import matplotlib.pyplot as plt

#Nun Darstellung des Datensatzes
#abspeichern der einzelnen Teile separat
dataset_1 = anscombe[anscombe["dataset"]=="I"]
dataset_2 = anscombe[anscombe["dataset"]=="II"]
dataset_3 = anscombe[anscombe["dataset"]=="III"]
dataset_4 = anscombe[anscombe["dataset"]=="IV"]

plt.plot(dataset_1["x"],dataset_1["y"])
#Ein Scatterplot als:
plt.plot(dataset_1["x"],dataset_1["y"],"o")

#Darstellung aller vier einzelnen Plots in einer Darstellung
#Definition von Subplots
#Argumente: 
    #1) anzahl der Zeilen
    #2) Anzahl der Spalten
    #3) Position des Subplots
    
fig = plt.figure()

#Man braucht 2 Zeilen und 2 Spalten
#Position 1
#erstes Subplot

axes1 = fig.add_subplot(2,2,1)
#zweites Subplot
axes2 = fig.add_subplot(2,2,2)
#drittes subplot
axes3 = fig.add_subplot(2,2,3)
#viertes Subplot
axes4 = fig.add_subplot(2,2,4)

#Befüllen der Subplots
axes1.plot(dataset_1["x"],dataset_1["y"],"o")
axes2.plot(dataset_2["x"],dataset_2["y"],"o")
axes3.plot(dataset_3["x"],dataset_3["y"],"o")
axes4.plot(dataset_4["x"],dataset_4["y"],"o")

#Hinzufügen von Überschriften
axes1.set_title("Dataset_1")
axes2.set_title("Dataset_2")
axes3.set_title("Dataset_3")
axes3.set_title("Dataset_4")

#Titel für die gesamte Figur
fig.suptitle("Anscombe Dataset")





#Kapitel Nr. 3.5) #############################################################
#Statistische Grafiken
#Der Datensatz
#Enthält Angaben zu Trinkgeldern zu verschiedenen Malzeiten unter anderem
tips = sns.load_dataset("tips")
print(tips.head())

tips.info()

#Für univariate Daten: Histogramm anfertigen

fig = plt.figure()
axes1 = fig.add_subplot(1,1,1)
axes1.hist(tips["total_bill"],bins=10)
axes1.set_xlabel("Frequency")
axes1.set_ylabel("Total Bill")
fig.show()

#Für bivariate Daten: Scatterplot

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(tips["total_bill"],tips["tip"])
axes1.set_xlabel("Total Bill")
axes1.set_ylabel("Tip")
axes1.set_title("Total Bill vs. Total Tip")
scatter_plot.show()

#Boxplots
boxplot = plt.figure()
axes1 = boxplot.add_subplot(1,1,1)
axes1.boxplot(
    #data as input
    #each piece as a list
    [tips[tips['sex'] == "Female"]["tip"],
          tips[tips["sex"] == "Male"]["tip"]
     
     ])
axes1.set_xlabel("Sex")
axes1.set_ylabel("Tip")

#3.5.3) multivariate plotting
#create a color variable based on the sex

def recode_sex(sex):
    if sex == "Female":
        return 0
    else: 
        return 1
    
tips["sex_color"] = tips["sex"].apply(recode_sex)

tips.head()

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(x=tips["total_bill"],y=tips["tip"],
              #set the size of the dots based on the party size
              #multiply the values by 10 and also to emphasize the difference
              s = tips["size"]*10,
              #set the color for the sex
              c = tips["sex_color"],
              #set the alpha so points are more transaprent
              alpha = 0.5)
axes1.set_title("Total Bill vs Tip colored by Sex and sized by 10")
axes1.set_xlabel("Total Bill")
axes1.set_ylabel("Tip")
scatter_plot.show()

#Kapitel Nr. 3.6) #############################################################
#Seaborn
#Loading the tips data set again and the seaborn library
#3.6.1 univariate distributions
#3.6.1.1 histograms
import seaborn as sns

tips = sns.load_dataset("tips")

#Creating Histograms using sns.distplot

hist = sns.distplot(tips["total_bill"])
hist.set_title("Total Bill histogram with density Plot")


#If just a histogram is needed then set the kde parameter to False
#or just using histplot
hist = sns.distplot(tips["total_bill"], kde = False)
hist.set_title("Total Bill histogram with density Plot")
hist.set_xlabel("Total Bill")
hist.set_ylabel("Frequency")

#3.6.1.2 Denisty Plot (Kernel Density Estimation)
#setting the parameter hist to False or just using kdeplot
den = sns.kdeplot(tips["total_bill"])
den.set_title("Total Bill histogram with density Plot")
den.set_xlabel("Total Bill")
den.set_ylabel("Frequency")

#3.6.1.3 Rug plots
hist_rug_plot = sns.distplot(tips["total_bill"], rug = True)
hist_rug_plot.set_title("Total Bill histogram with density 1 and rug plot")
hist_rug_plot.set_xlabel("Total Bill")
hist_rug_plot.set_ylabel("Density")


#3.6.1.4 Countplots

count = sns.countplot("day", data = tips)
count.set_title("count of days")
count.set_xlabel("Days")
count.set_ylabel("Absolute Frequency")

#3.6.2 bivariate distributions
#3.6.2.1 scatter plot

#use regplot because there is no specific scatter finction
#replot displays a regression line which can be set to fit_reg = False to not show it
scatter = sns.regplot(x="total_bill",y="tip",data=tips)
scatter.set_title("Scatterplot of total bill and tip")
scatter.set_xlabel("Total Bill")
scatter.setylabel("Tip")

#without the fitted line
scatter = sns.regplot(x="total_bill",y="tip",data=tips,fit_reg=False)
scatter.set_title("Scatterplot of total bill and tip")
scatter.set_xlabel("Total Bill")
scatter.setylabel("Tip")

#Also possible: Create scatter plot with showing the univariate plot on the side
scatter = sns.jointplot(x="total_bill",y="tip",data=tips)
scatter.set_axis_labels(xlabel="Total Bill",ylabel="Tip")
#add a title, set font size, move text above axes
scatter.fig.suptitle("Joint Plot of Total Bill and Tip",fontsize=20,y=1.03)


#3.6.2.2 Hexbin Plots







#Kapitel Nr. 4) #############################################################
#Data Assembly
#4.1) Introduction
#Focus on various data cleaning tasks, beginning with assembling a dataset for analysis

#4.4) Concatenation
#Appending a row or a column to the existing data

#concat function from pandas
#Importing the needed data
import pandas as pd

df1 = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/concat_1.csv")
df2 = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/concat_2.csv")
df3 = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/concat_3.csv")


print(df1)
print(df2)
print(df3)
#Note: alle have the same dimensions 
df1.shape == df2.shape == df3.shape
    
#Concatting ll dataframes on top of each other
#concatting the rows of all dataframes

row_concat=pd.concat([df1, df2, df3])

print(row_concat)
#Note that the index is also just stacked and not new reindexed

#Now trying different subsetting methods
print(row_concat.iloc[3,])
#If one is using only loc for the rows then it might potentially happen for the 3 without reindexing
#that this happens:
print(row_concat.loc[3])

#If now a new single row is created and should be appended to one of the df we face a problem
#Create a new row of data
new_row_series = pd.Series(["n1", "n2", "n3", "n4"])
print(new_row_series)

#Now attempting to append the series
print(pd.concat([df1, new_row_series]))
#We see by looking at the dimensions of the new row there lies the problem
new_row_series.shape
#Solution: Form a dataframe out of it with 1 row and multiple columns

new_row_df = pd.DataFrame(new_row_series)
print(new_row_df)

new_row_df=pd.DataFrame([["n1", "n2", "n3", "n4"]],columns=["A", "B", "C", "D"])
print(new_row_df)

#New attempt for concatinating
print(pd.concat([df1, new_row_df]))
#A new row is added


#if just a single object should be added then also the .append function can be used

print(df1.append(new_row_df))
print(df1.append(df2))


#Using a python dictionary
data_dict={"A": "n1",
           "B": "n2",
           "C": "n3",
           "D": "n4"}
data_dict.keys()
data_dict.values()

#Appending the dictionary

print(df1.append(data_dict,ignore_index=True))

#here the use of the ignore_index resets the index and builds a new one, a previous one isnt't repeated

#Example: Concatting all dfs and reindexing the outcome

row_concat_i = pd.concat([df1, df2, df3], ignore_index=True);print(row_concat_i)

#The index is resettet

#4.4.2 Adding columns

#As easy as the row version but setting the axis parameter to 1 instead of default 0

#Concatting columns
col_concat = pd.concat([df1, df2, df3], axis=1);print(col_concat)

#Subsetting the new dataframe as usual

print(col_concat["A"])

#Note: Now not the index is available multiple times but the column (names)
#Adding a new column to the frame by just declaring it
col_concat["new_col_list"] = ["n1","n2","n3","n4"]
print(col_concat)

#Now it i possible to just add a pd.Series
col_concat["new_col_series"] = pd.Series(["n1","n2","n3","n4"])
print(col_concat)


#Finally resetting the index if we concat
#by row
print(pd.concat([df1, df2, df3], ignore_index=True))
#By column
print(pd.concat([df1, df2, df3],axis=1, ignore_index=True))

#4.4.3 Concatentaion with different indices
#Now there will be different row names and different indice numbers so that they can't just be added and matche up with the existing data

#4.4.3.1 Concatenate rows with different columns
df1.columns = ["A","B","C","D"]
df2.columns = ["E","F","G","H"]
df3.columns = ["A", "C", "F", "H"]

print(df1)
print(df2)
print(df3)

#Now a combination is harder
row_concat = pd.concat([df1, df2, df3])
print(row_concat)

#it will be filled up with NaN at the positions that dont match
#to avoid the NaN values we have to merge only the columns that are in common in all dfs
#therefore we have to use the join function of concat and in it the parameter outer which is per default outer
#it has to be switched to inner
#follows the sql logic

row_concat = pd.concat([df1, df2, df3], join="inner")
print(row_concat)
#Only the index values are tzhe same , the columns are all different

#If we only use df1 and df3 which have columns in common then it will display like that


print(pd.concat([df1, df3], ignore_index=False, join="inner"))

#Outer is like a full outer join in sql

#4.4.3.2 concatenate columns with different rows
#Manipulate the indexes

df1.index = [0,1,2,3]
df2.index = [4,5,6,7]
df3.index = [0, 2, 5, 7]

#Watch the indexes
print(df1)
print(df2)
print(df3)

#Alle have slightly different ones

#Now concatenating with axis 1 and 0
#columnwise
col_concat = pd.concat([df1, df2, df3], axis=1)
print(col_concat)


#rowwise
col_concat = pd.concat([df1, df2, df3], axis=0)
print(col_concat)

#now only keeping the matching results
print(pd.concat([df1, df3], axis=1,join="inner"))


#4.5 Merging multiple datasets
#the pd.join uses pd.merge under the hood, pd.join merges based on the index while pd.merge has more options
#if only merging by the row index is necessary then pd.join works fine


#Loading the data
person = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_person.csv")
site = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_site.csv")
survey = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_survey.csv")
visited = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_visited.csv")




print(person)
print(site)
print(survey)
print(visited)


#merge has the following features
#the called df is referred to as the left one, then the right one has to be defined
#then the how and the on what specifies the columns to match on, if left and right have no cols in common
#then use left_on and right_on

#4.5.1 one to one
#merge two dfs where we want to join one or more columns to antother column
#modify the visited frame
print(visited)
print(site)

visited_subset = visited.loc[[0,2,6],]
print(visited_subset)
#the columns with corresponding values are site in visited and name in site
#default for how is inner


o2o_merge = site.merge(right=visited_subset,left_on="name", right_on="site")
print(o2o_merge)

#4.5.2 many to one
#now we will use the visit df without subsetting it so it has the keys in the col site repeated
#then in the merge the corresponding rows in the other one will be repeated as often as needed to fill the rows up
#what is exactly what we need for the lat long positions to show up
m2o_merge = site.merge(right=visited, left_on="name", right_on="site")
print(m2o_merge)

#4.5.3 many to many
#match based on multiple columns
#2 dfs that come from person merged with survey and o´the other one from visited merged with survey

ps = person.merge(right=survey, left_on="ident", right_on="person")
vs = visited.merge(right=survey, left_on="ident", right_on="taken")
print(ps)
print(vs)



#now we pass a list of columns on which should be merged
ps_vs = ps.merge(right=vs,
                 left_on=["ident","taken","quant","reading"],
                 right_on=["person","ident","quant","reading"])
print(ps_vs)

#Look at the first row

print(ps_vs.loc[0, ])

#to prepare for collisions in same names in the frame pandas adds ys and xs to the ends of column names
#x refers the values from the left frame , y the ones from the rigt one


#Kapitel Nr.5) Missing Data ######################################################

#5.2 What is a NaN value
#Looking at the NaN values from numpy we see that each representation isnt eqaul to one another

#import numpy missing values 

from numpy import NaN, NAN, nan

#Testing the equality
print(NaN == True)
print(NaN == False)
print(NaN == 0)


#Missing values are not equal to missing values

print(NaN == NAN)
print(NaN == nan)


#Pandas built in function for testing wether there are missing values: pd.isnull()

print(pd.isnull(NaN))
print(pd.isnull(NAN))
print(pd.isnull(nan))

#and for not null use: pd.notnull()

print(pd.notnull(NaN))
print(pd.notnull(42))
print(pd.notnull("Hallo"))



#5.3 Where do missing values come from?
#5.3.1 Loading data

#as the visited dataset was loaded in pandas directly detected missing values
print(visited)
print(pd.isnull(visited))

#pd.read_csv has three paarmeters related to missing values
#na_values
#keep_default_na
#na_filter




#na_values: specify additional missing values 
#pass in a str or a list 
#some health data displays missing values as 99 or 88 there is na_values=[99]



#keep_default_na is true or false, per default true if false then only the values speicified in na_values are used
#without default missing values

visited = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_visited.csv", keep_default_na=False)
print(visited)
#the entry is just blank
#na_filter is a boolean, per default true , means that missing values will be coded as NaN, if false then nothing will be recoded

#manually specify missing values
visited = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/survey_visited.csv", keep_default_na=False, na_values=[""])
print(visited)
#the blank entry is filled with NaN


#5.3.3 User Input values
#Missing data manually created by the user

#missing values in a series
num_legs=pd.Series({"goat": 4,
                    "amoeba": nan})
print(num_legs)

#missing value in a dataframe

scientists = pd.DataFrame({
    "Name": ["Rosaline Franklin", "William Gosset"],
    "Occupation": ["Chemist", "Statistican"],
    "Born": ["1920-07-25", "1876-06-13"],
    "Died": ["1958-04-16", "1937-10-16"],
    "missing": [NaN, nan]})

print(scientists)

#directly assigning a whole column of missig values to the dataframe
scientists["NaNs"] = nan

print(scientists)


#5.3.4 Re-Indexing
#Another way to introduce missing values is by reindexing when adding new data
##Reindexing is useful whenn adding new data but retaining its orignal order
#common usage when the index reprsents a time interval and more data shoul be added

#Look at the gapminder dataset from 2000 to 2010

gapminder = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/gapminder.tsv", sep="\t")

print(gapminder)
gapminder.shape
#looking at the life Expectation per year

life_exp = gapminder.\
    groupby(["year"])["lifeExp"].\
        mean()

print(life_exp)
#Now show the data from 2000 to 2010

len(life_exp)

gapminder.describe()

#save as dataframe and reindex the dataframe

life_exp = pd.DataFrame(life_exp, index= range(2000,2012))
print(life_exp.loc[range(2000,2010)])




#Chapter 5.4) Working with missing data

#5.4.1. Find and count missing data

ebola = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/country_timeseries.csv")
print(ebola.head())

#One way to look at missing data is to count them

print(ebola.count())

#subtract the number of nonmissing from the total number of rows

num_rows_ebola = ebola.shape[0]
num_missing_ebola = num_rows_ebola - ebola.count()
print(num_missing_ebola)

#Count the total number of missing values or count them for a specific column
#use the count_nonzero function from numpy combined with isnull

import numpy as np

print(np.count_nonzero(ebola.isnull()))
#for specific column
print(np.count_nonzero(ebola["Cases_Guinea"].isnull()))

#another way to count missing data is to use value_counts on a series, this prints a frequency table of values

#Cases Guinea
print(ebola.Cases_Guinea.value_counts(dropna=False).head())

#5.4.2 Cleaning missing data
#5.4.2.1 Recode/Replace

#Use the fillna method to replace the misisng values by a specific form
print(ebola.loc[0:15])

#Replace the NaNs in the first 15 rows of all columns
print(ebola.fillna(0).iloc[0:15,0:5])

#fillna has the parameter inplace what means teh underlying data will be automatically changed without creating anew copy with the changes
#keeping in mind for larger datasets to save memory

#5.4.2.2 Fill Forward

#fillna can fill in missing values forward or backwards
#forward filling takes the last known record and uses this as replacement for the missing value
#if the series begins with a nan then it will stay blank 

print(ebola.fillna(method="ffill").iloc[0:15,0:5])

#5.4.2.3 Backwards
#backward fills the misisng values with the newest value 

print(ebola.fillna(method="bfill").iloc[0:15,0:5])

#5.4.2.4 Interpolate missing values

#Pandas can interpolate missing values
print(ebola.interpolate().iloc[0:15,0:5])
#Interpolate has different methods
print(ebola.interpolate(method="polynomial", order=5).iloc[0:15,0:5])

#5.4.2.5 Drop missing Values

#Last way to handle missing data is to drop it from rows or columns or the whole dataset
#dropna has the parameters: how (a row or a column), when (any or all)
#thresh parameter lets the user specify how many NaN values are acceptible before a column is dropped

print(ebola.shape)
#Whar if we want to remove all missing values?
ebola_dropna = ebola.dropna()

print(ebola_dropna.shape)
#One row is left

#so we have to sepcify wehen and where values are dropped

print(ebola_dropna)

#5.4.3 Calculations with missing data

#Look at the case counts for multiple regions
#add region stogether in a new column
ebola["Cases_multiple"] = ebola["Cases_Guinea"]+\
                            ebola["Cases_Liberia"]+\
                            ebola["Cases_SierraLeone"]
                            
#Look at the results:
ebola_subset = ebola.iloc[:,[2,3,4,18]]
print(ebola_subset.head(n=10))

#The new column is created only when all three other columns are not NaN, if one NaN is included then the calculation will return a NaN
#functions like mean or sum have a skipna parameter
#it skipps the missing value

print(ebola.Cases_Guinea.sum(skipna=True))
print(ebola.Cases_Guinea.sum(skipna=False))


#Chapter 6) Tidy Data
#6.2) columns contain values not variables

#6.2.1) Keep one column fixed

#load the data
import pandas as pd
pew = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/pew.csv")

print(pew.head())
#The data regarding the income are spread across multiple columns 
#not every column is a (new) variable

#show only the first few columns:
pew.iloc[:,0:6]

#Data need to be melted down, pandas has a function for that: melt
#the parameters of melt:
#id_vars: the variables that will remain as is
#value_vars:columns that should be melted, by default it will melt all columns not specified in id_vars
#var_name: new column name when the value_vars are melted down , by default "variable"

#we do not need to specify value_vars since we want all columns to melt except the religion one

pew_long = pd.melt(pew, id_vars="religion")
print(pew_long)

#Rebame the default values 

pew_long = pd.melt(pew, id_vars="religion",
                   var_name="income",
                   value_name="count")
print(pew_long)


#6.2.2 Keep multiple columns fixed
#load the billboard dataset

billboard = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/billboard.csv")

print(billboard.iloc[0:5,0:7])

billboard_long = pd.melt(billboard,
                         id_vars = ["year","artist","track","time","date.entered"],
                         var_name="week",
                         value_name="rating")


print(billboard_long)

#6.3 Columns contain multiple variables

#multiple columns represent multiple variables, common in health data
#again looking at the ebola dataset
import pandas as pd
ebola = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/country_timeseries.csv")

print(ebola.columns)
print(ebola.shape)

#print selected rows
print(ebola.iloc[:5,[0,1,2,3,10,11]])

#Cases_Guinea and Deaths_Guinea are two different variables although similar named
#again melt the data down

ebola_long = pd.melt(ebola,id_vars = ["Date","Day"])

print(ebola_long.head())

#6.3.1) Split and add columns individually (simple method)
#Column of interest can be split up by the underscore

#get the variable column
#access the string methods
#split the column by the delimiter

variable_split = ebola_long.variable.str.split("_")

print(variable_split[:5])

#the new variable is returned as a Series
#returned in a list

print(type(variable_split))

#first element in the container
print(type(variable_split[0]))

print(variable_split[0][0])
print(variable_split[0][1])

status_values = variable_split.str.get(0)
country_values = variable_split.str.get(1)


print(status_values[:5])
print(country_values[:5])

#Now add the values to the ebola_long dataframe as new columns
ebola_long["status"] = status_values
ebola_long["country"] = country_values


print(ebola_long.head())


#6.3.2) Split and combine in a single step (simple method)
variable_split = ebola_long.variable.str.split("_",expand=True)
variable_split.columns = ["status","country"]
ebola_parsed = pd.concat([ebola_long,variable_split],axis=1)

print(ebola_parsed)    

#6.4) Variables in both rows and columns
#A column of data holds two variables
#this case: pivot or cast the variable into separate columns

weather = pd.read_csv("https://raw.githubusercontent.com/RobertHennings/Python_Pandas/main/weather.csv")
print(weather.iloc[:5,:12])

#for each site and each day there is given the maximum and the minimum temperature (tmin, tmax)
#the element column contains variables that need to be casted/pivoted to become new columns 
#and the day variables need to be melted into row values

#first melt/unpivot the day variables
weather_melt = pd.melt(weather,
                       #variables that should stay as is
                       id_vars=["id","year","month","element"],
                       #elements that shoul be transformed
                       var_name="day",
                       #values of the new transformed variable
                       value_name="temp")

print(weather_melt.head())


#Next step: pivot up the variables stored in the element column 
#pivot_table can be used on any dataframe and isnt tied to pandas

weather_tidy = weather_melt.pivot_table(
    index = ["id","year","month","day"],
    columns = "element",
    values = "temp")

print(weather_tidy.head())

weather_tidy_flat = weather_tidy.reset_index()
print(weather_tidy_flat.head())

#6.5) Multiple Observational Units in a table (Normalization)

#Look at the billboard data set
print(billboard_long.head())