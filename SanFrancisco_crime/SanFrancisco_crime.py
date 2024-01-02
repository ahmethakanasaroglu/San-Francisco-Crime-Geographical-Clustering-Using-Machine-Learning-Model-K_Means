from sklearn.cluster import KMeans            # numerik veriler üzerinde calısır
import pandas as pd
from sklearn.preprocessing import MinMaxScaler    # enlem ve boylamları aynı aralığa çekmek için kullanıcaz.

df=pd.read_csv("C:\\Users\\asaro\\Downloads\\train.csv")
df.head()
## Y enlem(latitude), X boylam(longitude).

df = df.drop(['PdDistrict','Address','Resolution','Descript','DayOfWeek'],axis=1) # işlemleri enlem ve boylamlardan yani x-y den yapacağımız için bu sütunlara gerek yok. istersek bilgi olması acısından tutulabilirdi.axis=0 olsaydı row drop olurdu.

df.tail()
print(df.isnull().sum())   # no have null data

#%% 2003-2015 verileri mevcut. Sadece 2014 verilerini kullanmak için filtering operation yapıcaz.

f = lambda x: (x["Dates"].split())[0]       # dates sütununu split et ve 0. sütunu al. split bosluga göre ayırdıgı için 2 sütun olcak. sadece tarihi alıcaz. 
df["Dates"] = df.apply(f,axis=1)      # dates sütununa f işlemini uygula dedik
print(df.head())            # gördügümüz gibi sadece date kısmı kaldı dates columnında. time tarafı yok.

f = lambda x: (x["Dates"].split('-'))[0]      # - ye göre split edip 0.ind1eksi al yani sadece yılı getiricek. birazdan sadece 2014 olanları al dicez
df["Dates"] = df.apply(f,axis=1)              # apply ettik.
print(df.head())                 # dates sütununda sadece yıl kaldı
print(df.tail())

df_2014 = df[(df.Dates=='2014')]     # dates 2014 yılı olanları getirdik
df_2014.head()

#%%    SCALING

scaler = MinMaxScaler()          # scale data for accurate results
# Y enlem, X boylam 

scaler.fit(df_2014[['X']])
df_2014['X_scaled'] = scaler.transform(df_2014[['X']])

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']]) 


df_2014.head()

#%%      decide how many clusters we will have using Elbow Method (find K value)
print(df_2014[['X_scaled', 'Y_scaled']].isnull().sum())
    
k_range = range(1,15)

list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled','Y_scaled']])     # clustered yapılacak column isimleri
    list_dist.append(model.inertia_)      # modelin her inertia değerini list_dist'e atıyoruz
    
from matplotlib import pyplot as plt

plt.xlabel('K')
plt.ylabel('Distortion Value (inertia)')
plt.plot(k_range,list_dist)
plt.show()                           # x ekseninde K değeri, y ekseninde inertia değeri olcak şekilde grafikle.
                   # X dirsek noktası X =5 değerinin oldugu yerde. yani küme sayımız 5 demek.
                   
        
#%% build model and perform the clustering operation using K-Means(machine learning algorithm)

model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled','Y_scaled']])  # scale edilmis enlem boylamları vererek fit ettirdik
y_predicted        # y_predict dedigimiz şey aslında classlarımızın numarasıdır.

df_2014['cluster'] = y_predicted
df_2014                          # her bir satır(suç) için 0-4(5 cluster) arası cluster degerleri atadı.

## bu kümelemeyi coğrafik verilere göre yaptı.(x_scaled ve y_scaled degerlerine göre)


#%% visualize our clustering results - geographical map building using our machine learning model results
# haritalama işlemleri için en sık kullanılan kütüphanelerden 'plotly'

import plotly.express as px

figure = px.scatter_mapbox(df_2014, lat='Y', lon='X',
                           center = dict(lat = 37.8, lon = -122.4),      # center of map, coordinate of San Francisco
                           zoom = 9,                                # zoom of the map
                           opacity = .9,                        # opacity of the map a value between 0 and 1 # transparanlık(mapdeki verilerin ne saydamlıkta görüneceği)
                           mapbox_style = 'stamen-terrain',   #(dağlık ve düz yerleri gösterir.)     # 'open-street-map'(caddeleri vs gösterir),      # basemap # verileri gösterdiğimiz haritanın altlığı
                           color = 'cluster',                 # renklendirmeyi cluster sütununa göre yap diyoruz
                           title = 'San Francisco Crime Districts',    # title direkt
                           width = 1100,                         # map genişliği
                           height = 700,                           # map yüksekliği
                           hover_data = ['cluster', 'Category', 'Y', 'X'])     # mapde mouse ile durduğumuz noktada detay bilgi verirken bilginin hangi sütunları içereceğini verdik.   

figure.show()

# %% Export our resulting map into a html file . So that it can be used in any web site easily 

import plotly 
plotly.offline.plot(figure , filename = 'maptest.html', auto_open = True)

#if u want to use another basemap or use other methods of plotly u can get info using below;
help(px.scatter_mapbox)




                           
                           
                  





























