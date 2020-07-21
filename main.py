# Bu program yapay zeka ve python kullanarak bir kişinin şeker hastası olup olmadığını tespit etmeye çalışacak.


# Kullanılacak kütüphaneleri içeri aktarım
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Başlık ve alt başlık oluşturma
st.write("""
# Şeker Hastalığı Tespiti
Yapay Zeka ile birinin şeker hastası olup olmadığı durumunu tespit etmeye çalışır.
""")
# Web uygulamasında resmi açıp, görüntülemek
img = Image.open(r'C:\Users\Arda\PycharmProjects\AIbasedWebApp4DiabetesDetection\test image.png')
st.image(img, caption='AI', use_column_width=True)

# Makine öğrenmesi modelinin (RandomForestClassifier ile) kendine eğiteceği veriyi alma
df = pd.read_csv(r'C:\Users\Arda\PycharmProjects\AIbasedWebApp4DiabetesDetection\DiabetesDataset.csv')
# Web uygulamasına alt başlık oluşturma
st.subheader('Veri seti içerisindeki veriler:')
# Veri setindeki veriyi web uygulaması üzerine tablo olarak yansıtma
st.dataframe(df)
# Veriye ait bazı istatistik bilgileri (maximum, minumum, vb.) gösterme
st.write(df.describe())
# Veriyi grafik olarak gösterme
chart = st.bar_chart(df)

# Veriyi bağımsız 'X' ve bağımlı 'Y' değişkenlerine bölme
X = df.iloc[:, 0:8].values  # X değişkeni veri setindeki tüm özellikleri tutar (yani Outcome hariç tüm sütunlar).
Y = df.iloc[:, -1].values  # Y değişkeni ise Outcome sütununu tutar.
# Veri setindeki verinin %75'ini modelin eğitimi için, %25'ini ise testi için bölme
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Şeker hastası olup olmadığını anlamak istediğimiz bireyin değerlerini almak
def get_user_input():
    pregnancies = st.sidebar.slider('Hamilelikler', 0, 15, 1)  # kaç kere hamile kalındığının değeri (minimum=0, maximum=15, varsayılan=1)
    glucose = st.sidebar.slider('Kan Şekeri', 0, 199, 110) # kan şekeri değeri (minimum=0, maximum=199, varsayılan=110)
    blood_pressure = st.sidebar.slider('Tansiyon', 0, 140, 72) # tansiyon değeri (minimum=0, maximum=140, varsayılan=72)
    skin_thickness = st.sidebar.slider('Cilt Kalınlığı', 0, 99, 23) # cilt kalınlığı değeri (minimum=0, maximum=99, varsayılan=23)
    insulin = st.sidebar.slider('İnsulin', 0, 126, 100) # insulin değeri (minimum=0, maximum=126, varsayılan=100)
    bmi = st.sidebar.slider('Vücut Kitle İndeksi', 0.0, 50.0, 21.5) # BMI değeri (minimum=0.0, maximum=50.0, varsayılan=21.5)
    dpf = st.sidebar.slider('Diyabet Soyağacı Fonksiyonu', 0.0, 2.49, 0.3725) # diyabet soyağacı fonksiyonu değeri (minimum=0.0, maximum=2.49, varsayılan=0.3725)
    age = st.sidebar.slider('Yaş', 18, 99, 30) # yaş değeri (minimum=18, maximum=99, varsayılan=30)

    # Kullanıcıdan alınan değerlerin bir sözlük (dictionary) yapısında anahtar-değer (key-value) çiftleri şeklinde kayıt altına alınması
    user_data = {'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi, 'dpf': dpf, 'age': age}
    # Kullanıcı verisinin dataframe'e dönüştürülmesi
    features = pd.DataFrame(user_data, index=[0])
    return features

# Kullanıcının girdiği değerleri bir değişkende tutmak
user_input = get_user_input()  # user_input değişkeni kullanıcı girdilerini görüntülemek için kullanılacak

# Web uygulamasına alt başlık oluşturma ve kullanıcı girdilerini görüntüleme
st.subheader('Kullanıcı Girdileri:')
st.write(user_input)

# Yapay Zeka modelini oluşturup eğitmek
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Web uygulamasına alt başlık oluşturma ve model metriklerini (performansını) görüntüleme
st.subheader('Yapay Zeka Modeli Test Doğruluk Puanı:')
# Modeli Y_test veri setine göre test eder ve RandomForestClassifier modeline X_test veri setini vererek, Y_test'deki değerleri tahmin etme doğruluk puanını belirler
st.write('%' + str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100))  # yüzdelik değer elde etmek için 100 ile çarpıldı

# Girdisi (sağlık verisi) alınan kullanıcının şeker hastası olma ihtimali olup olmadığını belirleyebilmek için model tahminlerini bir değişkene atama
prediction = RandomForestClassifier.predict(user_input)

# Web uygulamasına alt başlık oluşturma ve sınıflandırmayı (şeker hastası mı değil mi) gösterme
st.subheader('Sınıflandırma:')
st.write(prediction)
