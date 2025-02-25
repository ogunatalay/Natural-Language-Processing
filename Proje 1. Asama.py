import jsonlines
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam



# NLTK Türkçe stopwords'leri indirme
nltk.download('stopwords')
turkce_stopwords = set(stopwords.words('turkish'))



# Hakaret ve zorbalık içeren anahtar kelimeler listesi
hate_words = {'kötü', 'salak', 'aptal', 'nefret', 'çirkin', 'haksız', 'yalancı', 'amk', 'aq', 'manyak', 'köpek', 'gerizekalı'}

# Temizleme ve hakaret kelimelerini ağırlıklandırma fonksiyonu
def temizle_ve_agirliklandir(text):
    text = text.lower()  # Tüm harfleri küçük yapma
    text = re.sub(r"@\w+|https?://[A-Za-z0-9./]+", '', text)  # URL ve kullanıcı adı temizleme
    text = re.sub(r"[^a-zA-Z0-9ığüşöçİĞÜŞÖÇ\s]", '', text)  # Özel karakter temizleme

    # Kelimelere ayırma
    text_words = text.split()

    # Stopwords temizleme ve kelimelerin köklerine indirgenmesi
    filtered_words = [kelime for kelime in text_words if kelime not in turkce_stopwords]

    # Ekleri kaldırma (Türkçe ekleri ayıklamak için basit bir regex)
    filtered_words = [re.sub(r"(de|den|da|ki|mi|lar|ler)$", '', kelime) for kelime in filtered_words]

    # Tekrar eden kelimeleri kaldırma
    filtered_words = list(set(filtered_words))  # Set kullanarak tekrarları engelle

    # Hakaret kelimelerini daha fazla vurgulama
    for word in hate_words:
        if word in filtered_words:
            filtered_words += [word] * 5  # Hakaret kelimelerini 5 kat artır

    # Sonuç olarak düzenlenmiş metni döndürme
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Veri setini yükleme fonksiyonu
def load_data(filename):
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame(data)

# Veri setini yükle
train_df = load_data('train.jsonlines')


# Temizlenmiş metin sütunlarını oluşturma
train_df_cleaned = train_df.copy()
train_df_cleaned["CLEAN_TEXT"] = train_df_cleaned["text"].map(lambda x: temizle_ve_agirliklandir(x))

# 'label' sütunundaki 'offensive' ve 'not-offensive' değerlerini 1 ve 0'a çevirme
train_df_cleaned['label'] = train_df_cleaned['label'].map({'offensive': 1, 'not-offensive': 0})

# EDA - Veri Yapısı İnceleme
def eda(df, name):
    print(f"\n{name} DataFrame Head:")
    print(df.head())
    print(f"\n{name} Missing Values:")
    print(df.isnull().sum())

    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
    plt.title(f'{name} Data Box Plot')
    plt.show()

    sns.countplot(x='label', data=df)
    plt.title(f'{name} Label Distribution')
    plt.show()

# EDA uygulama
eda(train_df_cleaned, 'Train')

# Hakaret kelimelerini belirgin yapmak için ağırlıklandırma fonksiyonu
def generate_wordcloud_with_weights(text_data, title):
    custom_stopwords = turkce_stopwords.union({
        'bir', 'bu', 'de', 've', 'ile', 'ben', 'ama', 'çok', 'hiç', 'şu', 'ne', 'neden', 'gibi', 'daha', 'ya', 'ya da',
        'her', 'hiçbir', 'bunu', 'şey', 'oldu', 'olmaz', 'bu kadar', 'kadar', 'sonra', 'önce', 'şimdi',
        'dönem', 'şu an', 'bunu', 'onlar', 'yok', 'oluyor', 'olan', 'şimdiye', 'belki', 'öyle', 'için', 'ki',
        'ama', 'bile', 'hep', 'kim', 'nasıl', 'işte', 'başka', 'onun', 'sizin', 'bizim', 'bunlar', 'arasında',
        'bununla', 'işte', 'tamam', 'sadece', 'göstermek', 'birçok', 'katkı', 'bunu', 'bile', 'nerede', 'en',
        'değil', 'olmak', 'hepsi', 'neden', 'hangi', 'arada', 'çoğu', 'ama', 'sizin', 'biz', 'onlar', 'ama',
        'çünkü', 'şimdi', 'henüz', 'çok', 'daha', 'gerçekten', 'sadece', 'şu', 'ile', 'gibi', 'bunu', 'yani',
        'hadi', 'burası', 'gösterdi', 'göstermek', 'benim', 'onlar', 'hadi', 'ki', 'bak', 'belirli', 'şu kadar',
        'hiçbir zaman', 'bazen', 'gerek', 'bazı', 'için', 'yerine', 'sonunda', 'arasında', 'gibi', 'daha çok',
        'çünkü', 'neden', 'isterseniz', 'neredeyse', 'zaten', 'genellikle', 'aslında', 'henüz', 'neden', 'bunu',
        'kendisi', 'bu yüzden', 'tek', 'birisi', 'birkaç', 'çokça', 'kendi', 'şeyler', 'sonuçta', 'gerçekten',
        'şimdiye kadar', 'öncelikle', 'yapmak', 'olduğu', 'özellikle', 'sadece', 'işte', 'bütün', 'onlarca', 'birde',
        'herkes', 'herhangi', 'tartışmasız', 'belki', 'ya da', 'kesinlikle', 'hemen', 'şunlar', 'bunlar', 'en son',
        'şu şekilde', 'her zaman', 'gerekli', 'daha önce', 'genellikle', 'zaman zaman', 'en iyi', 'çok fazla',
        'hiçbir şekilde', 'bazı şeyler', 'katılıyorum', 'bu kadar', 'her biri', 'sonuç olarak', 'bu da', 'bazı',
        'olabilirdi', 'işin gerçeği', 'en büyük', 'göstermek için', 'çok fazla', 'hepsi', 'tek bir', 'gerçekten',
        'işin sırrı', 'var', 'kötü', 'bi', 'böyle', 'mi', 'iyi', 'artık', 'bana', 'olsun', 'hala', 'olur', 'olmuş',
        'yok', 'bunun', 'i', 'e', '3', '1', '2', '4', '5', '6', '7', '10', '9', '8', 'a', 'böy', '10', '20', '15'
    })

    # Kelimelerin sıklığını hesaplayın
    word_frequencies = {}
    for word in ' '.join(text_data).split():
        if word not in custom_stopwords:
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1

    # Hakaret kelimeleri için özel ağırlıklandırma
    for word in hate_words:
        if word in word_frequencies:
            word_frequencies[word] *= 5  # Hakaret kelimelerine 5 kat ağırlık ekleyin

    # WordCloud oluşturun
    wordcloud = WordCloud(stopwords=custom_stopwords, width=800, height=800, background_color='white').generate_from_frequencies(word_frequencies)

    # WordCloud görüntüsü
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# Wordcloud grafiğini oluştur
generate_wordcloud_with_weights(train_df_cleaned['CLEAN_TEXT'], 'Train Data WordCloud')

# Model oluşturma ve sınıflandırma
X = train_df_cleaned['CLEAN_TEXT']  # 'CLEAN_TEXT' metin verisi içeren sütun
y = train_df_cleaned['label']       # 'label' hedef etiketi içeren sütun

# Eğitim ve test verilerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF hesaplama ve her tweet için kelimelerinin ağırlıklarını elde etme fonksiyonu
def calculate_tfidf_weights(df):
    vectorizer = TfidfVectorizer(max_features=5000)  # max_features belirleyerek sadece en sık kullanılan 5000 kelimeyi alıyoruz
    tfidf_matrix = vectorizer.fit_transform(df['CLEAN_TEXT'])

    # Kelimelerin adlarını alalım
    feature_names = vectorizer.get_feature_names_out()

    # Sonuçları bir DataFrame olarak düzenleyelim
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    return tfidf_df, vectorizer  # vectorizer'ı da döndürelim

# TRAIN veri seti için TF-IDF hesaplama
train_tfidf, vectorizer = calculate_tfidf_weights(train_df_cleaned)

# TF-IDF değerlerine göre her tweetin ağırlıklarını görmek için bir örnek yazdırabiliriz
print("Train Data TF-IDF Head:")
print(train_tfidf.head())

# Modelleri tanımlama
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'K-Nearest Neighbours': KNeighborsClassifier(),
    'Support Vector Machines': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Eğitim ve test verisi ile her modeli değerlendir
for model_name, model in models.items():
    print(f"\n{model_name} Modeli Başlatılıyor...")

    # Pipelines kullanarak modeli kuruyoruz
    model_pipeline = make_pipeline(vectorizer, model)

    # Eğitim verisi üzerinde modeli eğitelim
    model_pipeline.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yapalım
    y_pred_test = model_pipeline.predict(X_test)

    # Sonuçları değerlendirme
    print(f"{model_name} Accuracy on Test Data: {accuracy_score(y_test, y_pred_test)}")
    print(f"{model_name} Classification Report on Test Data:")
    print(classification_report(y_test, y_pred_test))

# --- Görselleştirme Kısmı ---

# 400 en sık kullanılan kelimeyle yeni bir TF-IDF hesaplama
tfidf_vectorizer = TfidfVectorizer(max_features=200)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_df_cleaned['CLEAN_TEXT'])

# TF-IDF tablosunu DataFrame olarak görselleştirme
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Data Head (Top 20 Features):")
print(tfidf_df.iloc[:20, :20])

# 20x20'lik bir TF-IDF matrisi parçası seçme
tfidf_sample_df = tfidf_df.iloc[:20, :20]

# Isı haritasını çizdirme
plt.figure(figsize=(12, 8))
sns.heatmap(tfidf_sample_df, cmap="viridis", annot=True, cbar=True)
plt.title("TF-IDF Matrisinin Isı Haritası (20x20)")
plt.xlabel("Özellikler")
plt.ylabel("Örnekler")
plt.show()

# --- 3 Boyutlu SVD Uygulaması ---

# TF-IDF matrisini 3 boyuta indirmek için TruncatedSVD kullanıyoruz
def apply_svd_to_tfidf(tfidf_matrix, n_components=3):
    svd = TruncatedSVD(n_components=n_components)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    return svd_matrix, svd

# TF-IDF hesaplama ve 3 boyutlu SVD'yi uygulama
tfidf_matrix = vectorizer.fit_transform(train_df_cleaned['CLEAN_TEXT'])
svd_matrix, svd = apply_svd_to_tfidf(tfidf_matrix)

# SVD ile indirgenmiş verinin ilk 5 satırını yazdıralım
print("SVD ile 3 Boyuta İndirgenmiş Verinin İlk 5 Satırı:")
print(svd_matrix[:5])

# --- Görselleştirme Kısmı ---

# SVD sonucu 3 boyutlu veri ile scatter plot çizme
# SVD sonuçlarını bir DataFrame'e dönüştürme
svd_df = pd.DataFrame(svd_matrix, columns=[f"Component {i+1}" for i in range(svd_matrix.shape[1])])

# 3D scatter plot çizme
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot için X, Y, Z eksenlerini seçme
x = svd_df['Component 1']
y = svd_df['Component 2']
z = svd_df['Component 3']

# Scatter plot oluşturma
ax.scatter(x, y, z, c=train_df_cleaned['label'], cmap='viridis', s=50, alpha=0.7)

# Başlık ve etiketler
ax.set_title('3D Scatter Plot of TF-IDF Reduced to 3D with SVD')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()

# --- 2 Boyutlu SVD Uygulaması ---

# TF-IDF matrisini 2 boyuta indirmek için TruncatedSVD kullanıyoruz
def apply_svd_to_tfidf_2d(tfidf_matrix, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    return svd_matrix, svd

# TF-IDF hesaplama ve 2 boyutlu SVD'yi uygulama
tfidf_matrix = vectorizer.fit_transform(train_df_cleaned['CLEAN_TEXT'])
svd_matrix_2d, svd_2d = apply_svd_to_tfidf_2d(tfidf_matrix)

# SVD ile indirgenmiş verinin ilk 5 satırını yazdıralım
print("SVD ile 2 Boyuta İndirgenmiş Verinin İlk 5 Satırı:")
print(svd_matrix_2d[:5])

# --- Görselleştirme Kısmı ---

# SVD sonucu 2 boyutlu veri ile scatter plot çizme
# SVD sonuçlarını bir DataFrame'e dönüştürme
svd_df_2d = pd.DataFrame(svd_matrix_2d, columns=[f"Component {i+1}" for i in range(svd_matrix_2d.shape[1])])

# 2D scatter plot çizme
plt.figure(figsize=(10, 7))
plt.scatter(svd_df_2d['Component 1'], svd_df_2d['Component 2'], c=train_df_cleaned['label'], cmap='viridis', s=50, alpha=0.7)

# Başlık ve etiketler
plt.title('2D Scatter Plot of TF-IDF Reduced to 2D with SVD')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.show()

# Gerekli kütüphaneleri ekliyoruz
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# MLP modelini eklemeden önce metin verisini Tokenizer ile işleyelim
tokenizer = Tokenizer(num_words=5000)  # En fazla 5000 kelimeyi dikkate alalım
tokenizer.fit_on_texts(train_df_cleaned['CLEAN_TEXT'])

# Veriyi sayısal dizilere dönüştürelim
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Dizileri sabit uzunlukta olacak şekilde pad edelim
max_length = 100  # Her örneğin uzunluğunu 100 kelimeye eşitleyelim
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)


# MLP modelini oluşturma
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_length))  # Embedding katmanı
    model.add(Flatten())  # Düzleştirme katmanı
    model.add(Dense(128, activation='relu'))  # Gizli katman
    model.add(Dropout(0.5))  # Dropout katmanı
    model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı (binary sınıflandırma)

    # Modeli derleyelim
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Modeli oluşturma
input_dim = len(tokenizer.word_index) + 1  # Tokenizer'ın oluşturduğu kelime sözlüğü büyüklüğü
mlp_model = create_mlp_model(input_dim)

# Modeli eğitme
mlp_model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Test verisi üzerinde tahmin yapma
y_pred_mlp = (mlp_model.predict(X_test_pad) > 0.5).astype('int32')

# Sonuçları değerlendirme
print(f"MLP Model Accuracy on Test Data: {accuracy_score(y_test, y_pred_mlp)}")
print(f"MLP Model Classification Report on Test Data:")
print(classification_report(y_test, y_pred_mlp))







