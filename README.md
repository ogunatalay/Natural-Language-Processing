**MAKİNE ÖĞRENMESİ YÖNTEMLERİ İLE TÜRKÇE TWEETLERDEN SALDIRGAN DİL TESPİTİ** <br>
Bu çalışma, Türkçe tweetlerde yer alan nefret söylemi (hate speech) tespitini amaçlayan bir Doğal Dil İşleme (NLP) projesidir. Günümüzde sosyal medya platformları, bireylerin düşüncelerini özgürce ifade edebildiği alanlar olmakla birlikte, nefret söylemleri gibi zararlı içeriklerin de sıkça yayıldığı ortamlar haline gelmiştir. Bu nedenle, nefret söylemlerini otomatik olarak tespit edebilen bir sistem geliştirilmesi kritik bir ihtiyaçtır.

Bu projede, sosyal medyada karşılaşılan nefret söylemlerinin otomatik olarak "Saldırgan" (Offensive) ve "Saldırgan Olmayan" (Not Offensive) şeklinde sınıflandırılması hedeflenmiştir.

📌 **PROJE İÇERİĞİ** <br>
- NLTK ve Zemberek NLP kütüphaneleri kullanılarak metinler işlenmiştir.<br>
- Stopword temizleme, kullanıcı adı temizleme, özel karakter ve URL kaldırma işlemleri uygulanmıştır.<br>
- Hakaret kelimeleri ağırlıklandırılarak, modelin saldırgan metinleri daha iyi tespit etmesi sağlanmıştır.<br>
- Metinler TF-IDF ve Word Embedding yöntemleri ile sayısal forma dönüştürülmüştür.<br>
- SVD ile veri indirgeme işlemi yapılarak modellerin hesaplama maliyetleri azaltılmıştır. 
- Klasik Makine Öğrenmesi, Derin Öğrenme ve Transfer Öğrenme yöntemleri kullanılmıştır.<br>
- Çalışma kapsamında kullanılan modellerin performansı karşılaştırılmış ve en iyi performansı gösteren model son aşamada oluşturulan "Tahmin Modeli"nde kullanılmıştır.

📂 **VERİ SETİ HAKKINDA**<br>
Bu çalışmada OffensEval 2020 veri seti kullanılmıştır. Bu veri seti sosyal medya üzerindeki saldırgan dil kullanımını tanımlamak amacıyla Çağrı ÇÖLTEKİN tarafından geliştirilmiştir. Özellikle Türkçe dilindeki sosyal medya paylaşımlarını (özellikle Twitter) içeren bu veri seti, metinlerin saldırgan (offensive) ya da saldırgan olmayan (non-offensive) olarak etiketlendiği bir sınıflandırma problemine odaklanmaktadır. Offensive metinler “1”, not-offensive metinler “0” olmak üzere nümerik değere çevrilmiştir. Ayrıca bu veri seti, sosyal medya platformlarındaki dilin analiz edilmesi, nefret söylemi, trolling, hakaret ve aşağılama gibi olguların tanımlanmasında da kullanılabilir.Veri seti, JSONLines formatında ve UTF-8 kodlamasıyla paylaşılmaktadır. Her bir örnek (tweet) şu formatta bir veri içerir:<br>
{ "text" : "buralara değil yaz günü, kışın bile kar yağmıyor", "label" : "not-offensive" }<br>
- Text: Tweetin metni.<br>
- Label: Tweetin saldırgan olup olmadığını belirten etiket. Bu etiket iki sınıf içerir.<br>
•	Offensive: Saldırgan içerik.<br>
•	Not-offensive: Saldırgan olmayan içerik.<br>

**Veri Setinin İstatistikleri**<br>
•	Toplam Tweet Sayısı: 28,000 <br>
•	Offensive Tweet Sayısı: 5,407 <br>
•	Not-offensive Tweet Sayısı: 22,593 <br>

🧬 **KULLANILAN YÖNTEMLER VE MODELLER**<br>

**I .Veri Ön İşleme Süreci**<br>

Proje kapsamında, metinleri analiz edebilmek için farklı zamanlarda NLTK ve Zemberek kütüphanelerinden yararlanılmıştır.
- İlk aşamada, İngilizce için yaygın kullanılan NLTK (Natural Language Toolkit) kütüphanesi kullanılmıştır.<br>
- İkinci aşamada, Türkçeye özel Zemberek NLP kütüphanesine geçilmiş ve metinler bu araç ile işlenmiştir.<br>
  
Metinleri daha anlamlı hale getirebilmek için aşağıdaki işlemler gerçekleştirilmiştir:<br>
- Tokenizasyon: Metinler kelime bazında parçalara ayrılmıştır.<br>
- Köklerine indirgeme (Lemmatization/Stemming): Kelimelerin kök formları çıkarılmıştır.<br>
- Stopword temizleme: Türkçe'de sık kullanılan ancak anlam taşımayan kelimeler kaldırılmıştır.<br>
- Özel karakter ve URL temizliği: Metinlerde yer alan noktalama işaretleri, emojiler, bağlantılar (URLs) silinmiştir.<br>
- Hakaret kelimelerinin ağırlıklandırılması: Küfür veya hakaret içeren kelimelere ekstra ağırlık verilerek, sistemin saldırgan metinleri daha iyi ayırt etmesi sağlanmıştır. (Ağırlıklandırılma için kullanılan
  hakaret kelimeleri TDK'ın sayfasından ve çeşitli hukuk sayfaları referans alınarak belirlenmiştir.)<br>

**II. Vektörleştirme Yöntemleri**<br>

Metinleri sayısal veriye dönüştürmek için iki farklı vektörleştirme yöntemi kullanılmıştır:<br>
**TF-IDF (Term Frequency - Inverse Document Frequency):** İlk aşamada metinleri sayısal formata çevirmek için TF-IDF yöntemi kullanılmıştır. Bu yöntem, kelimelerin hem belge içindeki hem de tüm belgeler arasındaki önemini hesaplayarak istatistiksel bir ağırlıklandırma yapar.<br>
**Word Embedding (Kelime Gömme):** İkinci aşamada, kelimelerin anlam ilişkilerini daha iyi yakalamak için word embedding yöntemleri uygulanmıştır. Burada, BERT modelinden elde edilen kelime vektörleri kullanılmıştır.<br>


**III. Modelleme Aşaması** <br>
Çalışmada, üç farklı yöntem kullanılmıştır:<br>

**Geleneksel Makine Öğrenmesi Modelleri**<br>
Bu aşamada klasik makine öğrenmesi yöntemleri kullanılarak temel sınıflandırma yapılmıştır.<br>
- Logistic Regression (LR)
- Decision Tree (DT)
- Naive Bayes (NB)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Support Vector Machines (SVM) 

**Derin Öğrenme Modelleri** <br>
Klasik yöntemlerden sonra, sinir ağları kullanılarak daha gelişmiş modeller eğitilmiştir.<br>
- MLP (Multi-Layer Perceptron)
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- CNN (Convolutional Neural Network)

LSTM ve BiLSTM, metin içindeki uzun vadeli bağımlılıkları öğrenmede başarılı olmuştur.<br>
GRU, LSTM ile benzer bir performans gösterirken daha hızlı çalışmıştır.<br>
CNN, metin içindeki yerel örüntüleri yakalayarak sınıflandırmada etkili olmuştur.<br>

**Transfer Öğrenme Modelleri**<br>
Son aşamada, büyük dil modellerinden faydalanılarak transfer öğrenme yöntemi uygulanmıştır.<br>
Bu aşamada, DistilBERT modeli kullanılmıştır.<br>

DistilBERT, Türkçe metinler için özelleştirilmiş BERTurk modeline adapte edilmiştir.<br>
BERT modelinden çıkarılan word embedding’ler ile kelimelerin anlam ilişkileri daha iyi öğrenilmiştir.<br>
Transfer öğrenme, önceden büyük veri kümeleri üzerinde eğitilmiş modelleri kullanarak çok daha yüksek doğruluk oranı sağlamıştır.<br>


**📝SONUÇLAR VE ÇIKARIMLAR**<br>
**Performans Karşılaştırması:** <br>
Klasik makine öğrenmesi modelleri: %80 - %85 doğruluk oranı<br>
Derin öğrenme modelleri: %85 - %90 doğruluk oranı<br>
Transfer öğrenme modeli (DistilBERT) modeli: %92 doğruluk oranı ile en iyi sonucu vermiştir.<br>

**Hakaret Kelimelerinin Ağırlıklandırılması:** <br>
Ağırlıklandırılma yöntemi sayesinde, model metin içinde hakaret, argo veya küfür gibi kelimeleri tespit ettiğinde "%100 offensive" sonucu döndürmektedir.<br>
Bu sayede, metin içeriğiyle ilgili kesin bir karar mekanizması oluşturulmuştur.<br>

**Sonuçların Kullanımı (Tahmin Modeli):** <br>
Bu tahmin modeli, girilen herhangi bir metnin "Saldırgan" (Offensive) veya "Saldırgan Olmayan" (Not Offensive) olduğunu yüksek doğrulukla belirleyebilmektedir.<br>
Model, sosyal medya platformlarında nefret söylemi tespiti ve otomatik moderasyon süreçlerinde kullanılabilir.<br>

**Genel Değerlendirme:** <br>
Bu çalışma, Türkçe tweetlerde nefret söylemi tespitinin önemli bir sorun olduğunu ve derin öğrenme & transfer öğrenme yöntemlerinin bu alandaki etkinliğini ortaya koymuştur.<br>

Klasik makine öğrenmesi yöntemleri, belirli bir başarı sağlamış olsa da, metinlerin bağlamını tam olarak anlamada eksik kalmıştır.<br>
Derin öğrenme modelleri, daha yüksek başarı oranları sunsa da, en iyi sonucu DistilBERT sağlamıştır.<br>
Hakaret kelimelerinin ağırlıklandırılması yöntemi, sistemin doğruluğunu ve güvenilirliğini artırmıştır.<br>
Sonuç olarak, geliştirilen model, Türkçe metinlerde nefret söylemini tespit etmek için yüksek doğruluklu ve güvenilir bir araç sunmaktadır.<br>

