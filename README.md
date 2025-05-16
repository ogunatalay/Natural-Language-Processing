# Türkçe Tweetlerden Nefret Söylemi Tespiti — Makine Öğrenmesi ve Derin Öğrenme Yaklaşımı

## Proje Özeti

Bu proje, Türkçe Twitter verileri üzerinde nefret söylemi (saldırgan dil) tespiti yapmak için geliştirilmiş doğal dil işleme (NLP) tabanlı bir sınıflandırma sistemidir. Hem klasik makine öğrenmesi algoritmaları hem de ileri düzey derin öğrenme modelleri (Transformer tabanlı DistilBERT ve XLM-RoBERTa dahil) kullanılarak, Türkçe metinlerdeki nefret söylemi içeren ifadelerin otomatik sınıflandırılması hedeflenmiştir.

---

## İçindekiler

- [Giriş](#giriş)  
- [Veri Seti](#veri-seti)  
- [Veri Ön İşleme](#veri-ön-işleme)  
- [Kullanılan Modeller](#kullanılan-modeller)  
- [Performans Değerlendirme](#performans-değerlendirme)  
- [Açıklanabilir Yapay Zeka (XAI)](#açıklanabilir-yapay-zeka-xai)  
- [Sonuçlar](#sonuçlar)  
- [Gelecekteki Çalışmalar](#gelecekteki-çalışmalar)  
- [İletişim](#iletişim)

---

## Giriş

Sosyal medya platformlarında hızla yayılan nefret söylemi, toplumsal huzuru tehdit etmekte ve dijital güvenliği zayıflatmaktadır. Bu nedenle, zararlı içeriklerin otomatik olarak tespit edilmesi kritik bir öneme sahiptir. Bu proje, Türkçe tweetlerde nefret söylemini yüksek doğrulukla tespit etmek üzere makine öğrenmesi ve derin öğrenme yöntemlerini kullanır.

---

## Veri Seti

- **Kaynak:** OffensEval 2020 Twitter veri seti (Türkçe)
- **Özellikler:**
  - Toplam 31.515 tweet (28.000 eğitim, 3.515 test)
  - İki sınıf: `Offensive (1)` ve `Not-offensive (0)`
  - JSONLines formatında, UTF-8 kodlamalı

---

## Veri Ön İşleme

- Metinler küçük harfe dönüştürülür.
- URL ve kullanıcı adları (@username) temizlenir.
- Özel karakterler ve noktalama işaretleri kaldırılır.
- Türkçe stopwords temizlenir.
- Türkçe ekler köklerine indirgenir (Zemberek NLP kullanılarak).
- Tekrar eden kelimeler tekilleştirilir.
- Hakaret kelimelerine ekstra ağırlık verilir.
- Kelimeler TF-IDF yöntemi ile sayısal vektörlere dönüştürülür.
- Veri boyutu SVD ile indirgenerek model eğitim süresi optimize edilir.

---

## Kullanılan Modeller

### Klasik Makine Öğrenmesi

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

### Derin Öğrenme

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (Bi-LSTM)
- Gated Recurrent Unit (GRU)
- Distilled BERT (DistilBERT) — Türkçe adaptasyonu BERTurk
- XLM-RoBERTa (XLM-R) — Çok dilli Transformer modeli

---

## Performans Değerlendirme

- **Metrikler:** Accuracy, Precision, Recall, F1 Score  
- Derin öğrenme modelleri, özellikle DistilBERT ve XLM-R, klasik yöntemlere göre belirgin üstünlük sağladı.  
- En yüksek doğruluk oranları %85-90 bandında gerçekleşti.

---

## Açıklanabilir Yapay Zeka (XAI)

- **LIME:** Model kararlarının kelime bazında açıklanması.  
- **BertViz:** Transformer attention mekanizmasının görselleştirilmesi.  
- **Kelime Dikkat Projeksiyonu:** Modellerin kelimelere verdiği dikkat ağırlıkları incelendi.

---

## Sonuçlar

- Derin öğrenme tabanlı Transformer modelleri Türkçe nefret söylemi tespitinde en başarılı yöntemler olarak öne çıktı.  
- Model kararlarında hakaret kelimeleri güçlü belirleyici oldu.  
- Veri ön işleme ve özel ağırlıklandırma teknikleri, model performansını artırmada etkili oldu.  
- Proje, Türkçe doğal dil işleme ve sosyal medya analizinde ileri seviye bir referans sağlamaktadır.

---

## Gelecekteki Çalışmalar

- Veri setinin genişletilmesi ve çeşitlendirilmesi  
- Türkçe’ye özgü daha gelişmiş dil modellerinin geliştirilmesi  
- Gerçek zamanlı nefret söylemi tespiti sistemlerinin tasarlanması  
- Model şeffaflığının artırılması için açıklanabilirlik yöntemlerinin derinleştirilmesi

---

## İletişim

Proje hakkında soru ve katkılarınız için:  
**Ogün Atalay**  
Email: ogun.atalay33@gmail.com 
Kahramanmaraş Sütçü İmam Üniversitesi, Bilgisayar Mühendisliği Bölümü
