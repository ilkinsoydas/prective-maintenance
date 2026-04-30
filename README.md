# 🏭 Industrial Predictive Maintenance: An End-to-End ML Pipeline

Bu proje, endüstriyel tesislerdeki beklenmedik arıza duruşlarını (downtime) minimize etmek ve bakım maliyetlerini optimize etmek amacıyla geliştirilmiş, uçtan uca bir makine öğrenmesi çözümüdür. Trakya Üniversitesi Bilgisayar Mühendisliği kapsamında yürütülen bu çalışma, özellikle dengesiz veri setlerinde model performansının ve yorumlanabilirliğinin artırılmasına odaklanmaktadır.

## 🛠 Metodoloji ve Teknik Uygulama

### 1. Veri Madenciliği ve Özellik Mühendisliği (Feature Engineering)
* **Veri Seti Analizi:** 10,000'den fazla sensör verisi üzerinde keşifçi veri analizi (EDA) yapılarak, arıza paternleri incelenmiştir.
* **Zaman Serisi Dekompozisyonu:** `date` özniteliği; `day`, `month` ve `day_of_week` bileşenlerine ayrılarak modelin operasyonel döngüleri ve mevsimsel etkileri yakalaması sağlanmıştır.
* **Kategorik Veri Kodlama:** `device` kimlikleri, `LabelEncoder` metodolojisi ile modelin işleyebileceği nümerik vektörlere dönüştürülmüştür.

### 2. Stratejik Veri Bölümleme
* **Eğitim/Test Ayrımı:** Modelin genelleme kapasitesini (generalization) doğrulamak adına veri seti %88 eğitim ve %12 test olarak ayrılmıştır.
* **Stratified Sampling:** Sınıf dağılımının nadirliği göz önüne alınarak, her iki alt kümede de arıza oranının orijinal veri setiyle tutarlı kalması sağlanmıştır.

### 3. Sınıf Dengesizliği ile Mücadele (SMOTE)
* **Dengesizlik Analizi:** Arıza vakalarının (Minority Class) düşük frekansı nedeniyle modelin majör sınıfa doğru yanlılık (bias) gösterme riski SMOTE ile minimize edilmiştir.
* **SMOTE Uygulaması:** Sentetik örnekler üretilerek eğitim seti dengelenmiş ve modelin arıza tespit hassasiyeti artırılmıştır.

### 4. Hiperparametre Optimizasyonu ve Model Seçimi
* **Algoritma:** Yüksek performanslı gradyan artırma algoritması olan **XGBoost Classifier** kullanılmıştır.
* **GridSearchCV:** Optimum performansa ulaşmak için `max_depth`, `learning_rate`, `n_estimators`, `gamma` ve `subsample` parametreleri üzerinde 3-katlı çapraz doğrulama ile kapsamlı bir tarama gerçekleştirilmiştir.

### 5. Karar Eşiği (Threshold) Optimizasyonu
* **Maliyet Analizi:** Endüstriyel maliyet analizleri, "kaçırılan bir arızanın" (False Negative) çok daha maliyetli olduğunu gösterdiği için standart 0.50 eşiği yerine **0.30** eşiği tercih edilmiştir.
* **Sonuç:** Recall (Duyarlılık) değeri maksimize edilerek operasyonel risk düşürülmüştür.

### 6. Model Şeffaflığı ve XAI (Explainable AI)
* **Önem Analizi:** `feature_importances_` özniteliği üzerinden modelin hangi sensör verilerine (sıcaklık, titreşim vb.) dayanarak karar verdiği görselleştirilmiştir.
* **Uygulanabilirlik:** Bu adım, bakım ekiplerine "nedensellik" sunarak projenin operasyonel uygulanabilirliğini güçlendirmektedir.

### 7. Model Serializasyonu ve Deployment
* **Serializasyon:** Eğitilen en iyi model, **Joblib** kütüphanesi aracılığıyla `predictive_model.pkl` formatında serialize edilmiştir.
* **Canlı Entegrasyon:** Bu yapı, modelin eğitim sürecine ihtiyaç duymadan gerçek zamanlı veri akışlarına entegre edilmesine olanak tanır.

---
**Geliştiren:** İlkin Soydaş - Trakya Üniversitesi Bilgisayar Mühendisliği

### Oluşan Grafik

<img width="1973" height="1192" alt="image" src="https://github.com/user-attachments/assets/35aed0aa-b8ce-46bf-902e-afad00564931" />





