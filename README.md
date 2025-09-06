# 🩺 Chest Cancer Detection with Deep Learning (ResNet50, InceptionV3, VGG16, MobileNet)

## ! Modellerin sonuçları ile ilgili detaylar **[ChestCancer.pptx]** dosyasında yer almaktadır.
## 📌 Proje Özeti
Bu proje, **derin öğrenme tabanlı görüntü sınıflandırma modelleri** kullanılarak göğüs kanseri (chest cancer) tespitini amaçlamaktadır.  
Tıbbi görüntüler üzerinde kanserli ve sağlıklı dokuları ayırt etmek için farklı CNN mimarileri karşılaştırılmıştır:  
- ✅ **ResNet50**  
- ✅ **InceptionV3**  
- ✅ **VGG16**  
- ✅ **MobileNet**  

Amaç, farklı derin ağ mimarilerini karşılaştırarak **doğruluk, hız ve parametre boyutu** açısından en uygun modeli belirlemektir.  

---

## 🧠 Kullanılan Modeller

### 🔹 ResNet50
- Skip connection (artı yol) ile **vanishing gradient** sorununu çözer.  
- Çok katmanlı karmaşık verilerde yüksek doğruluk sağlar.  
- Eğitim sırasında stabil çalışır ve genellikle **overfitting** yapmaz.  

### 🔹 InceptionV3
- Her blokta birden fazla filtre boyutu kullanarak çok yönlü özellik öğrenir.  
- Hem geniş hem derin yapıya sahiptir.  
- Eğitim süresi uzundur ancak doğruluğu yüksektir.  

### 🔹 VGG16
- Basit ve düzenli katman yapısına sahiptir.  
- Tüm katmanlarda **3x3 filtre** kullanır.  
- Performansı iyidir fakat parametre sayısı çok fazladır → yavaş çalışır ve **overfitting** riski vardır.  

### 🔹 MobileNet
- **Mobil ve gömülü sistemler** için tasarlanmış hafif mimaridir.  
- Daha düşük doğruluk elde edebilir, fakat oldukça **hızlı ve verimlidir.**  

---

## 📊 Veri Seti
- Göğüs kanseri tespiti için kullanılan medikal görüntüler (X-ray veya CT taramaları).  
- Veriler **[(https://www.kaggle.com/datasets/yarenbi/chest-cancer-detection-dataset)]** üzerinden temin edilmiştir.  
- Eğitim / Doğrulama / Test oranları: **%70 / %15 / %15**  

---

## ⚙️ Kurulum
Projeyi çalıştırmak için:  

```bash
# 1. Gerekli kütüphaneleri yükleyin
pip install tensorflow keras matplotlib numpy scikit-learn

# 2. Script'leri çalıştırın




