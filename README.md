# Çok Katmanlı Algılayıcı (MLP) Sınıflandırma Projesi

## Giriş

Bu proje, banknot doğrulama veri seti üzerinde Çok Katmanlı Algılayıcı (MLP) modellerinin iki farklı mimarisini karşılaştırmaktadır. Temel amaç, bir gizli katmanlı (2-katmanlı) ve iki gizli katmanlı (3-katmanlı) MLP'lerin performanslarını analiz etmek ve bu modellerin hiperparametre optimizasyonunu yaparak en iyi sonucu elde etmektir.

Çalışmada, manuel olarak kodlanmış sıfırdan MLP implementasyonları ile popüler kütüphanelerin (Scikit-learn ve PyTorch) sunduğu MLP sınıflandırıcıları karşılaştırılmıştır. Bu karşılaştırma, model performanslarının yanı sıra eğitim süreleri açısından da gerçekleştirilmiştir.

## Metot

### Veri Seti

Çalışmada kullanılan veri seti, banknot doğrulama veri setidir (BankNote Authentication Dataset). Bu veri seti, banknotların gerçek ve sahte olarak sınıflandırılmasını amaçlamaktadır ve dört özellik içermektedir:
- Variansa dayalı özellik (Variance)
- Çarpıklık (Skewness)
- Basıklık (Kurtosis)
- Entropi (Entropy)

### Model Mimarileri

İki farklı MLP mimarisi uygulanmıştır:

1. **2-katmanlı MLP (1 gizli katman)**: 
   - Giriş katmanı (4 özellik)
   - Bir gizli katman (nöron sayısı deneylerde optimize edilmiştir)
   - Çıkış katmanı (1 nöron, sınıflandırma için)

2. **3-katmanlı MLP (2 gizli katman)**:
   - Giriş katmanı (4 özellik)
   - İlk gizli katman (nöron sayısı deneylerde optimize edilmiştir)
   - İkinci gizli katman (nöron sayısı deneylerde optimize edilmiştir)
   - Çıkış katmanı (1 nöron, sınıflandırma için)

### Uygulanan Teknikler

Modelin performansını artırmak için aşağıdaki teknikler uygulanmıştır:

- **Veri Normalizasyonu**: StandardScaler kullanılarak veriler normalize edilmiştir
- **He başlatma yöntemi**: Ağırlıklar, giriş boyutuna bağlı olarak özel bir başlatma şeması ile başlatılmıştır
- **Mini-batch Stokastik Gradyan İnişi (SGD)**: Eğitim sırasında mini-batch'ler kullanılarak optimizasyon yapılmıştır
- **Momentum**: Gradyan hesaplamalarında momentum terimi eklenmiştir
- **Öğrenme hızı azaltma (Learning Rate Annealing)**: Eğitim sırasında öğrenme hızı kademeli olarak azaltılmıştır
- **Farklı aktivasyon fonksiyonları**: Hem tanh hem de ReLU aktivasyon fonksiyonları test edilmiştir

### Hiperparametre Optimizasyonu

Aşağıdaki hiperparametreler için grid search yapılmıştır:

- Gizli katman nöron sayıları: [3, 5, 8, 10]
- İterasyon sayıları: [100, 300, 500, 1000]
- Aktivasyon fonksiyonları: ["tanh", "relu"]

## Sonuçlar

### Hiperparametre Optimizasyonu Sonuçları

Yapılan kapsamlı deneyler sonucunda, her iki model mimarisi için en iyi hiperparametreler belirlenmiştir. Doğruluk (accuracy) metriği temel performans ölçümü olarak kullanılmıştır.

Elde edilen sonuçlara göre:

- **2-katmanlı MLP için en iyi konfigürasyon**:
  - Gizli katman nöron sayısı: [Optimizasyon sonucu]
  - İterasyon sayısı: [Optimizasyon sonucu]
  - Aktivasyon fonksiyonu: [Optimizasyon sonucu]
  - Test seti doğruluğu: [Elde edilen doğruluk değeri]

- **3-katmanlı MLP için en iyi konfigürasyon**:
  - İlk gizli katman nöron sayısı: [Optimizasyon sonucu]
  - İkinci gizli katman nöron sayısı: [Optimizasyon sonucu]
  - İterasyon sayısı: [Optimizasyon sonucu]
  - Aktivasyon fonksiyonu: [Optimizasyon sonucu]
  - Test seti doğruluğu: [Elde edilen doğruluk değeri]

### Farklı Implementasyonların Karşılaştırması

Kendi yazdığımız MLP implementasyonları, Scikit-learn ve PyTorch kütüphanelerinin MLP sınıflandırıcıları ile karşılaştırılmıştır:

| Model | Doğruluk (Accuracy) | Eğitim Süresi (s) |
|-------|---------------------|-------------------|
| Manuel MLP | [Değer] | [Değer] |
| Scikit-learn MLP | [Değer] | [Değer] |
| PyTorch MLP | [Değer] | [Değer] |

## Tartışma

### Model Karmaşıklığı ve Performans İlişkisi

Bu çalışmada gözlemlenen önemli bulgulardan biri, model karmaşıklığı ve performans arasındaki ilişkidir. 3-katmanlı MLP modelinin 2-katmanlı MLP modeline göre daha iyi performans gösterdiği durumlar olsa da, bu her zaman geçerli değildir. Bazı durumlarda daha basit olan 2-katmanlı model daha iyi sonuçlar verebilmektedir.

Bu durum, gizli katman sayısının artmasının her zaman model performansını artırmadığını göstermektedir. Uygun hiperparametreler ile daha basit bir model, karmaşık bir modele göre daha iyi genelleme yapabilir ve aşırı öğrenme (overfitting) riskini azaltabilir.

### Aktivasyon Fonksiyonlarının Etkisi

ReLU ve tanh aktivasyon fonksiyonlarının performans üzerindeki etkileri incelendiğinde, ReLU genellikle daha hızlı yakınsama sağlarken, tanh bazı durumlarda daha kararlı sonuçlar üretmiştir. ReLU aktivasyonu ile eğitilen modellerin, özellikle He başlatma yöntemi ile kombine edildiğinde daha iyi sonuçlar verdiği gözlemlenmiştir.

### İyileştirme Önerileri

Gelecek çalışmalarda şu iyileştirmeler yapılabilir:

1. Daha kapsamlı hiperparametre optimizasyonu (örn. öğrenme hızı, momentum katsayısı, mini-batch boyutu)
2. Daha gelişmiş optimizasyon algoritmaları (Adam, RMSprop gibi)
3. Dropout gibi düzenlileştirme (regularization) tekniklerinin eklenmesi
4. K-kat çapraz doğrulama (k-fold cross-validation) ile model performansının değerlendirilmesi
5. Farklı veri setleri üzerinde benzer karşılaştırmaların yapılması

## Referanslar

1. Haykin, S. (2009). Neural networks and learning machines (3rd ed.). Pearson.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
4. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12, 2825-2830.
6. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8026-8037).
