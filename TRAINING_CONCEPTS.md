# 🤖 Machine Learning Training Concepts

Bu doküman, drone detection sisteminin eğitimi sırasında kullanılan temel kavramları açıklar.

---

## 📚 İçindekiler

1. [Epoch Nedir?](#1-epoch-nedir)
2. [Batch Size (Toplu İşlem)](#2-batch-size-topl-işlem)
3. [Iteration (Tekrar)](#3-iteration-tekrar)
4. [Learning Rate (Öğrenme Hızı)](#4-learning-rate-öğrenme-hızı)
5. [Validation vs Testing](#5-validation-vs-testing)
6. [Loss Function (Kayıp Fonksiyonu)](#6-loss-function-kayıp-fonksiyonu)
7. [Overfitting (Aşırı Uyum)](#7-overfitting-aşırı-uyum)
8. [GPU vs CPU](#8-gpu-vs-cpu)
9. [Model Boyutları (n, s, m, l, x)](#9-model-boyutları-n-s-m-l-x)
10. [YOLO Nedir?](#10-yolo-nedir)

---

## 1. Epoch Nedir?

### 📖 Tanım
**Epoch**, modelin tüm eğitim verisini **tam bir kez** gördüğü döngüdür.

### 🔄 Kavramsal Açıklama

Diyelim ki 1000 drone fotoğrafınız var:
```
Dataset: 1000 fotoğraf
├── 800 fotoğraf → Eğitim seti (train)
├── 150 fotoğraf → Doğrulama seti (validation)  
└── 50 fotoğraf  → Test seti (test)
```

- **1 Epoch** = 800 eğitim fotoğrafının tamamı model tarafından bir kez işlenir
- **10 Epoch** = 800 fotoğraf 10 kez işlenir
- **100 Epoch** = 800 fotoğraf 100 kez işlenir

### 🎓 Öğrenme Aşamaları

#### 1 Epoch Sonrası
```
Model: "Fotoğraf 1, 2, 3... bazı pattern'ler gördüm ama tam hatırlayamıyorum"
Durum: İlk aşama
Başarı: Düşük (%30-40)
```

#### 10 Epoch Sonrası
```
Model: "Ah evet, drone'lar genelde küçük, beyaz, dört pervaneli"
Durum: Temel öğrenme
Başarı: Orta (%60-70)
```

#### 100 Epoch Sonrası
```
Model: "Drone'ları artık çok iyi tanıyorum! Küçük detayları bile ayırt edebiliyorum"
Durum: İleri öğrenme
Başarı: Yüksek (%85-95)
```

### ⚖️ Epoch Dengesi

| Epoch Sayısı | Sonuç | Süre |
|-------------|-------|------|
| Çok Az (1-10) | Düşük başarı, yetersiz öğrenme | Kısa |
| Uygun (50-200) | İyi başarı, dengeli öğrenme | Orta |
| Çok Fazla (500+) | Overfitting, ezberleme | Uzun |

### 🎯 Projenizde
- **Epoch Sayısı**: 100
- **Süre**: ~8-10 saat (GPU ile)
- **Hedef**: %80-90 doğruluk oranı

---

## 2. Batch Size (Toplu İşlem Boyutu)

### 📖 Tanım
**Batch Size**, modelin aynı anda işlediği veri örneklerinin sayısıdır.

### 🔄 Kavramsal Açıklama

Diyelim ki 800 fotoğrafınız var ve batch size = 8:

```
Iteration 1: Fotoğraf 1-8    → İşlenir, model güncellenir
Iteration 2: Fotoğraf 9-16   → İşlenir, model güncellenir
Iteration 3: Fotoğraf 17-24  → İşlenir, model güncellenir
...
Iteration 100: Fotoğraf 793-800 → İşlenir, model güncellenir

Toplam: 100 iteration = 1 Epoch
```

### 💡 Batch Size Seçimi

| Batch Size | Avantaj | Dezavantaj |
|-----------|---------|------------|
| Küçük (1-8) | Daha iyi öğrenme, daha fazla bellek | Yavaş eğitim |
| Orta (16-32) | Dengeli, standart kullanım | Ortalama bellek |
| Büyük (64+) | Hızlı eğitim | Overfitting riski, çok bellek |

### 🎯 Projenizde
- **Batch Size**: 8
- **Neden**: GPU bellek sınırlaması
- **Iteration Sayısı**: ~100 iteration / epoch

### 🔢 Formül
```
Iterations per Epoch = Total Training Images / Batch Size
Iterations per Epoch = 800 / 8 = 100
```

---

## 3. Iteration (Tekrar)

### 📖 Tanım
**Iteration**, bir batch'in işlenip modelin güncellendiği tek işlemdir.

### 🔄 Epoch vs Iteration

```
Dataset: 800 fotoğraf
Batch Size: 8

1 Epoch = 100 Iterations

Epoch 1:
  Iteration 1: Fotoğraf 1-8
  Iteration 2: Fotoğraf 9-16
  ...
  Iteration 100: Fotoğraf 793-800

Epoch 2:
  Iteration 101: Fotoğraf 1-8
  Iteration 102: Fotoğraf 9-16
  ...
  Iteration 200: Fotoğraf 793-800
```

### 📊 İlişki
```
1 Epoch = Total Images / Batch Size iterations
1 Epoch = 800 / 8 = 100 iterations
```

---

## 4. Learning Rate (Öğrenme Hızı)

### 📖 Tanım
**Learning Rate**, modelin her adımda ne kadar büyük değişiklik yapacağını belirleyen parametredir.

### 🎓 Açıklama

Model öğrenme sürecini bir dağa tırmanış olarak düşünün:

#### Yüksek Learning Rate (0.1)
```
Dağcı: "Büyük adımlar atacağım!" 
Risk: Tepeden aşırı geçebilir
Sonuç: Kararsız, çok hızlı
```

#### Düşük Learning Rate (0.0001)
```
Dağcı: "Çok küçük adımlar atacağım"
Risk: Çok yavaş, belki hiç ulaşamaz
Sonuç: Çok yavaş öğrenme
```

#### Optimal Learning Rate (0.01)
```
Dağcı: "Dengeli adımlar atacağım"
Sonuç: İyi denge, güzel öğrenme
```

### 📊 Learning Rate Değerleri

| Değer | Açıklama | Kullanım |
|-------|----------|----------|
| 0.001 - 0.0001 | Çok düşük | İnce ayar (fine-tuning) |
| 0.001 - 0.01 | Normal | Standart eğitim |
| 0.01 - 0.1 | Yüksek | Yeni model eğitimi |
| 0.1+ | Çok yüksek | Dengesiz öğrenme |

### 🎯 Projenizde
- **Initial Learning Rate (lr0)**: 0.01
- **Final Learning Rate (lrf)**: 0.01 (lr0 × 0.01 = 0.0001)
- **Strategy**: Sabit veya kademeli azalma

---

## 5. Validation vs Testing

### 📖 Tanımlar

#### Training Set (Eğitim Seti)
- Modelin **öğrendiği** veriler
- Model bu veriler üzerinde güncellenir
- Projede: 800 fotoğraf

#### Validation Set (Doğrulama Seti)
- Modelin **test edildiği** veriler (ama sürekli)
- Her epoch sonunda kontrol edilir
- Overfitting'in erkenden yakalanması için
- Projede: 150 fotoğraf

#### Testing Set (Test Seti)
- Modelin **hiç görmediği** veriler
- Final performans testi
- Tek seferlik kullanım
- Projede: 50 fotoğraf

### 🔄 Süreç

```
Epoch 1:
  1. Training Set'i işle (800 fotoğraf)
  2. Model güncelle
  3. Validation Set'i test et (150 fotoğraf)
  4. Skorları kaydet
  
Epoch 2:
  1. Training Set'i işle
  2. Model güncelle
  3. Validation Set'i test et
  4. Skorları kaydet
  
...
(Sürekli tekrar)

Final:
  → En iyi model Testing Set'te test edilir
  → Bu son sonuçtur!
```

### 📊 Overfitting Tespiti

```
Epoch 50:
  Training Accuracy: %95  ← Model veriyi ezberliyor
  Validation Accuracy: %70  ← Genelleştirme kötü
  Durum: Overfitting! ❌

Epoch 80:
  Training Accuracy: %92  ← İyi öğrenme
  Validation Accuracy: %88  ← Genelleştirme iyi
  Durum: İdeal! ✅
```

---

## 6. Loss Function (Kayıp Fonksiyonu)

### 📖 Tanım
**Loss Function**, modelin ne kadar hatalı olduğunu ölçen fonksiyondur.

### 🎯 Amaç
- Loss (Kayıp) yüksek = Model kötü
- Loss (Kayıp) düşük = Model iyi

### 📊 Loss Değerleri

```
İdeal Eğitim:
Epoch 1:  Loss = 2.5
Epoch 10: Loss = 1.2
Epoch 50: Loss = 0.5
Epoch 100: Loss = 0.2  ✅ Mükemmel!

Kötü Eğitim:
Epoch 1:  Loss = 3.0
Epoch 10: Loss = 2.8
Epoch 50: Loss = 2.5  ⚠️ Hiç öğrenmiyor
```

### 🔧 YOLO'da Loss Bileşenleri

1. **Box Loss**: Bounding box (kutucuk) hataları
2. **Class Loss**: Sınıflandırma hataları
3. **DFL Loss**: Detaylı yerleşim hataları

### 🎯 Hedef
Loss değerini **sürekli azaltmak**!

---

## 7. Overfitting (Aşırı Uyum)

### 📖 Tanım
**Overfitting**, modelin eğitim verisini **ezberlemesi** ve yeni verilerde kötü performans göstermesidir.

### 🎓 Analoji

Öğrenci sınavda sadece ezberlenen sorulara iyi cevap verir:

```
Eğitim: "Bu soruları ezberledim!"
Sınav:   "Yeni sorular! Ne yapacağım?" ❌
```

### 📊 Overfitting Belirtileri

```
Training Metrics:   %95 accuracy  ✅ (Çok iyi görünüyor)
Validation Metrics: %65 accuracy  ❌ (Ama kötü!)

Durum: Overfitting!
Problem: Model veriyi ezberliyor, genelleştiremiyor
```

### 🛠️ Çözümler

1. **Data Augmentation** (Veri Çeşitlendirme)
   - Fotoğrafları döndür, renklendir, kırp
   - Daha fazla çeşitlilik

2. **Early Stopping** (Erken Durdurma)
   - Validation başarı düşmeye başladığında dur
   - Patience: 50 epoch

3. **Regularization** (Düzenlileştirme)
   - Weight decay kullan
   - Dropout ekle

4. **Daha Fazla Veri**
   - 800 → 2000 fotoğraf
   - Daha fazla çeşitlilik

### 🎯 Projenizde
- **Patience**: 50 epoch
- **Augmentation**: Açık (config.yaml)
- **Early Stopping**: Aktif

---

## 8. GPU vs CPU

### 🎨 Görsel Açıklama

#### CPU (Merkezi İşlemci)
```
CPU: 🏠 (Ev gibi düşün)
- 4-16 oda (core)
- Her oda: Oturma odası, mutfak, yatak odası
- Çok yönlü ama yavaş
```

#### GPU (Grafik İşlemci)
```
GPU: 🏭 (Fabrika gibi düşün)
- 1000+ odacık (core)
- Her oda: Aynı işi yapar
- Tek yönlü ama ÇOK HIZLI!
```

### 📊 Performans Karşılaştırması

#### RTX 3060 GPU ile
```
100 Epoch = 8-10 saat ✅
Iteration Süresi = ~1 saniye
Başarı: %85-90
```

#### İşlemci (CPU) ile
```
100 Epoch = 100-200 saat ⏰
Iteration Süresi = ~60 saniye
Başarı: Aynı (ama çok yavaş!)
```

### 🚀 GPU Kullanımı Neden Önemli?

**GPU**: 
- **Paralel işlem**: Binlerce fotoğrafı aynı anda
- **Özel donanım**: AI işlemleri için optimize
- **100-200x daha hızlı**

**CPU**: 
- **Sıralı işlem**: Birer birer
- **Genel donanım**: Her işe uygun ama yavaş

### ⚡ Hız Örneği

```
800 fotoğraf işlemek:

CPU: 60 saniye (1 dakika) per iteration
GPU: 0.6 saniye per iteration

100x daha hızlı! 🚀
```

### 🎯 Projenizde
- **GPU**: Otomatik aktif (CUDA)
- **Beklenen Süre**: 8-10 saat (100 epoch)
- **GPU Doğrulaması**: Eğitim öncesi otomatik

---

## 9. Model Boyutları (n, s, m, l, x)

### 📊 YOLO Model Ölçekleri

| Model | Boyut | Hız | Doğruluk | Kullanım |
|-------|-------|-----|----------|----------|
| **n** (nano) | En küçük | 🚀🚀🚀 Çok hızlı | 📉 Düşük | Mobil, edge cihazlar |
| **s** (small) | Küçük | 🚀🚀 Hızlı | 📊 Orta-düşük | Hızlı uygulamalar |
| **m** (medium) | Orta | 🚀 Dengeli | 📊📊 Orta-yüksek | **ÖNERİLEN** ✅ |
| **l** (large) | Büyük | 🐢 Yavaş | 📈 Yüksek | Yüksek doğruluk |
| **x** (xlarge) | En büyük | 🐢🐢 Çok yavaş | 📈📈 Çok yüksek | Araştırma |

### 🎯 Projenizde
- **Model**: YOLOv11m (medium)
- **Neden**: Hız ve doğruluk dengesi
- **Beklenen**: %85-90 doğruluk, 8-10 saat eğitim

### 📊 Farklı Model Karşılaştırması

```
YOLOv11n (nano):
  Eğitim Süresi: 4-5 saat
  Doğruluk: %75-80
  Dosya Boyutu: 6 MB

YOLOv11m (medium) ✅:
  Eğitim Süresi: 8-10 saat
  Doğruluk: %85-90
  Dosya Boyutu: 21 MB

YOLOv11l (large):
  Eğitim Süresi: 15-20 saat
  Doğruluk: %90-95
  Dosya Boyutu: 46 MB
```

---

## 10. YOLO Nedir?

### 📖 Tanım
**YOLO** = "You Only Look Once" (Sadece Bir Kez Bak)

### 🎯 Konsept

Eski yöntemlerde:
```
1. Fotoğrafta nesne var mı ara
2. Varsa nerede ara
3. Ne olduğunu söyle
4. Hepsini birleştir

→ Yavaş! (birkaç saniye)
```

YOLO'da:
```
→ Fotoğrafa BİR KEZ bak
→ Her şeyi AYNI ANDA tespit et

→ ÇOK HIZLI! (saniyede 60+ frame)
```

### 🚀 YOLOv11 Özellikleri

1. **Hız**: Saniyede 60-200 FPS
2. **Doğruluk**: %85-95 mAP
3. **Gerçek Zamanlı**: Canlı videoda çalışır
4. **Çoklu Nesne**: Birçok drone'u aynı anda tespit eder
5. **Boyut**: Küçük model dosyası (21 MB)

### 🎯 Drone Detection İçin Neden Uygun?

```
✅ Drone'lar küçük nesneler → YOLO küçük nesneleri iyi tespit eder
✅ Gerçek zamanlı → Canlı video için gerekli
✅ Çoklu tespit → Aynı anda birçok drone'u bulabilir
✅ Hızlı → Real-time işlem için ideal
```

---

## 📚 Özet

### Temel Kavramlar

| Kavram | Açıklama | Örnek |
|--------|----------|-------|
| **Epoch** | Tüm veriyi bir kez görme | 100 epoch = veri 100 kez işlendi |
| **Batch Size** | Aynı anda işlenen veri sayısı | 8 fotoğraf / batch |
| **Iteration** | Bir batch'in işlenmesi | 100 iteration / epoch |
| **Learning Rate** | Öğrenme hızı | 0.01 (orta hız) |
| **Loss** | Hata miktarı | 0.2 (düşük = iyi) |
| **Overfitting** | Ezberleme | Training %95, Validation %70 |

### Eğitim Süreci

```
1. Veriyi hazırla (800 fotoğraf)
2. Modeli yükle (YOLOv11m)
3. 100 epoch eğit:
   - Her epoch: 100 iteration
   - Her iteration: 8 fotoğraf işle
   - Her epoch sonu: Validation test
4. En iyi modeli kaydet
5. Test et (%85-90 doğruluk beklenir)
```

### Performans Beklentileri

- **Süre**: 8-10 saat (GPU ile)
- **Doğruluk**: %85-90
- **Model Boyutu**: 21 MB
- **Tespit Hızı**: 60+ FPS
- **Kullanım**: Real-time video, görüntü analizi

---

## 🎓 Öğrendikleriniz

✅ Epoch ne demek  
✅ Batch size'ın önemi  
✅ GPU neden gerekiyor  
✅ Overfitting nedir  
✅ YOLO nasıl çalışır  
✅ Eğitim süreci nasıl ilerler  

**Artık kendi modelinizi eğitebilirsiniz!** 🚀

---

**Sorularınız için**: GitHub Issues veya Discussions  
**Öğrenmeye devam**: [Ultralytics Docs](https://docs.ultralytics.com/)
