# SnakeAI - İki Versiyonla Gelişmiş Yapay Zeka Snake Oyunu

## 🎮 Proje Hakkında

Bu proje, yapay sinir ağları ve genetik algoritmalar kullanarak Snake oyununu oynayabilen bir AI sistemidir. **İki farklı versiyonu bulunmaktadır:**

1. **Orijinal Processing Versiyonu** - En yüksek performans ve tam özellikler
2. **Geliştirilmiş Python Versiyonu** - Öğrenme ve geliştirme amaçlı

## 🎯 Hangi Versiyonu Seçmelisiniz?

### 🏆 **Processing (Orijinal) - En İyi Performans İçin**
- **Popülasyon**: 2000 yılan (Python'da 20)
- **AI Başarısı**: 100-200+ skor
- **Save/Load**: ✅ Tam destek
- **Eğitim Süresi**: ~10 dakika (10 nesil)
- **Tüm Özellikler**: ✅ Aktif

### 🐍 **Python (Geliştirilmiş) - Öğrenme İçin**
- **Popülasyon**: 20 yılan
- **AI Başarısı**: 50-70 skor
- **Save/Load**: ❌ Desteklenmiyor
- **Eğitim Süresi**: ~20 dakika (10 nesil)
- **Özelleştirme**: ✅ Çok esnek

**💡 Tavsiye:** İkisini de kurun - Python ile öğrenin, Processing ile en iyi sonuçları alın!

---

## 🚀 Kurulum ve Çalıştırma

### ⭐ Processing Versiyonu (Tavsiye Edilen)

#### Gereksinimler
- [Processing 3.5.4+](https://processing.org/download/)

#### Kurulum
1. Processing'i indirin ve kurun
2. SnakeAI klasörünü Processing IDE'de açın
3. Run tuşuna basın

#### Özellikler
- **2000 yıldan oluşan popülasyon**
- **Save/Load özellikleri**
- **En iyi grafik performansı**
- **Orijinal tüm özellikler**

### 🐍 Python Versiyonu (Geliştirme Amaçlı)

#### Gereksinimler
- Python 3.7+
- Pygame
- NumPy

#### Kurulum
```bash
# Sanal ortam oluşturun
python3 -m venv snake_ai_env

# Sanal ortamı aktive edin
source snake_ai_env/bin/activate  # Linux/Mac
# veya
snake_ai_env\Scripts\activate  # Windows

# Kütüphaneleri yükleyin
pip install pygame numpy

# Çalıştırın
python improved_snake_ai.py
```

---

## 🎮 Processing Versiyonu - Detaylı Kullanım

### 🎯 **Tavsiye Edilen Eğitim Süresi**

#### **Hızlı Test (Başlangıç): 20-30 Nesil**
- **Süre**: ~30-45 dakika
- **Sonuç**: Temel AI davranışı oluşur
- **Skor**: 50-100 arası
- **Amaç**: AI'nın temel öğrenmesini görmek

#### **İyi Sonuçlar (Orta Seviye): 50-100 Nesil**
- **Süre**: ~1.5-3 saat
- **Sonuç**: İyi performanslı AI
- **Skor**: 100-150 arası
- **Amaç**: Kaliteli bir AI modeli oluşturmak

#### **En İyi Sonuçlar (İleri Seviye): 200-500+ Nesil**
- **Süre**: ~6-15 saat
- **Sonuç**: Mükemmel performanslı AI
- **Skor**: 200-500+ arası
- **Amaç**: Optimize edilmiş master AI

### 🎛️ **Processing Kontrolleri**
- **Save Butonu**: En iyi modeli kaydet
- **Load Butonu**: Kaydedilmiş modeli yükle
- **Graph Butonu**: Evolution grafiğini göster
- **+/- Butonları**: Mutasyon oranını ayarla

### 📊 **Processing Arayüzü**
- **Sol Panel**: Neural network görselleştirme
- **Sağ Panel**: Snake oyunu
- **Kontroller**: Üst kısımda butonlar
- **Bilgiler**: Nesil, skor, mutasyon oranı

---

## 🧠 Yapay Zeka Mimarisi (Her İki Versiyon İçin Geçerli)

### Neural Network
- **Giriş Katmanı**: 24 nöron (8 yön × 3 özellik)
- **Gizli Katmanlar**: 2 katman, her birinde 16 nöron
- **Çıkış Katmanı**: 4 nöron (Yukarı, Aşağı, Sol, Sağ)
- **Aktivasyon**: Sigmoid/ReLU

### Vision Sistemi (Görme) - 24 Girdi

**📊 Yapı: 24 = 8 Yön × 3 Özellik**

**🧭 8 Bakış Yönü:**
| No | Yön | Ok | Açıklama |
|----|-----|----|----------|
| 1 | Yukarı | ↑ | Düz yukarı |
| 2 | Yukarı-Sağ | ↗ | Sağ üst çapraz |
| 3 | Sağ | → | Düz sağ |
| 4 | Aşağı-Sağ | ↘ | Sağ alt çapraz |
| 5 | Aşağı | ↓ | Düz aşağı |
| 6 | Aşağı-Sol | ↙ | Sol alt çapraz |
| 7 | Sol | ← | Düz sol |
| 8 | Yukarı-Sol | ↖ | Sol üst çapraz |

**👁️ Her Yön İçin 3 Özellik:**
- **🍎 Food Distance**: Bu yönde yiyeceğe olan uzaklık
- **🐍 Body Distance**: Bu yönde kendi vücuduna olan uzaklık
- **🧱 Wall Distance**: Bu yönde duvara olan uzaklık

**📏 Mesafe Değerleri:**
- **0.000**: Çok yakın (hemen başında)
- **0.050**: Yakın (~20 birim)
- **0.100**: Orta (~10 birim)
- **1.000**: Uzak veya görünmüyor

**🧮 24 Girdinin Tam Sırası:**
| Girdi No | Yön | Food | Body | Wall |
|----------|-----|------|------|------|
| 1-3 | Yukarı | G1 | G2 | G3 |
| 4-6 | Yukarı-Sağ | G4 | G5 | G6 |
| 7-9 | Sağ | G7 | G8 | G9 |
| 10-12 | Aşağı-Sağ | G10 | G11 | G12 |
| 13-15 | Aşağı | G13 | G14 | G15 |
| 16-18 | Aşağı-Sol | G16 | G17 | G18 |
| 19-21 | Sol | G19 | G20 | G21 |
| 22-24 | Yukarı-Sol | G22 | G23 | G24 |

### Genetik Algoritma
- **Processing**: 2000 yılan popülasyonu
- **Python**: 20 yılan popülasyonu
- **Seçim**: Fitness tabanlı seçilim
- **Çaprazlama**: Tek noktalı çaprazlama
- **Mutasyon**: Rastgele ağırlık değişimi

---

## 📈 Performans Karşılaştırması

| Özellik | Processing | Python | Fark |
|--------|------------|---------|------|
| **Popülasyon** | 2000 yılan | 20 yılan | 100x |
| **Eğitim Hızı** | ~1 dk/nesil | ~2 dk/nesil | 2x |
| **AI Başarısı** | 100-200+ | 50-70 | 2-3x |
| **Save/Load** | ✅ Var | ❌ Yok | - |
| **Grafik** | Native | WSL->X11 | 5x+ |

### 🏆 AI Başarısı Zaman Çizelgesi

| Nesil Sayısı | Processing Skor | Python Skor |
|--------------|-----------------|-------------|
| **10** | 50-80 | 20-40 |
| **50** | 100-150 | 40-60 |
| **100** | 150-200 | 50-70 |
| **200** | 200-300 | 60-80 |
| **500** | 300-500+ | 70-90 |

---

## 🐍 Python Versiyonu - Oyun Modları

### 1. Human Control (İnsan Kontrolü)
- **Açıklama**: Klasik Snake oyunu
- **Kontroller**: Ok tuşları, R: reset
- **FPS**: 10 (yavaş ve kontrollü)

### 2. Single AI Control (Tek AI)
- **Açıklama**: Tek bir AI yılanını izleme (EĞİTİLMEMİŞ)
- **Durum**: ❌ Eğitilmemiş - tamamen rastgele ağırlıklar
- **Beklenen Skor**: 0-10 arası (rastgele performans)
- **Özellik**: Hafif heuristic bias (yiyecek arama eğilimi)
- **Kontroller**: R: reset
- **FPS**: 20 (daha hızlı)
- **Not**: 50+ skor görürseniz, bu şans ve iyi random ağırlıklar sayesindedir, eğitim değil!

### 3. Evolution Training (Evrim Eğitimi)
- **Açıklama**: AI popülasyonu eğitimi (GERÇEK EĞİTİM)
- **Durum**: ✅ Gerçek genetik algoritma eğitimi
- **Popülasyon**: 20 yılan
- **Eğitim Süresi**: 50+ nesil önerilen
- **Beklenen Skor**: 50-70 (50 nesil), 70-90 (100+ nesil)
- **Özellik**: En iyi bireyler seçilir, çaprazlanır, mutasyon uygulanır
- **Kontroller**: T: mode değiştir, Space: daha fazla eğitim
- **FPS**: 15 (orta hız)
- **Amaç**: Gerçekten öğrenmiş AI geliştirmek

---

## ⚠️ ÖNEMLİ NOT: Eğitim Durumu

### 🎯 Tek AI Mod (Seçenek 2) Hakkında
- **Eğitilmiş DEĞİLDİR** - Tamamen rastgele ağırlıklar
- 50+ skor görürseniz, bu **şans ve rastlantı** sonucudur
- Heuristic bias sayesinde çok nadiren yüksek skor yapabilir
- **Gerçek öğrenme için Evolution mod (Seçenek 3) gerekir**

### 🎓 Eğitimli AI İçin
1. **Evolution Training (Seçenek 3)** çalıştırın
2. **50+ nesil** eğitin
3. AI öğrenme sürecini izleyin
4. Sonuçları Single AI modunda test edin

---

## 🛠️ Dosya Yapısı

```
SnakeAI/
├── README.md                    # Bu dosya
│
├── Processing (Orijinal)/
│   ├── SnakeAI/                # Processing sketch
│   │   ├── SnakeAI.pde         # Ana program
│   │   ├── Snake.pde           # Yılan sınıfı
│   │   ├── NeuralNet.pde       # Neural network
│   │   ├── Population.pde      # Popülasyon yönetimi
│   │   ├── Matrix.pde          # Matris işlemleri
│   │   ├── Food.pde            # Yiyecek
│   │   ├── Button.pde          # UI butonları
│   │   └── EvolutionGraph.pde  # Evolution grafiği
│   └── LICENSE                 # Lisans dosyası
│
└── Python (Geliştirilmiş)/
    ├── improved_snake_ai.py    # Ana oyun (tavsiye)
    ├── snake_ai_python.py      # Basit Python versiyonu
    ├── test_snake_logic.py     # Test script'i
    ├── debug_ai.py            # Debug versiyonu
    ├── snake_ai_env/          # Sanal ortam
    └── README_PYTHON.md       # Python dokümantasyonu
```

---

## 🔧 Kullanım Stratejileri

### 🎯 **Yeni Başlayanlar İçin**
1. **Python versiyonu ile başla** - Kod anlama
2. **Processing'de 20-30 nesil** - Temel sonuçlar
3. **Modeli kaydet** - Save butonu
4. **Karşılaştır** - İki versiyonu izle

### 🏆 **En İyi Sonuçlar İçin**
1. **Processing'de 100+ nesil** - Kaliteli AI
2. **Mutasyon oranını ayarla** - +/- butonları
3. **En iyi modeli kaydet** - Save butonu
4. **Grafiği izle** - Graph butonu

### 🔬 **Geliştiriciler İçin**
1. **Python kodunu incele** - Mantığı anla
2. **Yeni özellikler ekle** - Esnek Python kodu
3. **Processing'de test et** - Performansı ölç
4. **İki versiyonu karşılaştır**

---

## 📊 Save & Load Özelliği

### ✅ **Processing Versiyonu**
- **Model Kaydetme**: Weights CSV formatında
- **Model Yükleme**: Önceden eğitilmiş modeller
- **Evolution Grafiği**: Nesil ilerlemesi
- **Test Etme**: Farklı durumlar

### ❌ **Python Versiyonu**
- **Durum**: Save/Load özelliği yok
- **Alternatif**: Model manuel kopyalama

### 💾 **Kullanım Önerileri**
```processing
// Processing'de en iyi modeli kaydetmek:
1. 100+ nesil çalıştır
2. Save butonuna tıkla
3. Dosyaya isim ver (örn: "best_model.csv")
4. İleride Load ile yükle
```

---

## 🐛 Hata Ayıklama ve Sorunlar

### Processing Versiyonu
- **Sorun**: Yavaş başlangıç
- **Çözüm**: 2000 yılanlık popülasyon normaldir
- **Tavsiye**: Sabırlı olun, ilk 20-30 nesil yavaştır

### Python Versiyonu
- **Sorun**: WSL grafik sorunları
- **Çözüm**: X server kurulumu
- **Alternatif**: Windows'ta doğrudan çalıştırın

### Genel
- **AI hemen ölür**: Normaldir, eğitim gerekir
- **Düşük skor**: Daha fazla nesil gerekir
- **Kötü performans**: Popülasyon boyutunu kontrol et

---

## 📈 Başarı Metrikleri

### 🏆 **Processing Versiyonu**
- **Başlangıç**: 0-30 skor
- **20 nesil**: 50-100 skor
- **100 nesil**: 150-200 skor
- **500+ nesil**: 300-500+ skor

### 🐍 **Python Versiyonu**
- **Başlangıç**: 0-15 skor
- **10 nesil**: 30-50 skor
- **50 nesil**: 50-70 skor
- **100+ nesil**: 60-90 skor

---

## 🎯 Öneriler

### 🥇 **En İyi Deneyim İçin:**
1. **Processing ile başla** (2000 popülasyon)
2. **100+ nesil çalıştır**
3. **Modeli kaydet**
4. **Sonuçları izle**

### 🔬 **Öğrenme İçin:**
1. **Python kodunu incele**
2. **Parametreleri değiştir**
3. **Debug araçlarını kullan**
4. **Karşılaştırma yap**

### 💡 **Verimli Çalışma:**
1. **Processing:** En iyi sonuçlar için
2. **Python:** Hızlı prototipleme için
3. **İkisi:** Öğrenme + performans

---

## 🔗 Bağlantılar

- **Orijinal Proje**: https://github.com/greerviau/SnakeAI
- **Processing**: https://processing.org/
- **Python Pygame**: https://www.pygame.org/
- **Genetik Algoritmalar**: https://en.wikipedia.org/wiki/Genetic_algorithm

---

## 📄 Lisans

Bu proje açık kaynaklıdır. Orijinal SnakeAI projesinin lisansına tabidir.

---

## 💡 Son Not

**🏆 Processing en iyi performansı sunar ama Python öğrenmek için harikadır. İkisini de kullanarak hem en iyi sonuçları alabilir hem de yapay zekanın nasıl çalıştığını öğrenebilirsiniz!**

**🎯 Başlamak için Processing'i 20-30 nesil çalıştırın ve AI'nın nasıl öğrendiğini izleyin!**