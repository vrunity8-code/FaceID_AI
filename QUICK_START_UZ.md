# FaceID 3D Alignment - Tezkor Qo'llanma

## Ishga Tushirish

```bash
cd d:\FaceID_test
py main.py
```

## Klaviatura Tugmalari

| Tugma | Funksiya |
|-------|----------|
| `q` | Dasturdan chiqish |
| `r` | Yangi yuzni ro'yxatga olish |
| `t` | 3D mesh (to'r) ni yoqish/o'chirish |
| `p` | 3D pose o'qlari (X, Y, Z) ni yoqish/o'chirish |
| `l` | 3D landmarks (nuqtalar) ni yoqish/o'chirish |

## Xususiyatlar

### ✅ Implementatsiya Qilingan

1. **3D Face Mesh Reconstruction** - Yuzning 3D modeli
2. **Head Pose Estimation** - Bosh harakati (pitch, yaw, roll)
3. **68 3D Landmarks** - 68 ta 3D yuz nuqtalari
4. **Real-time Visualization** - Jonli ko'rsatish
5. **Interactive Controls** - Interaktiv boshqaruv

### 📊 Vizualizatsiya

- **Yashil wireframe**: 3D yuz mesh (tugma: `t`)
- **RGB o'qlar**: Bosh pozitsiyasi (tugma: `p`)
  - Qizil: X-o'q (chap-o'ng)
  - Yashil: Y-o'q (yuqori-past)
  - Ko'k: Z-o'q (oldinga-orqaga)
- **Rangli nuqtalar**: 68 ta yuz nuqtasi (tugma: `l`)

### 📐 Pose Ma'lumotlari

Ekranning yuqori chap burchagida:
- **Pitch**: Boshni yuqoriga/pastga egish
- **Yaw**: Yuzni chap/o'ngga burish
- **Roll**: Boshni yon tomonga egish

## Fayl Strukturasi

```
FaceID_test/
├── main.py                     # Asosiy dastur (3D alignment qo'shilgan)
├── modules/
│   ├── detector.py            # Yuz aniqlash
│   ├── recognizer.py          # Yuz tanish
│   ├── alignment_3d.py        # ✨ YANGI: 3D alignment
│   ├── utils_3d.py            # ✨ YANGI: 3D vizualizatsiya
│   ├── tracker.py             # Multi-yuz tracking
│   ├── utils.py               # Umumiy utilities
│   └── liveness/
│       ├── fas.py             # Anti-spoofing
│       └── ppg.py             # rPPG liveness
├── requirements.txt           # Dependencies (yangilangan)
└── README.md                  # To'liq dokumentatsiya

```

## Test Qilish

```bash
# 3D alignment test
py test_3d_alignment.py

# To'liq sistema test
py main.py
```

## Texnik Ma'lumotlar

- **3D Model**: Simplified Basel Face Model (BFM)
- **Landmarks**: 68 nuqta (yuz konturu, qosh, ko'z, burun, og'iz)
- **Pose Estimation**: OpenCV PnP algorithm
- **Performance**: ~5ms qo'shimcha latency har bir yuz uchun
- **FPS**: 30+ (bir yuz uchun)

## Dependencies

Barcha kerakli kutubxonalar o'rnatilgan:
- ✅ insightface (RetinaFace + ArcFace)
- ✅ opencv-python
- ✅ numpy
- ✅ scipy
- ✅ scikit-image (3D rendering uchun)
- ✅ PyYAML

## Muammolarni Hal Qilish

### Agar kamera ochilmasa:
```bash
# Boshqa kamera index sinab ko'ring
cap = cv2.VideoCapture(1)  # main.py da
```

### Agar 3D mesh ko'rinmasa:
- `t` tugmasini bosing (toggle mesh)
- Yuz aniqlanganligini tekshiring
- Yoritish yaxshi ekanligini tekshiring

### Agar pose burchaklar noto'g'ri bo'lsa:
- Kamera yuz qarorda bo'lsin
- Yuz to'liq ko'rinsin
- 45° dan ortiq burilmang

## Qo'shimcha Ma'lumot

To'liq dokumentatsiya: `README.md`
Walkthrough: Artifacts papkasida
