# 🧠 Mindwave v1.0

**Cognitive AI — ระบบเรียนรู้แบบเรียลไทม์**

ระบบ AI ระดับ Production ที่รองรับ neural evolution, การฝึกสอนขณะทำงาน (runtime training) และระบบความเชื่อแบบความน่าจะเป็น

---

## ✨ คุณสมบัติเด่น

### 🎯 ความสามารถหลัก
- ✅ **Runtime Training** — ฝึกสอนได้ขณะรัน (ไม่ต้องรีสตาร์ท)
- ✅ **Neural Evolution** — พัฒนาโครงสร้างอัตโนมัติตามประสิทธิภาพ
- ✅ **Probabilistic Beliefs** — belief_mean ± variance (ไม่มี True/False แบบไบนารี)
- ✅ **Multi-Epoch Learning** — ฝึกซ้ำอัตโนมัติ (ค่าเริ่มต้น: 3 รอบ)
- ✅ **4-Tier Memory** — short / middle / long / immortal
- ✅ **8 ช่องทาง IO** — CLI, File, Socket, REST, EventBus, Internet, Sound, Video

### 🧬 โครงข่ายประสาทเทียม
- Forward / Backward propagation
- Gradient descent พร้อมอัปเดต weight อัตโนมัติ
- การพัฒนาโครงสร้าง (ADD_NODE / CONNECTION, PRUNE, MUTATE)
- ติดตามการใช้งานและประวัติ evolution

### 📚 ระบบฝึกสอน
- รองรับหลายแหล่งข้อมูล: File, URL, Image, Directory, Wildcard
- ระบบแท็ก: `<fact>`, `<qa>`, `<context>`, `<rule>`, `<ignore>`
- รองรับ plain text (แบ่งข้อความอัตโนมัติ)
- ติดตามความคืบหน้า

---

## 🚀 เริ่มต้นใช้งานอย่างรวดเร็ว

### การติดตั้ง

```bash
# เข้าไปที่โฟลเดอร์ Mindwave
cd MindWave

# ติดตั้งส่วนหลัก (จำเป็น)
pip install numpy

# ติดตั้ง dependency ตามฟีเจอร์ที่ใช้
pip install pdfplumber python-docx        # เอกสาร
pip install fastapi uvicorn               # REST API
pip install SpeechRecognition pyaudio pyttsx3  # เสียง
pip install pytesseract Pillow opencv-python   # Vision

# หรือติดตั้งทั้งหมด
pip install -r requirements.txt
```

### การใช้งานพื้นฐาน

```bash
# เริ่มต้น Mindwave
python3 Main.py

# พร้อมตัวเลือก
python3 Main.py --context ai --verbose --epochs 5
python3 Main.py --rest-port 8000  # เปิด REST API
```

### การฝึกสอน

```python
# ภายใน Mindwave CLI
/train data.txt
/train Datasets/*.txt
/train docs/
/train https://...
/train photo.jpg
```

### คำสั่งหลัก

```
/train <source>   # ฝึกจากแหล่งข้อมูลใดก็ได้
/beliefs          # ดูความเชื่อที่เรียนรู้
/neurostat        # สถิติ neural + evolution
/summary          # ภาพรวมทั้งหมด
/status           # สถานะสมอง
/help             # คำสั่งทั้งหมด
```

---

## 📊 ตัวอย่างการใช้งาน

```python
from Core.BrainController import BrainController
from Core.Train.TrainingPipeline import TrainingPipeline

# สร้าง Brain
brain = BrainController()

# ฝึกสอน
pipeline = TrainingPipeline(brain)
result = pipeline.train("data.txt", epochs=3)

# ถามคำถาม
result = brain.respond("AI คืออะไร", context="general")
print(result["response"])
```

---

## 🎓 รูปแบบข้อมูลฝึกสอน

### แท็กเป็นทางเลือก (Plain text ก็ใช้ได้)

```
<qa>
Q: AI คืออะไร?
A: AI คือระบบที่เรียนรู้และแก้ปัญหาได้
</qa>

<fact>Neural network คือโครงข่ายประสาทเทียม</fact>

<context:math>
สมการ ax²+bx+c=0
</context:math>

<rule>ตอบเป็นภาษาไทยเป็นหลัก</rule>

<ignore>ข้ามเนื้อหานี้</ignore>

# ข้อความธรรมดาไม่มีแท็กก็ใช้ได้
Neural networks are inspired by biological neurons...
```

---

## 🧪 การทดสอบ

```bash
# รัน integration test ทั้งหมด
python3 Test/Integration/test_full_integration.py
```

ผลลัพธ์ที่คาดหวัง:

```
✅ All Integration Tests Passed!
Summary:
  • Brain: xxx
  • Beliefs: 17 (0 stable, 7 strong)
  • Memory: 2 atoms
  • Neural: 20 nodes, 36 connections
```

---

## 📚 เอกสารประกอบ

| เอกสาร | คำอธิบาย |
|----------|-------------|
| ARCHITECTURE.md | โครงสร้างระบบและ data flow |
| MODULES.md | รายละเอียดโมดูลทั้งหมด |
| API.md | เอกสาร API |
| TRAINING.md | คู่มือการฝึกพร้อมตัวอย่าง |
| IO_CHANNELS.md | เอกสารระบบ IO |
| CONTRIBUTING.md | แนวทางการพัฒนา |
| CHANGELOG.md | ประวัติเวอร์ชัน |

---

## ⚙️ การตั้งค่า

แก้ไขไฟล์ `config.yaml`:

```yaml
brain:
  context: "general"
  personality: "Professional"

neural:
  learning_rate: 0.01
  enable_evolution: true
  evolve_every: 50

training:
  epochs: 3
```

---

## 🔧 REST API

เริ่มเซิร์ฟเวอร์:

```bash
python3 Main.py --rest-port 8000
```

ตัวอย่าง endpoint:

```bash
# Respond
curl -X POST http://localhost:8000/respond \
  -H "Content-Type: application/json" \
  -d '{"text":"AI คืออะไร","context":"general"}'

# Learn
curl -X POST http://localhost:8000/learn \
  -d '{"text":"Neural network คือโครงข่ายประสาทเทียม"}'

# Status
curl http://localhost:8000/status

# Health
curl http://localhost:8000/health
```

---

## 📦 โครงสร้างโปรเจกต์

```
MindWave/
├── Main.py
├── config.yaml
├── requirements.txt
├── CHANGELOG.md
│
├── Core/
│   ├── BrainController.py
│   ├── Brain/
│   ├── IO/
│   ├── Train/
│   ├── Memory/
│   └── Neural/
│
├── Test/
│   └── Integration/
│
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── MODULES.md
│   ├── API.md
│   ├── TRAINING.md
│   ├── IO_CHANNELS.md
│   └── CONTRIBUTING.md
│
└── examples/
    ├── basic_usage.py
    └── training_example.py
```

---

## 🎯 ประสิทธิภาพ

- **Training**: 8 หน่วย × 3 รอบ < 1 วินาที
- **Response**: ~10–50ms (ไม่รวม IO)
- **Memory**: ~50–100MB โดยทั่วไป
- **Evolution**: ทำงานอัตโนมัติทุก 50 ตัวอย่าง

---

## 🐛 ปัญหาที่ทราบ

### Threading
- BeliefSystem, MemoryController, LearnMode ยังไม่ thread-safe
- ควรใช้ lock เมื่อเข้าถึงพร้อมกัน

### Evolution
- ต้องมีข้อมูลฝึก 50+ ตัวอย่างถึงจะทำงาน
- อาจไม่พัฒนา หาก loss ต่ำมากอยู่แล้ว

---

## 🚧 แผนพัฒนา

**v1.1 (วางแผน):**
- ตัวโหลด config แบบยืดหยุ่น
- เครื่องมือวัดประสิทธิภาพ
- Web UI dashboard

**v1.2 (วางแผน):**
- Multi-modal learning
- Transfer learning
- กลยุทธ์ evolution ขั้นสูง

---

## 📄 License

[ระบุ License]

---

## 🙏 ขอบคุณ

พัฒนาด้วย:
- Python 3.10+
- NumPy (neural operations)
- FastAPI (REST API)
- python-docx, pdfplumber (ประมวลผลเอกสาร)

---

## 💬 การสนับสนุน

- 📧 [ข้อมูลติดต่อ]
- 📝 [Issue tracker]
- 📚 เอกสารเพิ่มเติม (docs/)

---

**Made with 🧠 by Pakawat Tanyaphirom**

*Version 1.0 — February 2026*
