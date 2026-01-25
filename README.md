# 🧠 Mindwave SI

**Mindwave SI** คือโครงการพัฒนา **Synthetic Intelligence (SI)**  
ที่ออกแบบเป็น “สมองเทียมเชิงโครงสร้าง” ไม่ใช่แค่โมเดล ML

ระบบนี้สามารถ:
- คิด (Reason)
- จำ (Memory)
- เรียนรู้ (Learning)
- ประเมินตนเอง (Introspection)
- ตัดสินใจและกระทำ (Decision & Motion)

โดยมี **BrainController** เป็นศูนย์ควบคุม Runtime ทั้งหมด

## 🎯 Design Philosophy

- แยก “ความคิด” ออกจาก “การควบคุม”
- Brain ไม่รู้จักโลกภายนอกโดยตรง
- ทุกการกระทำต้องผ่าน Rule และ Safety
- ระบบต้องสามารถเติบโตจากประสบการณ์

> มนุษย์ออกแบบกรอบ  
> สมองเรียนรู้ภายในกรอบนั้น

## 🔄 Runtime Flow (Concept)
```text
Input (Vision / Sound / Text)
        ↓
    InputAdapter
        ↓
   WorkingMemory
        ↓
        Brain
        ↓
 Review / Confidence
        ↓
 Rules / Safety / Policy
        ↓
     Decision
        ↓
 Motion / Output
        ↓
 Feedback → Memory → Learning
```

## 📁 Project Structure
```text
Mindwave/
├─ Brain/                         # แกนสมอง (Cognition + Learning + Control)
│  ├─ Meta/                       # การรับรู้ตัวตน (Self-awareness)
│  │  ├─ SelfModel.py             # โมเดลตัวตน: ความสามารถ ข้อจำกัด สถานะโดยรวม
│  │  ├─ GoalTracker.py           # ติดตามเป้าหมาย ระยะสั้น/ยาว และความคืบหน้า
│  │  └─ Introspection.py         # มองย้อนการตัดสินใจ วิเคราะห์ตนเอง
│  ├─ Memory/                     # ระบบความจำแบบลำดับชั้น
│  │  ├─ LongTerm/                # ความจำระยะยาว (คงทน)
│  │  │  ├─ KnowledgeStore.py     # ความรู้เชิงโครงสร้าง (facts / concepts)
│  │  │  ├─ ExperienceStore.py    # ประสบการณ์ + ผลลัพธ์ (episodic memory)
│  │  │  └─ WeightArchive.py      # snapshot น้ำหนักสมอง (rollback / compare)
│  │  ├─ MiddleTerm/              # ความจำบริบท (contextual)
│  │  │  ├─ ContextBuffer.py      # บริบทต่อเนื่องของเหตุการณ์
│  │  │  └─ TopicContext.py       # บริบทตามหัวข้อที่กำลังโฟกัส
│  │  ├─ ShortTerm/               # ความจำระยะสั้น (working memory)
│  │  │  ├─ WorkingMemory.py      # สิ่งที่กำลังคิดใน cycle ปัจจุบัน
│  │  │  └─ AttentionMap.py       # แผนที่ความสนใจ (attention weighting)
│  │  ├─ Emotions/                # อารมณ์ = ตัวปรับน้ำหนักการคิด/เรียนรู้
│  │  │  ├─ EmotionState.py       # อารมณ์ปัจจุบัน
│  │  │  ├─ EmotionProfile.py     # บุคลิกทางอารมณ์
│  │  │  ├─ EmotionWeight.py      # emotional bias ต่อ decision
│  │  │  ├─ EmotionHistory.py     # ประวัติอารมณ์
│  │  │  └─ EmotionEncoder.py     # แปลงอารมณ์เป็น vector
│  │  ├─ Topics/                  # การจัดการหัวข้อและการโฟกัส
│  │  │  ├─ TopicState.py         # topic ปัจจุบัน
│  │  │  ├─ TopicProfile.py       # ความถนัด/ความสนใจต่อ topic
│  │  │  ├─ TopicEmbedding.py     # vector representation ของ topic
│  │  │  ├─ TopicWeight.py        # น้ำหนักความสำคัญของ topic
│  │  │  ├─ TopicHistory.py       # ประวัติการเปลี่ยน topic
│  │  │  └─ TopicRouter.py        # ส่งข้อมูลไป module ที่เหมาะสม
│  │  └─ MemoryEngine.py          # ตัวกลางจัดการ memory ทุกระดับ
│  ├─ Neural/                     # ระบบประสาทเทียม
│  │  ├─ Function/                # ฟังก์ชันพื้นฐานของ neural
│  │  │  ├─ Activation.py         # activation functions registry
│  │  │  ├─ LossFunction.py       # loss + gradient
│  │  │  └─ Metrics.py            # ตัวชี้วัด (accuracy, uncertainty)
│  │  ├─ Weights/                 # การจัดการน้ำหนัก
│  │  │  ├─ WeightSet.py          # ชุดน้ำหนักหนึ่งสถานะ
│  │  │  ├─ WeightStore.py        # จัดเก็บ/โหลดน้ำหนัก
│  │  │  ├─ WeightLinker.py       # mapping น้ำหนัก ↔ connection
│  │  │  └─ WeightStats.py        # วิเคราะห์พฤติกรรมน้ำหนัก
│  │  ├─ Learning/                # กลไกการเรียนรู้หลายรูปแบบ
│  │  │  ├─ LearningEngine.py     # ตัวประสาน learner ทั้งหมด
│  │  │  ├─ GradientLearner.py    # backprop ปกติ
│  │  │  ├─ SelfLearner.py        # เรียนรู้จาก introspection
│  │  │  ├─ AdvisorLearner.py     # เรียนจากคำแนะนำภายนอก
│  │  │  ├─ EvolutionLearner.py   # ปรับโครงสร้างสมอง
│  │  │  └─ ReplayLearner.py      # เรียนจาก replay memory
│  │  ├─ Brain.py                 # โครงสร้างสมอง neural (ไม่รู้โลกภายนอก)
│  │  └─ BrainController.py       # ศูนย์ควบคุม Runtime (หัวใจของระบบ)
│  ├─ Review/                     # การประเมินผล
│  │  ├─ ApproveEngine.py         # ตัดสินว่าผลลัพธ์ผ่านหรือไม่
│  │  ├─ ConfidenceScorer.py      # ประเมินความมั่นใจของ output
│  │  └─ PerformanceMonitor.py    # ติดตาม performance ระยะยาว
│  └─ Rules/                      # กฎ นโยบาย และความปลอดภัย
│     ├─ RuleEngine.py            # ตัว dispatch กฎทั้งหมด
│     ├─ Adaption/                # กฎปรับตัว/โครงสร้าง
│     ├─ Learning/                # กฎการเรียนรู้
│     ├─ Memory/                  # กฎการบันทึก/ลืม
│     ├─ Routing/                 # กฎการส่งข้อมูล
│     ├─ Safety/                  # ความปลอดภัย (โดยเฉพาะ motion)
│     └─ System/                  # runtime & system policy
├─ IO/                            # อินพุต / เอาต์พุต
│  ├─ Reader/                     # อ่านไฟล์ทุกชนิด
│  │    ├─ ExcelReader.py
│  │    ├─ PDFReader.py
│  │    ├─ TextReader.py
│  │    └─ WordReader.py
│  ├─ Sensors/                    # ประสาทสัมผัส (vision / sound)
│  ├─ Actuators/                  # การกระทำ (motion / output)
│  ├─ Internet/                   # เรียนรู้จากเว็บ
│  │   ├─ WebLearner.py           # executor (ดึงจริง)
│  │   ├─ SourceProfile.py        # ความน่าเชื่อถือของแหล่ง
│  │   ├─ ContentParser.py        # html / text → knowledge
│  │   ├─ FactExtractor.py        # fact / concept / relation
│  │   ├─ InternetGateway.py      # interface กับ BrainController
│  └─ InputAdapter.py             # แปลง input → format กลาง
├─ Documentation/                 # เอกสารออกแบบ
├─ MainCore.py                    # entry point ของระบบ
├─ README.md                      # ภาพรวมโครงการ
└─ __init__.py
```
## 🧠 BrainController

BrainController คือหัวใจของระบบ  
ทำหน้าที่เป็น **Runtime Orchestrator**

### Responsibilities
- ควบคุม lifecycle ของ SI
- ประสาน Brain, Memory, Rules, IO
- ตรวจสอบ Safety ก่อนทุก action
- ตัดสินใจว่า “คิด / เรียนรู้ / กระทำ” เมื่อใด

### Conceptual API
- initialize()
- tick()
- receive_input()
- think()
- evaluate_rules()
- decide()
- act()
- learn()
- monitor()
- shutdown()

## 👁️ Vision / 🔊 Sound / 🦾 Motion

### Vision Pipeline
Camera → Preprocess → Encode → VisionBuffer → WorkingMemory

### Sound Pipeline
Microphone → Preprocess → Encode → SoundBuffer → WorkingMemory

### Motion Pipeline
Decision → MotionIntent → Planner → MotorController → Feedback

## 📜 Rule & Safety System

- Rule จัดเก็บในรูปแบบ JSON
- Brain ไม่สามารถแก้ Rule ได้
- ครอบคลุม:
  - Safety
  - Learning
  - Memory
  - Routing
  - Runtime Policy

ทุกการกระทำต้องผ่าน Rule Engine ก่อนเสมอ
