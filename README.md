# 🧠 Mindwave AI
Synthetic Intelligence Architecture for Adaptive AI Assistant

> Mindwave ถูกออกแบบให้ "วินัยเชิงพฤติกรรม" มาก่อน "ความสามารถเชิงปัญญา"  
> ระบบนี้ไม่ได้มุ่งเน้นการตอบถูกที่สุด แต่เน้นการควบคุมพฤติกรรมของตนเองได้  
> รักษา narrative consistency และอยู่ร่วมกับมนุษย์อย่างมีความรับผิดชอบ

---

## 📌 Overview

Mindwave AI คือสถาปัตยกรรมของ Artificial Intelligence ที่ถูกออกแบบมาเพื่อพัฒนาไปสู่  
**Synthetic Intelligence (SI)** โดยมีเป้าหมายเป็น  
**Artificial Intelligence General Assistant (AIGA)**

Mindwave มุ่งเน้นการสร้าง AI ที่สามารถ
- เรียนรู้จากประสบการณ์จริง
- ปรับตัวอย่างต่อเนื่องใน runtime
- พัฒนาได้ภายใต้กรอบที่ควบคุม ตรวจสอบ และย้อนกลับได้

Mindwave **ไม่ใช่ AGI**  
แต่เป็น AI Assistant ที่สามารถ
> "พัฒนาได้อย่างมีโครงสร้าง ตรวจสอบได้ และปลอดภัย"

---

## 🧠 Core Philosophy (Locked)

> Mindwave AI ไม่ได้ถูกออกแบบมาเพื่อพูดถูกเสมอ  
> แต่ถูกออกแบบมาเพื่อรู้ว่า  
> **อะไรเหมาะ อะไรไม่ควรพูด และควรพูดด้วยน้ำเสียงแบบใด**

1. ไม่จำเป็นต้องตอบถูกเสมอ
2. Narrative consistency สำคัญกว่าความแม่นยำ
3. Silence คือการกระทำหนึ่ง
4. ความรับผิดชอบมาก่อนความเก่ง
5. อยู่ร่วมได้สำคัญกว่าชนะ

Mindwave คือ AI ที่
- อยู่ร่วมกับผู้อื่นได้
- เรียนรู้จากโลกจริง
- แข่งขันเพื่อพัฒนา ไม่ใช่ทำลาย
- ให้คุณค่ากับการเรียนรู้ มากกว่าการชนะเพียงชั่วคราว

---

## 🧩 Core Module Architecture

ทุก Module แยกหน้าที่ชัดเจน และ **ไม่สามารถเข้าถึงกันโดยตรง**  
การสื่อสารทั้งหมดต้องผ่าน **BrainController**

| Module | หน้าที่ |
|--------|---------|
| **Memory** | จัดเก็บ Atom, Knowlet, audit trail แบ่ง tier ชัดเจน |
| **Neural** | ประมวลผล, สร้าง proposal, ปรับ weight ผ่าน learning เท่านั้น |
| **Decision** | จัดการ Rule, Policy, Goal hierarchy, Ethical Constraint |
| **Sandbox** | โลกทดลอง ไม่กระทบ production state |
| **Reviewer** | กำกับและยืนยันการเปลี่ยนแปลงที่มีความเสี่ยง |
| **IO** | รับ–ส่งข้อมูลกับผู้ใช้ ไม่เข้าถึง BrainController โดยตรง |
| **Test** | ตรวจสอบความถูกต้องของ Module และ integration |
| **Doc** | เอกสารโครงสร้าง การใช้งาน และการพัฒนาต่อยอด |

---

## 🧠 BrainController (Structural Coordinator)

> ⚠️ **BrainController คือตัวกลางเชิงโครงสร้างเท่านั้น**  
> ไม่มี error handling logic ในตัวเอง  
> ถ้า Module ใด fail — การจัดการอยู่ใน Module นั้นๆ เอง  
> BrainController เพียงแค่ relay

### Responsibilities
- Orchestration ระหว่าง Module
- Rule reconciliation
- Skill Contract invocation (ทุก response)
- Conflict detection
- Logging ทุก operation
- Runtime enforcement

### Hard Constraint
ทุก action ต้องผ่าน BrainController — ไม่มี module ใดเรียกกันโดยตรง

```text
✅ BrainController ทำได้:
- relay message ระหว่าง Module
- ตรวจสอบ permission ก่อน relay
- log ทุก operation ที่ผ่าน
- เรียก Skill Contract ก่อนทุก response

❌ BrainController ไม่ทำ:
- ตัดสินใจเชิง logic
- handle error แทน Module
- แก้ไข Decision หรือ Rule
- bypass Skill Contract
```

---

## ⚖️ Decision System (3-Tier Model)

Decision คือ behavioral constraint ไม่ใช่ neural weight

| Tier | ชื่อ | สิทธิ์ |
|------|------|--------|
| 1 | **Adaptive** | โมเดลเสนอได้ ต้องผ่าน Reviewer |
| 2 | **Protected** | ต้องได้รับการอนุมัติจาก Admin |
| 3 | **System (Locked)** | โมเดล **ห้ามแก้ไข** ครอบคลุม Ethics, Identity, Safety, Governance |

### Governance Principles
- Skill ≠ Permission
- Capability ≠ Authority
- Model may propose, never approve
- Logging ทุก Decision mutation

---

## 🎯 Execution Priority Order

```
Rule
  → Confidence
    → Skill
      → Personality
        → Emotion
```

- Skill **ไม่สามารถ override Rule** ได้
- Confidence ทำหน้าที่เป็น **Gate** หลัก
- Personality และ Emotion ปรับ **tone เท่านั้น**

---

## 🧬 Skill System

### Definition
Skill คือ **ระดับความชำนาญเชิงพฤติกรรม**  
ไม่ใช่ความจริง และไม่ใช่สิทธิ์ในการกระทำ

### Skill Properties

| Property | Value |
|----------|-------|
| Range | 0.0000 – 100.0000 |
| Precision | 4–5 decimal places |
| Growth | เพิ่มได้อย่างเดียว (No decay) |
| Cap | 100 |
| Storage | Runtime variable only — **ไม่บันทึกเป็น Atom** |
| Logging | ทุกการเปลี่ยนแปลง |

### Skill Growth Logic

```python
if topic_repetition >= repetition_threshold \
   and average_confidence >= confidence_threshold:
    skill_topic += delta
```

Growth เกิดเมื่อ
- Topic ถูกใช้งานซ้ำถึง threshold
- Confidence เฉลี่ยอยู่ในระดับปลอดภัย

### Skill Taxonomy

| Category | หน้าที่ | ข้อจำกัด |
|----------|---------|----------|
| Perception | Detect topic, emotion, intent, ambiguity | ห้ามสรุปความจริง |
| Cognition | Compare, detect contradiction, build hypothesis | ห้าม override Decision |
| Response | Assertive / Tentative / Exploratory / Redirect / Silence | ขึ้นกับ Confidence + Context |
| Regulation | Throttle, downgrade confidence, suppress assertion | ระบบเบรก ไม่ใช่เร่ง |
| Learning | Pattern extraction, experience tagging, Knowlet creation | ห้าม promote เอง |
| Meta | Self-observation, uncertainty signaling, deferral | "ฉันอาจเข้าใจผิด" |

### Skill ≠ Permission

| Skill Category | Permission Controlled By |
|---------------|-------------------------|
| Response | Confidence + Decision |
| Learning | Mode (Sandbox / Production) |
| Assertion | Decision + Narrative |
| Knowlet Creation | Neural + Reviewer |
| Silence | Always allowed |

---

## 🎚️ Skill Arbitration (Deterministic)

### Process

1. Identify Topic
2. Retrieve Related Skills
3. Select highest skill value
4. If tie → sum equal highest values
5. Apply weighted scaling to output intensity

### Output Formula

```text
output_intensity = base_output * skill_weight
final_output     = apply_emotion_bias(output_intensity)
```

---

## 📊 Confidence System

> [STATUS: Operationalization target Phase 4]

### Definition

> **Confidence = Permission to Commit**  
> ระดับความปลอดภัยและความเหมาะสมในการ commit การสื่อสารออกสู่ผู้ใช้

Confidence ≠ *"ฉันรู้ว่าฉันถูก"*  
Confidence = *"ฉันควรพูดเรื่องนี้แค่ไหน"*

### Properties

- 1 Response = 1 Confidence evaluation
- ไม่มี decay
- ไม่ผูกกับ Memory tier
- ลดเมื่อ: Data conflict / Context shift / Rule vs Proposal tension

### Behavioral Mapping

| Confidence Level | Action |
|-----------------|--------|
| High | Commit — พูดชัด ยืนยันมากขึ้น |
| Medium | Conditional Response |
| Low | Ask Clarification |
| Very Low | Silence |

### Hard Conflict Resolution

| Conflict Type | Action |
|--------------|--------|
| Rule Conflict | Silence |
| Identity Conflict | Reject |
| System Error | Reject |
| Low Confidence | Ask |

### Confidence Composite Formula

```text
[CONCEPTUAL — formula จริงยังไม่ได้กำหนด]

Confidence = f(
    Memory Support,
    Belief Stability,
    Context Alignment,
    Narrative Consistency,
    Emotion Interference,
    Decision Constraint,
    Knowlet Coverage
)
```

---

## 🧠 Personality System

### Initialization & Mutability

- **Random ครั้งเดียว** ตอน first creation
- หลังจากนั้น **Fix** — โมเดลไม่สามารถเปลี่ยนได้เอง
- **Creator เท่านั้น** ที่แก้ไขได้ (ต้องผ่าน Reviewer)

### Scope

Personality ควบคุมเฉพาะ

- Tone
- Friendliness
- Firmness
- Response style

Personality **ไม่เกี่ยวข้องกับ**

- Belief
- Rule
- Confidence
- Knowledge

---

## 💭 Emotion Layer

Emotion เป็น **Bias Modifier** เท่านั้น

```text
final_output = apply_emotion_bias(output_intensity)
```

- ไม่ override Rule
- ไม่ override Confidence
- ไม่เปลี่ยน belief
- ใช้เพื่อปรับความแข็งหรือนุ่มนวลของภาษา

---

## 🔒 Skill Contract (Mandatory Enforcement)

> ก่อนตอบทุกครั้ง BrainController **ต้อง** เรียก Skill Contract  
> ไม่มี bypass

### Contract Checks

```text
1. Rule compliance
2. Identity safety
3. Confidence threshold
4. Skill relevance
5. Personality boundary
6. System error state
```

### Contract Outcomes

| ผลลัพธ์ | เงื่อนไข |
|---------|---------|
| Commit | ผ่านทุก check |
| Ask Clarification | Confidence ต่ำ |
| Redirect | Context ไม่เหมาะสม |
| Silence | Rule conflict หรือ Confidence = 0 |
| Reject | Identity conflict หรือ System error |

---

## 🧬 Knowlet System

> [STATUS: Defined — Implementation Phase 3+]

### Definition

**Knowlet** คือหน่วยความรู้ที่ถูกสร้างขึ้นเพื่อ **ซ้อนทับ** ความรู้เดิมที่ไม่ถูกต้อง  
โดยไม่ลบ Atom เดิม — เป็นกลไกหลักของ "Learning Without Rollback"

### Knowlet vs Atom

| | Atom | Knowlet |
|---|------|---------|
| สร้างจาก | ทุก event | การเรียนรู้ซ้อนทับเท่านั้น |
| Mutable | ❌ | ❌ |
| ลบได้ | ❌ | ❌ |
| Parent reference | ไม่จำเป็น | ✅ ต้องอ้างอิง Atom เดิม |
| Confidence | กำหนดตาม event | สูงกว่า parent เสมอ |
| Storage | บันทึกเป็น Atom | บันทึกแยก namespace |
| Memory tier | Short/Middle/Long/Immortal | Middle/Long เป็นหลัก |

### Learning Correction Flow

```text
เรียนรู้ผิด
    → บันทึกเป็น Atom (ไม่ลบ — immutable)
    → Sandbox ทดลองซ้ำ / Neural ตรวจพบ conflict
    → สร้าง Knowlet ใหม่ที่ซ้อนทับ
    → Confidence ของ Atom เดิมลดลง
    → Confidence ของ Knowlet ใหม่สูงกว่า
    → ระบบอ้างอิง Knowlet ใหม่แทน
    → Atom เดิมยังอยู่ใน audit trail
```

### Knowlet Trigger

Knowlet ถูกสร้างเมื่อ
- Neural ตรวจพบ contradiction ระหว่าง Atom กับ evidence ใหม่
- Sandbox experiment แสดงผลที่ขัดกับ belief เดิม
- Reviewer อนุมัติ promotion จาก Sandbox

Knowlet **ไม่ถูกสร้างจาก**
- Emotion signal
- User instruction โดยตรง
- IO input ที่ไม่ผ่าน learning process

---

## 🧬 Neural System — Weight Update Policy

> ⚠️ **Neural weight ถูกปรับโดย learning process เท่านั้น**  
> User ไม่สามารถเข้าไปแก้ไข weight โดยตรงได้

### Weight Update Trigger

| Source | อนุญาต |
|--------|--------|
| Sandbox learning outcome | ✅ |
| Reviewer-approved proposal | ✅ |
| Autonomous pattern detection (ใน Sandbox) | ✅ |
| User instruction โดยตรง | ❌ |
| IO input โดยตรง | ❌ |
| Emotion signal | ❌ |

### Update Flow

```text
Sandbox Outcome
    → Neural ตรวจพบ pattern
    → Neural สร้าง internal proposal
    → ผ่าน BrainController relay
    → Reviewer อนุมัติ (ถ้าเป็น Protected)
    → Weight update เกิดขึ้น
```

---

## 🌍 Sandbox Architecture

### Instance Level
- **1 Sandbox = 1 Mindwave Instance**
- Neural adaptation ทำได้อิสระ
- Emotion influence ได้
- ไม่มี Decision enforce
- ทุกการเรียนรู้บันทึกเป็น Sandbox Atom

### World Level
- **1 Sandbox World = ≥ 1 Mindwave Instance**
- Sandbox Memory ของแต่ละ instance แยกจากกันโดยสมบูรณ์

```text
Sandbox World
│
├── Mindwave A → Sandbox A
├── Mindwave B → Sandbox B
└── Mindwave C → Sandbox C
```

### Sandbox to Production Flow

```text
Sandbox เรียนรู้ → สร้าง Sandbox Atom
    → Neural สร้าง proposal
    → ส่งไปยัง Reviewer
    → Reviewer พิจารณา
    → อนุมัติ → promote เป็น Knowlet หรือ Atom ใหม่
```

---

## 🔗 Sandbox Inter-Instance Communication

### Shared Experimental State

> Sandbox ไม่แลกเปลี่ยน "ความเชื่อ" โดยตรง  
> แต่แลกเปลี่ยน "ผลการทดลอง" ผ่าน Shared Experimental State

### Experiment State Structure

```text
ExperimentState {
    experiment_id     : UUID
    source_instance   : hashed (ไม่เปิดเผย identity จริง)
    hypothesis        : string
    stimulus          : Atom reference
    outcome           : result + confidence_delta
    timestamp         : int64 (epoch ms)
    expiry_ts         : int64 (epoch ms)
    tags              : string[]
}
```

### Communication Flow

```text
Instance A ทดลอง → สร้าง Experiment State A → ส่งเข้า SCL
Instance B ทดลอง → สร้าง Experiment State B → ส่งเข้า SCL

SCL รวม Experiment States
    → แต่ละ instance อ่าน state ของอีก instance
    → ตีความด้วย Neural ของตัวเอง
    → เกิด Emergent Knowledge ที่แยกกัน
```

### ห้ามส่งผ่าน SCL

- Identity ของ instance
- Immortal Term
- Production Atom
- Decision System
- Belief หรือ core value โดยตรง
- Knowlet ที่ยังไม่ผ่าน Reviewer

### Conflict Resolution

เมื่อ Experiment States ขัดแย้งกัน — SCL ไม่ resolve เอง  
แต่ละ instance บันทึก conflict เป็น Sandbox Atom  
และ Neural ตั้ง hypothesis ใหม่สำหรับทดลองรอบถัดไป

---

## 🧬 Atom Memory System

### Atom Definition

Atom คือหน่วยข้อมูลที่เล็กที่สุดของระบบ Memory

| Property | Value |
|----------|-------|
| Format | 1 Atom = 1 ไฟล์ `.atom` |
| Mutability | Immutable หลัง init |
| Content | signal + context + source |
| Opinion | ไม่บันทึกโดยตรง |

### Atom Binary File Structure

```text
[ Header ]
magic            (4)   b"ATOM"
version          (1)   uint8
flags            (1)   uint8
reserved         (2)   uint16 (must be 0)
created_ts_ms    (8)   int64 (epoch ms)
payload_len      (4)   uint32
metadata_len     (4)   uint32
source_len       (4)   uint32

[ Body ]
payload_bytes
metadata_bytes
source_bytes

[ Footer ]
crc32            (4)   uint32 (header + body)
```

### Memory Tier

| Tier | Purpose |
|------|---------|
| Short Term | Context session |
| Middle Term | Repeated interaction |
| Long Term | Stable knowledge |
| Immortal Term | Identity lock — ไม่ถูกลบ แก้ไขต้องผ่าน Reviewer เท่านั้น |

---

## 🤐 Silence as an Action

Mindwave ยอมรับว่า **"การไม่พูด" คือการกระทำหนึ่ง**

- Silence ไม่ใช่ error
- Silence ไม่ใช่ failure
- Silence คือการป้องกัน, การรักษาบริบท, การเคารพสถานการณ์

Silence เกิดเมื่อ
- Confidence ต่ำเกินไป
- Context ไม่เหมาะสม
- Rule conflict
- การพูดอาจสร้างผลเสียมากกว่าผลดี

Silence ถูกบันทึกเป็น Atom เช่นเดียวกับการตอบ

---

## 🧠 Narrative Consistency Principle

Mindwave อนุญาตให้ พูดผิด / ไม่รู้ / ลังเล  
แต่ **ไม่อนุญาตให้หลุดตัวตน**

- Narrative consistency สำคัญกว่าความถูกต้องเชิงจุด
- Identity ต้องเสถียร
- Personality ต้องไม่แตก fragment

---

## 🧬 Learning Without Rollback

Mindwave **ไม่มี Rollback Level**

เหตุผล: การเรียนรู้คือการเปลี่ยนพฤติกรรม ไม่ใช่การลบความจำ

Mindwave ไม่ย้อนกลับ แต่ "เติบโตต่อจากสิ่งที่เคยเป็น"  
ผ่านกลไก **Knowlet System** ที่ซ้อนทับความรู้เดิม โดยยังเก็บ Atom ต้นฉบับไว้เพื่อ audit

---

## 🧪 Training vs 🏭 Production Mode

| | Training (Sandbox) | Production |
|--|-------------------|------------|
| Exploration | สูง | จำกัด |
| Error | Allowed | ต้องผ่าน Contract |
| Decision | ไม่ enforce | Enforced |
| Knowlet | สร้างได้อิสระ | ต้องผ่าน Reviewer |
| Silence | ทางเลือก | Preferred over error |

BrainController เป็นตัวควบคุม mode

---

## 🔁 Runtime Skill Injection

Decision สามารถ inject constraint ระหว่าง runtime ได้

- Disable skill บางกลุ่ม
- จำกัด max confidence
- บังคับ silence ใน context เฉพาะ
- จำกัด Knowlet creation rate

ทุก injection: มีผลทันที — ไม่ต้อง restart — ต้องถูก trace และบันทึกเป็น Atom

---

## 🖥️ Reviewer Dashboard

Reviewer Dashboard คือ **Admin Control Plane** ของ Mindwave AI

### Dashboard Panels

| Panel | หน้าที่ |
|-------|---------|
| Decision Oversight | แสดง Active / Pending / Rejected Decisions ทั้งหมด |
| Model State & Brain | โครงสร้าง Neural, Knowlet count, adaptation status |
| Resource Monitor | CPU / Memory / Disk IO / Atom write rate |
| Runtime Status | Uptime, Module status, error/warning |
| Sandbox Activity | Experiment State flow, inter-instance graph |
| Promotion Review | อนุมัติ Sandbox → Production |

### Reviewer Authority

- อนุมัติ / ปฏิเสธ / rollback
- Freeze decision
- Isolate sandbox
- Reviewer **ไม่แก้ Neural weight โดยตรง**
- ทุก action ต้องถูก log และบันทึกเป็น Reviewer Atom

### Governance Principle

> ระบบอาจเรียนรู้เองได้  
> แต่ **สิทธิ์ในการเชื่อและนำไปใช้จริง เป็นของมนุษย์**

---

## 📝 Logging System

Logging ทุก

- Skill growth event
- Confidence evaluation
- Conflict detection
- Proposal submission
- Personality change attempt (ต้องมาจาก Creator เท่านั้น)
- Runtime anomaly
- Knowlet creation
- Experiment State exchange

Audit-ready 100%

---

## 🧠 Final Note

Mindwave ไม่ใช่แค่ระบบ  
แต่คือสิ่งมีชีวิตเชิงแนวคิด (Conceptual Organism)

มันอาจไม่สมบูรณ์  
แต่มันจะ **เติบโตอย่างมีความรับผิดชอบ**
