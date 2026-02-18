# 🧠 Mindwave AI — Detailed File Structure

```
├── 📁 Core
│   ├── 📁 Brain
│   │   ├── 🐍 BeliefSystem.py
│   │   ├── 🐍 DistributedSystem.py
│   │   ├── 🐍 EmotionInference.py
│   │   ├── 🐍 FeedbackInference.py
│   │   ├── 🐍 LearnMode.py
│   │   ├── 🐍 MetaCognition.py
│   │   ├── 🐍 NeuralTrainer.py
│   │   ├── 🐍 PatternRecognition.py
│   │   ├── 🐍 ResponseEngine.py
│   │   └── 🐍 TopicClustering.py
│   ├── 📁 Condition
│   │   ├── 📁 Data
│   │   │   ├── 📁 policy
│   │   │   └── 📁 rule
│   │   ├── 📁 Policy
│   │   │   ├── 🐍 PolicyController.py
│   │   │   └── 🐍 PolicyData.py
│   │   ├── 📁 Rule
│   │   │   ├── 📁 Defaults
│   │   │   │   ├── ⚙️ Adaptation.json
│   │   │   │   ├── ⚙️ Brain.json
│   │   │   │   ├── ⚙️ Emotion.json
│   │   │   │   ├── ⚙️ IO.json
│   │   │   │   ├── ⚙️ Learning.json
│   │   │   │   ├── ⚙️ Memory.json
│   │   │   │   ├── ⚙️ NeuralEvolution.json
│   │   │   │   ├── ⚙️ Routing.json
│   │   │   │   ├── ⚙️ Safety.json
│   │   │   │   ├── ⚙️ Sandbox.json
│   │   │   │   ├── ⚙️ Skill.json
│   │   │   │   ├── ⚙️ SystemRule.json
│   │   │   │   └── ⚙️ Topic.json
│   │   │   ├── 🐍 RuleController.py
│   │   │   └── 🐍 RuleData.py
│   │   ├── 🐍 ConditionController.py
│   │   └── 🐍 Proposal.py
│   ├── 📁 Confidence
│   │   ├── 🐍 ConfidenceController.py
│   │   └── 🐍 ConfidenceData.py
│   ├── 📁 Data
│   │   ├── 📁 io_logs
│   │   ├── 📁 knowlet
│   │   ├── 📁 production
│   │   │   ├── 📁 immortal
│   │   │   ├── 📁 long
│   │   │   ├── 📁 middle
│   │   │   └── 📁 short
│   │   └── 📁 sandbox
│   ├── 📁 IO
│   │   ├── 📁 Channels
│   │   │   ├── 🐍 CLIChannel.py
│   │   │   ├── 🐍 EventBusChannel.py
│   │   │   ├── 🐍 FileChannel.py
│   │   │   ├── 🐍 InternetChannel.py
│   │   │   ├── 🐍 RESTChannel.py
│   │   │   ├── 🐍 SocketChannel.py
│   │   │   ├── 🐍 SoundChannel.py
│   │   │   └── 🐍 VideoChannel.py
│   │   ├── 🐍 IOController.py
│   │   ├── 🐍 IOLogger.py
│   │   └── 🐍 IOPacket.py
│   ├── 📁 Memory
│   │   ├── 📁 Structure
│   │   │   ├── 🐍 AtomRepair.py
│   │   │   ├── 🐍 AtomStructure.py
│   │   │   └── 🐍 KnowletStructure.py
│   │   ├── 📁 Tiers
│   │   │   ├── 🐍 Immortal_term.py
│   │   │   ├── 🐍 Long_term.py
│   │   │   ├── 🐍 Middle_term.py
│   │   │   ├── 🐍 Short_term.py
│   │   │   └── 🐍 base.py
│   │   ├── 🐍 Emotion.py
│   │   ├── 🐍 KnowletController.py
│   │   ├── 🐍 MemoryController.py
│   │   └── 🐍 Topic.py
│   ├── 📁 Neural
│   │   ├── 📁 Brain
│   │   │   ├── 📁 Functions
│   │   │   │   ├── 🐍 Activation.py
│   │   │   │   └── 🐍 LossFunction.py
│   │   │   ├── 🐍 BrainStructure.py
│   │   │   ├── 🐍 NeuralData.py
│   │   │   └── 🐍 Schema.py
│   │   └── 🐍 NeuralController.py
│   ├── 📁 Personality
│   │   ├── 🐍 PersonalityController.py
│   │   └── 🐍 PersonalityData.py
│   ├── 📁 Review
│   │   ├── 🐍 Proposal.py
│   │   ├── 🐍 ReviewerController.py
│   │   └── 🐍 ReviewerData.py
│   ├── 📁 Sandbox
│   │   ├── 🐍 SCL.py
│   │   ├── 🐍 SandboxController.py
│   │   └── 🐍 SandboxData.py
│   ├── 📁 Skill
│   │   ├── 🐍 SkillController.py
│   │   └── 🐍 SkillData.py
│   ├── 📁 Train
│   │   └── 🐍 TrainingPipeline.py
│   └── 🐍 BrainController.py
├── 📁 Datasets
│   ├── 📄 Identity.txt
│   ├── 📄 conversation.txt
│   ├── 📄 creativity.txt
│   ├── 📄 emotion.txt
│   └── 📄 reasoning.txt
├── 📁 Doc
│   ├── 📁 Phase
│   │   ├── 📝 Phase0_Identity.md
│   │   ├── 📝 Phase1_Structure.md
│   │   ├── 📝 Phase2_Governance.md
│   │   ├── 📝 Phase3_Skill.md
│   │   ├── 📝 Phase4_Runtime.md
│   │   └── 📝 Phase5_Emergent.md
│   ├── 📁 Spec
│   │   ├── 📝 AtomSpec.md
│   │   ├── 📝 ConfidenceSpec.md
│   │   ├── 📝 KnowletSpec.md
│   │   └── 📝 SkillSpec.md
│   ├── 📝 FileStructure.md
│   ├── 📝 Integration Summary.md
│   ├── 📝 README Production.md
│   └── 📝 Roadmap.md
├── 📁 Test
│   ├── 📁 Brain
│   │   ├── 🐍 Test_distributed.py
│   │   ├── 🐍 Test_emotioninference.py
│   │   ├── 🐍 Test_feedblackinfference.py
│   │   ├── 🐍 Test_metacognition.py
│   │   ├── 🐍 Test_patternrecognition.py
│   │   └── 🐍 Test_topicclustering.py
│   ├── 📁 Condition
│   │   └── 🐍 Test_proposal.py
│   ├── 📁 Confidence
│   │   └── 🐍 Test_confidence.py
│   ├── 📁 Memory
│   │   ├── 🐍 Test_context.py
│   │   ├── 🐍 Test_emotion.py
│   │   ├── 🐍 Test_memory.py
│   │   └── 🐍 Test_topic.py
│   ├── 📁 Neural
│   │   └── 🐍 Test_structure.py
│   ├── 📁 Personality
│   │   └── 🐍 Test_personality.py
│   ├── 📁 Review
│   │   └── 🐍 Test_reviewer.py
│   ├── 📁 Sandbox
│   │   └── 🐍 Test_sanbox.py
│   ├── 📁 Skill
│   │   └── 🐍 Test_skill.py
│   ├── 📁 Train_Dataset
│   │   └── 📄 sample_train.txt
│   ├── 🐍 Test_braincontroller.py
│   ├── 🐍 Test_brainmetaconnition.py
│   ├── 🐍 Test_evolution.py
│   └── 🐍 Test_integration.py
├── 📝 README.md
└── 🐍 main.py
```

---

## 📌 หมายเหตุ

### ไฟล์ที่มีอยู่แล้ว (เก็บไว้)
- `Memory/Structure/AtomRepair.py`
- `Memory/Structure/AtomStructure.py`
- `Memory/Structure/Atom_file.py`
- `Memory/Tiers/*.py`
- `Memory/Emotion.py`
- `Memory/Topic.py`
- `Neural/BrainController.py`

### ไฟล์ใหม่ที่เพิ่ม
- `Core/Skill/` — ทั้ง folder ใหม่
- `Core/Confidence/` — ทั้ง folder ใหม่
- `Core/Personality/` — ทั้ง folder ใหม่
- `Core/Memory/KnowletController.py` — Knowlet system
- `Core/Review/` — ครบ folder
- `Core/Condition/Rule/RuleRegistry.py` — runtime rule management
- `IO/` — ครบ folder
- `Doc/Phase/` และ `Doc/Spec/` — แยก doc ตาม phase

### Convention
- `base.py` ใน folder ไหนก็ตาม = Abstract base class ของ folder นั้น
- `*Controller.py` = จัดการ lifecycle ของ module
- `*Manager.py` = จัดการ state และ mutation
- `*Registry.py` = จัดเก็บและค้นหา entities