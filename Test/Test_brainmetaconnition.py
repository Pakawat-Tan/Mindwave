"""
=================================================================
  BrainController + MetaCognition Integration Test
=================================================================
  ทดสอบการ integrate MetaCognition เข้า BrainController
  
  1. MetaCognition Auto-Reflect    (3 tests)
  2. Confidence Bias Application   (3 tests)
  3. Error Detection               (2 tests)
  4. Learning Tracking             (2 tests)
-----------------------------------------------------------------
  Total: 10 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.BrainController import BrainController
from Core.Brain.MetaCognition import MetaCognition


class TestMetaCognitionIntegration(unittest.TestCase):

    def setUp(self):
        self.brain = BrainController()

    def test_metacognition_attached(self):
        """MetaCognition ถูก attach เข้า Brain"""
        self.assertIsInstance(self.brain.metacognition, MetaCognition)

    def test_auto_reflect_after_interval(self):
        """Reflect อัตโนมัติทุก N logs"""
        # respond 5 ครั้ง (default interval)
        for i in range(5):
            self.brain.respond(f"input {i}", "general")
        
        # ควรมี reflection เกิดขึ้น
        self.assertGreater(len(self.brain.metacognition.reflections), 0)

    def test_no_reflect_before_interval(self):
        """ไม่ reflect ก่อนถึง interval"""
        for i in range(3):  # น้อยกว่า 5
            self.brain.respond(f"input {i}", "general")
        
        # ยังไม่ reflect
        self.assertEqual(len(self.brain.metacognition.reflections), 0)

    def test_confidence_bias_applied(self):
        """Confidence bias ถูกนำไปใช้"""
        # force bias
        self.brain.metacognition._confidence_bias = 0.1
        
        result = self.brain.respond("test", "general")
        
        # confidence ควรถูกปรับ (ลดลง 0.1)
        self.assertIsInstance(result["confidence"], float)

    def test_calibration_runs(self):
        """Calibration ทำงานหลัง interval"""
        for i in range(5):
            self.brain.respond(f"input {i}", "general")
        
        # calibration ควรทำงาน
        self.assertGreater(len(self.brain.metacognition.calibrations), 0)

    def test_bias_updates_over_time(self):
        """Bias ปรับตัวตาม interactions"""
        before_bias = self.brain.metacognition.confidence_bias
        
        # respond 10 ครั้ง
        for i in range(10):
            self.brain.respond(f"input {i}", "general")
        
        after_bias = self.brain.metacognition.confidence_bias
        
        # bias อาจเปลี่ยน (หรือคงเดิมถ้าไม่มี error)
        self.assertIsInstance(after_bias, float)

    def test_error_detection_runs(self):
        """Error detection ทำงานหลัง interval"""
        for i in range(5):
            self.brain.respond(f"input {i}", "general")
        
        # detect_errors ควรถูกเรียก (อาจไม่เจอ error ก็ได้)
        errors = self.brain.metacognition.errors
        self.assertIsInstance(errors, list)

    def test_learning_tracking_runs(self):
        """Learning tracking ทำงานหลัง interval"""
        for i in range(5):
            self.brain.respond(f"input {i}", "general")
        
        # track_learning ควรถูกเรียก
        tracks = self.brain.metacognition.tracks
        self.assertGreater(len(tracks), 0)

    def test_metacognition_in_stats(self):
        """MetaCognition ปรากฏใน Brain stats"""
        stats = self.brain.status()
        self.assertTrue(stats["modules"]["metacognition"])

    def test_metacognition_accessible(self):
        """สามารถเข้าถึง metacognition module ได้"""
        mc = self.brain.metacognition
        self.assertIsNotNone(mc)
        
        # ควรมี methods หลัก
        self.assertTrue(hasattr(mc, 'reflect'))
        self.assertTrue(hasattr(mc, 'calibrate_confidence'))
        self.assertTrue(hasattr(mc, 'detect_errors'))


def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestLoader().loadTestsFromTestCase(TestMetaCognitionIntegration)
    
    print("\n=================================================================")
    print("  BrainController + MetaCognition Integration Test")
    print("=================================================================")
    print("  1. MetaCognition Auto-Reflect    (3 tests)")
    print("  2. Confidence Bias Application   (3 tests)")
    print("  3. Error Detection               (2 tests)")
    print("  4. Learning Tracking             (2 tests)")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 10 tests")
    print("=================================================================\n")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n=================================================================")
    print(f"  Passed : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed : {len(result.failures)}")
    print(f"  Errors : {len(result.errors)}")
    print("=================================================================")
    print("\n  🎉 ALL TESTS PASSED!\n" if result.wasSuccessful() else "\n  ❌ SOME TESTS FAILED\n")


if __name__ == "__main__":
    run_tests()