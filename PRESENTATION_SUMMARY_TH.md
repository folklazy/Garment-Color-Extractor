> **คำตัดสินหลังการเปรียบเทียบ:** ผู้ใช้มอบหมายให้เลือกหนึ่งวิธีหลักแล้ว เลือก **T4_MLP_62_64_32_12_v1** ตาม primary strict accuracy68.73%. อ่าน [method profile แบบบทความวิจัย](reports/SELECTED_COLOR_METHOD_TH.md) และ [selection decision](registry/SELECTED_METHOD.json). ข้อเสนอ shortlist ในเนื้อหาการเปรียบเทียบด้านล่างเป็นบันทึกก่อนคำตัดสินนี้; ผลและข้อได้เปรียบ macro F1 ของ M4 คงเดิม.

# สรุปสำหรับนำเสนอ — เปรียบเทียบวิธีสีเสื้อผ้า

1. **โจทย์:** เมื่อmaskเสื้อผ้าถูกต้องแล้ว วิธีใดจำแนกสีที่เห็นในภาพได้ดีกว่า ใช้CCIHPannotationเป็นreference ไม่ใช้กฎของเราเองสร้างคำตอบ.
2. **วิธี:** M1HSV/M2anchor/M3LABglobal/M4LABcategory/M5HGB+ExtraTrees+prototype เทียบrulesและT4MLP; ข้อมูลและmaskตรงกัน7วิธี. ฝึก300ภาพ/calibration50ภาพ/validation100ภาพที่เลือก/97ภาพมีeligiblegarments รวม614regions.
3. **ผล:** T4_MLP_62_64_32_12_v1นำstrictaccuracy 68.73% และ M4นำmacroF1 58.07%. M5เป็นmodelที่train/export/readbackจริง ไม่ใช้ผลM3แทน. ดู [ตาราง/กราฟหลัก](REPORT.md#ผลรวม--frozen-method-order).
4. **ความน่าเชื่อถือ:** ตรึงสมาชิก/สูตรก่อนfitรอบใหม่; ไม่tuneจากValidation; ตรวจexactmembership/GT/sourcehashes/metrics/reload. ไม่ลบตัวอย่างที่ทายผิด.
5. **ขอบเขต:** เป็นผลย่อยบนoraclemask ไม่ยืนยันCCTV/nightIR/end-to-endparsingหรือสีวัสดุจริง. ข้อมูลหายากน้อย ไม่มีstatisticalsignificance/speedranking. ตัวนำเป็นcandidateสำหรับงานต่อ ไม่ใช่finaldeploymentacceptance.

ใช้กราฟที่ไม่มีภาพdatasetได้จาก [method comparison](reports/color-methods-20261101/METHOD_COMPARISON.svg) และ [per-color F1](reports/color-methods-20261101/PER_COLOR_F1.svg). ตัวเลขเต็มและข้อจำกัดอยู่ [REPORT.md](REPORT.md); ไม่อัปโหลดCCTVassets.
