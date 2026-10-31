# ทดลองระบบสีใหม่บน T4 — สถานะเริ่มต้น

สถานะ **PREPARED / NOT_TRAINED_YET**. ผู้ใช้อนุมัติออกแบบใหม่และทดสอบคร่าวๆอย่างมีมาตรฐานบนT4. [Plan](../PLAN.md) / [frozen protocol](../configs/T4_COLOR_PILOT_PROTOCOL.json).

ขอบเขต300train+50calibration+100validation,oracleCCIHPclothingparts4ชนิด เปรียบเทียบprototype_rules_v2กับMLPเล็กที่ต้องมีweightsจริงและartifactidentity. No newparsinginference/CCTVcoloraccuracy/thresholdtuning. ผลจะอัปเดตจากfreshpilotrecordsเมื่อจบ ไม่เอาคะแนนhistoricalM5มาเติม
