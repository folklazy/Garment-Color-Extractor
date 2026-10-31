# สรุปผลสีเสื้อผ้า 7 วิธี

ทดสอบครบ7/7วิธี บน614oracle garment regionsจาก100Validationภาพที่เลือก (97ภาพให้regionsผ่านเกณฑ์) โดยใช้Training300ภาพและcalibration50ภาพชุดเดียวกัน. [รายงานหลัก](REPORT.md) / [กติกา](EXPERIMENT_PROTOCOL.md).

| วิธี | ถูก/ทั้งหมด | Strict accuracy % | Macro F1 fixed12 % |
|---|---:|---:|---:|
| M1 — HSV rules | 393/614 | 64.01 | 54.19 |
| M2 — 11-anchor soft projection | 339/614 | 55.21 | 29.05 |
| M3 — Global LAB prototypes | 406/614 | 66.12 | 56.35 |
| M4 — Category LAB prototypes | 415/614 | 67.59 | 58.07 |
| M5 — HGB + ExtraTrees + prototype | 407/614 | 66.29 | 56.22 |
| prototype_rules_v2 — Library rules v2 | 358/614 | 58.31 | 47.47 |
| T4_MLP_62_64_32_12_v1 — T4 MLP 62–64–32–12 | 422/614 | 68.73 | 52.55 |

Strictaccuracyนำ: **T4_MLP_62_64_32_12_v1 68.73%** สูงกว่าM4 1.14percentagepoints. MacroF1นำ: **M4 58.07%**. MLPมีaccuracyสูงกว่าM4 7regionsแต่macroF1ต่ำกว่า5.52percentagepoints จึงเสนอshortlistทั้งสองตามวัตถุประสงค์. M5คือensembleที่fitจริง ไม่ใช่aliasM3. Rules/MLP reuseหลักฐานเดิมโดยไม่ฝึกซ้ำ.

เป็นexploratorysubsetcolorclassification ไม่ใช่CIHP5000/parsingaccuracy/CCTVaccuracy; Orange3/Purple2regions. ไม่มีspeedrankingหรือdeploymentacceptance. ไม่มีการเปลี่ยนdefaultAPIอัตโนมัติ. Canonicaltablesและsource/artifacthashesอยู่ [run evidence](reports/color-methods-20261101/MANIFEST.json).
