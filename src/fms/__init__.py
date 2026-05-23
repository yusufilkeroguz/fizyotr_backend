"""Functional Movement Screen (FMS) video analizcileri.

FMS, 7 harekelik bir tarama protokolu:
    1. Deep Squat (derin cokelme)
    2. Hurdle Step (engel adimi)
    3. In-Line Lunge (dogru hat lunge)
    4. Shoulder Mobility (omuz mobilitesi)
    5. Active Straight-Leg Raise / ASLR
    6. Trunk Stability Push-Up
    7. Rotary Stability

Her test 0-3 arasi skorlanir:
    3 = kriterlerin tamamini kompanzasyon olmadan yerine getirdi
    2 = tamamladi ama kompanzasyon/kisitli mobilite var
    1 = hareketi tamamlayamadi
    0 = testi sirasinda agri
Ikinci ve ucuncu tarafli testlerde sol/sag icin ayri skor alinir;
son skor iki tarafin kucugudur.
"""
