import numpy as np
import pickle

# ============================================================
# MODEL COEFFICIENTS — extracted from trained GBR
# CV R² = 0.9876 | RMSE = 0.2273 mm | MAPE = 6.37%
# Trained on 500 PLAXIS 3D FEA simulations
# University of Ibadan — Apara Olalewa — 252070
# ============================================================

INTERCEPT = 2.7429
COEFS = {
    'log_e50':   -0.5921,
    'cref':      -0.0008,
    'phi':       -0.0026,
    'thickness': -2.3419,
    'load':       0.0006641,
    'isMedium':   0.001783,
    'isSoft':     0.001857,
    'isStiff':   -0.000401,
}

SOIL_GROUPS = {
    '1': ('Lagos_Blue_Clay',  [5000,  15000]),
    '2': ('Soft_Laterite',    [15000, 40000]),
    '3': ('Medium_Laterite',  [40000, 80000]),
    '4': ('Stiff_Laterite',   [80000, 150000]),
}

def cbr_to_e50(cbr, soil_group):
    e50 = 17.6 * (cbr ** 0.64) * 1000
    lo, hi = SOIL_GROUPS[soil_group][1]
    return round(min(hi, max(lo, e50)))

def predict_rut(e50, cref, phi, thickness_mm, load_kPa, soil_group):
    log_e50    = np.log10(max(5000, e50))
    thickness  = thickness_mm / 1000
    is_medium  = 1 if SOIL_GROUPS[soil_group][0] == 'Medium_Laterite' else 0
    is_soft    = 1 if SOIL_GROUPS[soil_group][0] == 'Soft_Laterite'   else 0
    is_stiff   = 1 if SOIL_GROUPS[soil_group][0] == 'Stiff_Laterite'  else 0

    log_rut = (
        INTERCEPT
        + COEFS['log_e50']   * log_e50
        + COEFS['cref']      * cref
        + COEFS['phi']       * phi
        + COEFS['thickness'] * thickness
        + COEFS['load']      * load_kPa
        + COEFS['isMedium']  * is_medium
        + COEFS['isSoft']    * is_soft
        + COEFS['isStiff']   * is_stiff
    )
    return round(max(0.05, 10 ** log_rut), 4)

def rut_status(rut):
    if rut < 3:  return "✅ ACCEPTABLE"
    if rut < 6:  return "⚠️  MODERATE"
    if rut < 10: return "🔴 SEVERE"
    return              "🚨 CRITICAL"

# ============================================================
# MAIN — interactive prompt
# ============================================================
print("=" * 60)
print("  SW NIGERIA RUTTING DEFORMATION PREDICTOR")
print("  Apara Olalewa | 252070 | University of Ibadan")
print(f"  Model: Gradient Boosting | CV R² = 0.9876 | MAPE = 6.37%")
print("=" * 60)

while True:
    print("\nSelect input mode:")
    print("  1 — CBR input (for practitioners)")
    print("  2 — Full parameters (for researchers)")
    print("  3 — Run known site (Lagos-Ibadan Expressway km 90)")
    print("  Q — Quit")

    mode = input("\nChoice: ").strip().upper()

    if mode == 'Q':
        print("\nExiting. Model by Apara Olalewa, University of Ibadan.")
        break

    # ── SELECT SOIL GROUP ──
    if mode != '3':
        print("\nSoil / Subgrade Type:")
        print("  1 — Lagos Blue Clay     (CBR 2–8%,  E50: 5,000–15,000 kPa)")
        print("  2 — Soft Laterite       (CBR 8–20%, E50: 15,000–40,000 kPa)")
        print("  3 — Medium Laterite     (CBR 20–40%,E50: 40,000–80,000 kPa)")
        print("  4 — Stiff Laterite      (CBR 40–80%,E50: 80,000–150,000 kPa)")
        soil_key = input("Soil group (1-4): ").strip()
        if soil_key not in SOIL_GROUPS:
            print("Invalid selection."); continue
        soil_name = SOIL_GROUPS[soil_key][0]

    # ── CBR MODE ──
    if mode == '1':
        cbr = float(input(f"CBR value (%): "))
        e50 = cbr_to_e50(cbr, soil_key)
        print(f"  → Derived E50Ref = {e50:,} kPa (Powell, 1984)")
        cref = float(input("Cohesion cRef (kPa) [press Enter for typical value]: ").strip() or
                     {'1':28,'2':18,'3':14,'4':10}[soil_key])
        phi  = float(input("Friction angle phi (°) [press Enter for typical value]: ").strip() or
                     {'1':22,'2':29,'3':33,'4':37}[soil_key])

    # ── ADVANCED MODE ──
    elif mode == '2':
        e50  = float(input("E50Ref (kPa): "))
        cref = float(input("Cohesion cRef (kPa): "))
        phi  = float(input("Friction angle phi (°): "))

    # ── KNOWN SITE ──
    elif mode == '3':
        print("\nLoading: Lagos-Ibadan Expressway km 90")
        print("Source: Akintayo & Ibrahim (2024), UICIVIL 2024")
        soil_key  = '4'
        soil_name = 'Stiff_Laterite'
        cbr       = 26
        e50       = cbr_to_e50(26, '4')
        cref      = 10
        phi       = 37
        print(f"  CBR = 26% → E50Ref = {e50:,} kPa")
        print(f"  cRef = {cref} kPa | phi = {phi}°")
    else:
        print("Invalid selection."); continue

    # ── PAVEMENT DESIGN ──
    if mode != '3':
        thickness = float(input("Asphalt thickness (mm) [75–250]: "))
        print("\nAxle load type:")
        print("  1 — Light vehicle       400 kN/m²")
        print("  2 — Medium truck        600 kN/m²")
        print("  3 — Standard axle 80kN  700 kN/m²")
        print("  4 — Heavy commercial    800 kN/m²")
        print("  5 — Overloaded HCV      900 kN/m²")
        load_map = {'1':400,'2':600,'3':700,'4':800,'5':900}
        load_key  = input("Load type (1-5): ").strip()
        load      = load_map.get(load_key, 700)
    else:
        thickness = 150
        load      = 700

    # ── PREDICT ──
    rut    = predict_rut(e50, cref, phi, thickness, load, soil_key)
    status = rut_status(rut)
    margin = 20 - rut

    print("\n" + "=" * 60)
    print(f"  RESULT")
    print("=" * 60)
    print(f"  Soil type        : {soil_name.replace('_',' ')}")
    print(f"  E50Ref           : {e50:>10,} kPa")
    print(f"  Cohesion         : {cref:>10.1f} kPa")
    print(f"  Friction angle   : {phi:>10.1f} °")
    print(f"  Asphalt thickness: {thickness:>10.0f} mm")
    print(f"  Axle load        : {load:>10} kN/m²")
    print(f"  ─────────────────────────────────────")
    print(f"  Predicted rut    : {rut:>10.4f} mm")
    print(f"  Status           : {status}")
    print(f"  Margin to failure: {margin:>10.3f} mm (threshold = 20mm)")
    print("=" * 60)

    again = input("\nRun another scenario? (Y/N): ").strip().upper()
    if again != 'Y':
        print("\nDone. Model by Apara Olalewa, University of Ibadan.")
        break