import csv
import time
import subprocess
import numpy as np
from scipy.stats import qmc
from plxscripting.easy import new_server

HOST        = 'localhost'
PORT_INPUT  = 10000
PORT_OUTPUT = 10001
PASSWORD    = '#S=8DDB#X6iZR#Vc'
OUTPUT_CSV  = r'C:\Users\DELL\Desktop\rutting_dataset_stratified.csv'

# ── SOIL GROUP DEFINITIONS ────────────────────────────────────────────────────
# Each group: (name, e50_min, e50_max, cref_min, cref_max,
#              phi_min, phi_max, eur_ratio, n_runs)
SOIL_GROUPS = [
    ('Lagos_Blue_Clay',  5000,   15000,  15, 40, 18, 26, 5, 125),
    ('Soft_Laterite',   15000,   40000,  10, 25, 26, 32, 3, 125),
    ('Medium_Laterite', 40000,   80000,   8, 20, 30, 36, 3, 125),
    ('Stiff_Laterite',  80000,  150000,   5, 15, 34, 40, 3, 125),
]

# Shared across all groups
THICK_MIN = 0.075   # 75mm
THICK_MAX = 0.250   # 250mm
LOAD_MIN  = 400     # kN/m²
LOAD_MAX  = 900     # kN/m²

# ── GENERATE ALL 500 SCENARIOS ────────────────────────────────────────────────
print("Generating stratified Latin Hypercube samples...")
all_scenarios = []
run_counter = 0

for (soil_name, e50_min, e50_max, c_min, c_max,
     phi_min, phi_max, eur_ratio, n) in SOIL_GROUPS:

    # LHS sampler — 4 dimensions per group
    # (E50, cRef, phi, thickness) — load sampled separately below
    sampler = qmc.LatinHypercube(d=5, seed=42)
    samples = sampler.random(n=n)

    # Scale to physical ranges
    e50_arr   = np.round(10 ** (
        samples[:,0] * (np.log10(e50_max) - np.log10(e50_min))
        + np.log10(e50_min)
    )).astype(int)
    cref_arr  = np.round(samples[:,1] * (c_max   - c_min)   + c_min,   1)
    phi_arr   = np.round(samples[:,2] * (phi_max  - phi_min) + phi_min, 1)
    thick_arr = np.round(samples[:,3] * (THICK_MAX - THICK_MIN) + THICK_MIN, 4)
    load_arr  = np.round(samples[:,4] * (LOAD_MAX  - LOAD_MIN)  + LOAD_MIN,  0).astype(int)

    for i in range(n):
        run_counter += 1
        phi   = float(phi_arr[i])
        psi   = round(max(0.0, phi - 30.0), 1)
        e50   = int(e50_arr[i])
        eoed  = e50
        eur   = e50 * eur_ratio

        all_scenarios.append({
            'run':       run_counter,
            'soil_type': soil_name,
            'e50':       e50,
            'eoed':      eoed,
            'eur':       eur,
            'cref':      float(cref_arr[i]),
            'phi':       phi,
            'psi':       psi,
            'thickness': float(thick_arr[i]),
            'load':      int(load_arr[i]),
        })

print(f"Total scenarios: {len(all_scenarios)}")
print(f"  Lagos Blue Clay : {sum(1 for s in all_scenarios if s['soil_type']=='Lagos_Blue_Clay')}")
print(f"  Soft Laterite   : {sum(1 for s in all_scenarios if s['soil_type']=='Soft_Laterite')}")
print(f"  Medium Laterite : {sum(1 for s in all_scenarios if s['soil_type']=='Medium_Laterite')}")
print(f"  Stiff Laterite  : {sum(1 for s in all_scenarios if s['soil_type']=='Stiff_Laterite')}")
print()

# ── RESUME LOGIC ──────────────────────────────────────────────────────────────
completed_runs = set()
try:
    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and len(row) >= 10 and row[9] != 'ERROR':
                completed_runs.add(int(row[0]))
    print(f"Resuming — {len(completed_runs)} runs already completed")
    print(f"Next run: {max(completed_runs) + 1 if completed_runs else 1}\n")
except FileNotFoundError:
    print("No existing CSV — starting fresh\n")

# ── CONNECT TO PLAXIS ─────────────────────────────────────────────────────────
print("Connecting to PLAXIS Input...")
s_i, g_i = new_server(HOST, PORT_INPUT, password=PASSWORD)
print("Connected ✅")

subgrade_mat = None
for mat in g_i.Materials:
    if mat.Name.value == 'Subgrade':
        subgrade_mat = mat
        break
print(f"Material: {subgrade_mat.Name.value} ✅")

g_i.gotosoil()
borehole = g_i.Boreholes[0]
print(f"Borehole: {borehole.Name.value} ✅\n")
print(f"Starting runs...\n")
print(f"{'─'*90}")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
results    = []
failed     = []
start_time = time.time()

with open(OUTPUT_CSV, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header only for new file
    if not completed_runs:
        writer.writerow([
            'run', 'soil_type', 'e50_kPa', 'eoed_kPa', 'eur_kPa',
            'cref_kPa', 'phi_deg', 'psi_deg',
            'thickness_m', 'load_kPa', 'rut_mm'
        ])

    for sc in all_scenarios:

        run_num = sc['run']

        if run_num in completed_runs:
            continue

        try:
            # Step 1: Update subgrade material
            g_i.gotostages()
            subgrade_mat.setproperties(
                'E50Ref',  sc['e50'],
                'EOedRef', sc['eoed'],
                'EURRef',  sc['eur'],
                'cRef',    sc['cref'],
                'phi',     sc['phi'],
                'psi',     sc['psi']
            )

            # Step 2: Update asphalt thickness
            g_i.gotosoil()
            g_i.setsoillayerlevel(borehole, 1, -sc['thickness'])
            g_i.setsoillayerlevel(borehole, 2, -(sc['thickness'] + 0.30))

            # Step 3: Update axle load
            g_i.gotostructures()
            g_i.Polygon_1.SurfaceLoad.setproperties('sigz', -sc['load'])
            g_i.Polygon_2.SurfaceLoad.setproperties('sigz', -sc['load'])

            # Step 4: Remesh
            g_i.gotomesh()
            g_i.mesh(0.25)

            # Step 5: Calculate
            g_i.gotostages()
            for phase in g_i.Phases:
                g_i.set(phase.ShouldCalculate, False)
            for phase in g_i.Phases:
                g_i.set(phase.ShouldCalculate, True)
            g_i.calculate()

            # Step 6: Extract result
            g_i.view(g_i.Phases[-1])
            time.sleep(3)

            s_o, g_o = new_server(HOST, PORT_OUTPUT, password=PASSWORD)
            ResultUz = g_o.ResultTypes.Soil.Uz
            soils    = g_o.SoilVolumes
            plx_vals = g_o.getresults(soils, g_o.Phases[-1], ResultUz, 'node')
            uz_min   = min(list(plx_vals))
            rut_mm   = round(abs(uz_min) * 1000, 4)

            # Step 7: Close output window
            subprocess.run(
                ['taskkill', '/F', '/IM', 'Plaxis3DOutput.exe'],
                capture_output=True
            )
            time.sleep(2)

            # Step 8: Save to CSV
            writer.writerow([
                run_num, sc['soil_type'],
                sc['e50'], sc['eoed'], sc['eur'],
                sc['cref'], sc['phi'], sc['psi'],
                sc['thickness'], sc['load'],
                rut_mm
            ])
            csvfile.flush()
            results.append(rut_mm)

            # Progress
            elapsed   = time.time() - start_time
            done      = len(completed_runs) + len(results)
            remaining = 500 - done
            avg       = elapsed / max(len(results), 1)
            eta       = (avg * remaining) / 60
            print(
                f"Run {run_num:03d}/500 | {sc['soil_type']:<18} | "
                f"E50={sc['e50']:>7,} | c={sc['cref']:5.1f} | "
                f"phi={sc['phi']:4.1f} | psi={sc['psi']:4.1f} | "
                f"t={sc['thickness']*1000:5.0f}mm | q={sc['load']:3d} | "
                f"Rut={rut_mm:.3f}mm | ETA={eta:.0f}min"
            )

        except Exception as e:
            print(f"Run {run_num:03d}/500 | {sc['soil_type']:<18} | FAILED: {str(e)[:60]}")
            failed.append(run_num)
            subprocess.run(
                ['taskkill', '/F', '/IM', 'Plaxis3DOutput.exe'],
                capture_output=True
            )
            time.sleep(2)
            writer.writerow([
                run_num, sc['soil_type'],
                sc['e50'], sc['eoed'], sc['eur'],
                sc['cref'], sc['phi'], sc['psi'],
                sc['thickness'], sc['load'],
                'ERROR'
            ])
            csvfile.flush()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = time.time() - start_time
success = len(results) + len(completed_runs)
print(f"\n{'='*65}")
print(f"  COMPLETE")
print(f"  Successful : {success}/500")
print(f"  Failed     : {len(failed)}")
print(f"  Total time : {total/3600:.2f} hours")
print(f"  CSV saved  : {OUTPUT_CSV}")
if results:
    print(f"  Rut range  : {min(results):.3f} – {max(results):.3f} mm")
print(f"{'='*65}")