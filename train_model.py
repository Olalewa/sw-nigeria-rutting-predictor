import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\DELL\Desktop\rutting_dataset_stratified.csv')
df['log_e50'] = np.log10(df['e50_kPa'])
df['log_rut'] = np.log10(df['rut_mm'])
soil_dummies  = pd.get_dummies(df['soil_type'], prefix='soil', drop_first=True)
df = pd.concat([df, soil_dummies], axis=1)

FEATURES = ['log_e50', 'cref_kPa', 'phi_deg', 'thickness_m', 'load_kPa',
            'soil_Medium_Laterite', 'soil_Soft_Laterite', 'soil_Stiff_Laterite']

X     = df[FEATURES].values
y_log = df['log_rut'].values
y_raw = df['rut_mm'].values

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(
    idx, test_size=0.2, random_state=42,
    stratify=df['soil_type'].values
)
X_train, X_test = X[idx_train], X[idx_test]
y_train         = y_log[idx_train]
y_test_raw      = y_raw[idx_test]
df_test         = df.iloc[idx_test].copy().reset_index(drop=True)

# ============================================================
# FINAL MODEL — XGBoost only
# ============================================================
print("Training final XGBoost model...")
xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    random_state=42, verbosity=0
)
xgb.fit(X_train, y_train)
print("Done ✅\n")

# ============================================================
# EVALUATE
# ============================================================
y_pred_raw = 10 ** xgb.predict(X_test)
r2   = r2_score(y_test_raw, y_pred_raw)
rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
mae  = np.mean(np.abs(y_test_raw - y_pred_raw))
mape = np.mean(np.abs((y_test_raw - y_pred_raw) / y_test_raw)) * 100
errs = np.abs((y_test_raw - y_pred_raw) / y_test_raw) * 100

kf   = KFold(n_splits=5, shuffle=True, random_state=42)
cv   = cross_val_score(xgb, X, y_log, cv=kf, scoring='r2')

print(f"{'='*55}")
print(f"  FINAL MODEL — XGBoost")
print(f"  CV R²  : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"  R²     : {r2:.4f}")
print(f"  RMSE   : {rmse:.4f} mm")
print(f"  MAE    : {mae:.4f} mm")
print(f"  MAPE   : {mape:.2f}%")
print(f"{'='*55}")

print(f"\n  Error distribution (n=100 test samples):")
for pct in [5, 10, 15, 20]:
    w = (errs < pct).sum()
    print(f"    Within {pct:2d}%: {w:3d}/100  ({w}%)")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
feat_labels = ['E50 (subgrade stiffness)', 'Cohesion (cRef)',
               'Friction angle (phi)', 'Asphalt thickness',
               'Axle load', 'Medium Laterite', 'Soft Laterite', 'Stiff Laterite']

imp        = xgb.feature_importances_
sorted_idx = np.argsort(imp)[::-1]

print(f"\n{'='*55}")
print(f"  FEATURE IMPORTANCE — XGBoost")
print(f"{'='*55}")
for rank, i in enumerate(sorted_idx, 1):
    bar = '█' * int(imp[i] * 100)
    print(f"  {rank}. {feat_labels[i]:<28} {imp[i]:.4f}  {bar}")

# ============================================================
# SPOT CHECK PER SOIL TYPE
# ============================================================
df_test['pred'] = y_pred_raw
df_test['err%'] = errs

print(f"\n{'='*65}")
print(f"  SPOT CHECK — 3 per soil type")
print(f"{'='*65}")
print(f"  {'Soil Type':<20} {'PLAXIS':>8} {'XGBoost':>8} {'Error%':>8}")
print(f"  {'-'*50}")
for soil in ['Lagos_Blue_Clay','Soft_Laterite','Medium_Laterite','Stiff_Laterite']:
    rows = df_test[df_test['soil_type'] == soil].head(3)
    for _, row in rows.iterrows():
        flag = '✅' if row['err%'] < 15 else '⚠️'
        print(f"  {soil:<20} {row['rut_mm']:>8.4f} "
              f"{row['pred']:>8.4f} {row['err%']:>7.2f}%  {flag}")
    print()

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model_path = r'C:\Users\DELL\Desktop\rutting_xgb_final.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model':    xgb,
        'features': FEATURES,
        'model_name': 'XGBoost',
        'cv_r2_mean': float(cv.mean()),
        'cv_r2_std':  float(cv.std()),
        'rmse_mm':    float(rmse),
        'mape_pct':   float(mape),
        'note':       '10**prediction gives rut_mm'
    }, f)
print(f"Final model saved: {model_path} ✅")

print(f"\n{'='*55}")
print(f"  SLIDE-READY NUMBERS")
print(f"  Model          : XGBoost (gradient boosted trees)")
print(f"  Training data  : 500 PLAXIS 3D FEA simulations")
print(f"  CV R²          : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"  RMSE           : {rmse:.4f} mm")
print(f"  MAPE           : {mape:.2f}%")
print(f"  #1 predictor   : {feat_labels[sorted_idx[0]]}")
print(f"  #2 predictor   : {feat_labels[sorted_idx[1]]}")
print(f"  #3 predictor   : {feat_labels[sorted_idx[2]]}")
print(f"  Limitation     : Medium Laterite boundary effects (~15-25% error)")
print(f"  Context        : Field rut measurement uncertainty = ±10-15%")
print(f"{'='*55}")