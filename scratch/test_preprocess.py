import pandas as pd
import numpy as np

# Load
meta = pd.read_csv('data/raw/gloria/GLORIA_meta_and_lab.csv')
qc = pd.read_csv('data/raw/gloria/GLORIA_qc_flags.csv').dropna(subset=['GLORIA_ID'])
rrs = pd.read_csv('data/raw/gloria/GLORIA_Rrs.csv')

# 1. Merge
df = meta.merge(qc, on='GLORIA_ID').merge(rrs, on='GLORIA_ID')

# 2. QC Filter: Remove failed quality
df_clean = df[df['Flagged'] == 0].copy()

# Preserve valid Chla OR valid TSS independently
df_clean = df_clean[df_clean['Chla'].notna() | df_clean['TSS'].notna()].copy()

# Add target flags
df_clean['has_valid_chla'] = df_clean['Chla'].notna() & (df_clean['Chla'] >= 0)
df_clean['has_valid_tss'] = df_clean['TSS'].notna() & (df_clean['TSS'] >= 0)

# Sentinel-2 MSI Band Centers
s2_bands = {
    'B1': 443,  # Coastal aerosol
    'B2': 490,  # Blue
    'B3': 560,  # Green
    'B4': 665,  # Red
    'B5': 705,  # Red Edge 1
    'B6': 740,  # Red Edge 2
    'B7': 783,  # Red Edge 3
    'B8': 842,  # NIR broad
    'B8A': 865  # NIR narrow
}

# 3. Extract Sentinel-2 nominal center bands
for b_name, wl in s2_bands.items():
    df_clean[f'Rrs_{b_name}'] = df_clean[f'Rrs_{wl}']

# 4. Reflectance conversion: rho = pi * Rrs
for b_name in s2_bands.keys():
    df_clean[f'rho_{b_name}'] = np.pi * df_clean[f'Rrs_{b_name}']

# 5. Calculate three spectral indices:
# Index 1: NDCI (Normalized Difference Chlorophyll Index) - Mishra & Mishra 2012
# Uses B5 (705 nm) and B4 (665 nm)
df_clean['NDCI'] = (df_clean['rho_B5'] - df_clean['rho_B4']) / (df_clean['rho_B5'] + df_clean['rho_B4'])

# Index 2: NDTI (Normalized Difference Turbidity Index) - Lacaux et al. 2007
# Uses B4 (665 nm) and B3 (560 nm)
df_clean['NDTI'] = (df_clean['rho_B4'] - df_clean['rho_B3']) / (df_clean['rho_B4'] + df_clean['rho_B3'])

# Index 3: NDWI (Normalized Difference Water Index) - McFeeters 1996
# Uses B3 (560 nm) and B8 (842 nm) or B8A (865 nm)
df_clean['NDWI_B8'] = (df_clean['rho_B3'] - df_clean['rho_B8']) / (df_clean['rho_B3'] + df_clean['rho_B8'])
df_clean['NDWI_B8A'] = (df_clean['rho_B3'] - df_clean['rho_B8A']) / (df_clean['rho_B3'] + df_clean['rho_B8A'])

# Also 2BDA Chlorophyll ratio: rho_705 / rho_665
df_clean['2BDA'] = df_clean['rho_B5'] / df_clean['rho_B4']

print('Cleaned dataset shape:', df_clean.shape)
print('Total samples:', len(df_clean))
print('Samples with valid Chla:', df_clean['has_valid_chla'].sum())
print('Samples with valid TSS:', df_clean['has_valid_tss'].sum())
print('Samples with both valid:', (df_clean['has_valid_chla'] & df_clean['has_valid_tss']).sum())

print('\nCorrelation with targets:')
corr_cols = ['NDCI', 'NDTI', '2BDA', 'NDWI_B8', 'rho_B3', 'rho_B4', 'rho_B5']
print('Correlation with Chla (log1p):')
log_chla = np.log1p(df_clean.loc[df_clean['has_valid_chla'], 'Chla'])
print(df_clean.loc[df_clean['has_valid_chla'], corr_cols].apply(lambda s: s.corr(log_chla)).round(3))

print('\nCorrelation with TSS (log1p):')
log_tss = np.log1p(df_clean.loc[df_clean['has_valid_tss'], 'TSS'])
print(df_clean.loc[df_clean['has_valid_tss'], corr_cols].apply(lambda s: s.corr(log_tss)).round(3))
