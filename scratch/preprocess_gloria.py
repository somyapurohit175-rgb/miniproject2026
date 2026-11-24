import os
import pandas as pd
import numpy as np

def run_preprocessing():
    print("Loading raw GLORIA files...")
    meta_path = 'data/raw/gloria/GLORIA_meta_and_lab.csv'
    qc_path = 'data/raw/gloria/GLORIA_qc_flags.csv'
    rrs_path = 'data/raw/gloria/GLORIA_Rrs.csv'

    meta = pd.read_csv(meta_path)
    qc = pd.read_csv(qc_path).dropna(subset=['GLORIA_ID'])
    rrs = pd.read_csv(rrs_path)

    print(f"Loaded: meta={meta.shape}, qc={qc.shape}, rrs={rrs.shape}")

    # 1. Merge datasets using GLORIA_ID
    df = meta.merge(qc, on='GLORIA_ID').merge(rrs, on='GLORIA_ID')
    print(f"Merged dataset shape: {df.shape}")

    # 2. Remove failed-quality observations (Flagged == 0)
    df_clean = df[df['Flagged'] == 0].copy()
    print(f"After quality control (Flagged == 0): {df_clean.shape[0]} observations")

    # Target Preservation: preserve valid Chla OR valid TSS independently
    valid_chla = df_clean['Chla'].notna() & (df_clean['Chla'] >= 0)
    valid_tss = df_clean['TSS'].notna() & (df_clean['TSS'] >= 0)
    df_clean = df_clean[valid_chla | valid_tss].copy()

    df_clean['has_valid_chla'] = valid_chla
    df_clean['has_valid_tss'] = valid_tss
    df_clean['has_both_targets'] = valid_chla & valid_tss

    print(f"Preserved observations (valid Chla OR valid TSS): {len(df_clean)}")
    print(f"  - Valid Chla: {df_clean['has_valid_chla'].sum()}")
    print(f"  - Valid TSS: {df_clean['has_valid_tss'].sum()}")
    print(f"  - Both Valid: {df_clean['has_both_targets'].sum()}")

    # 3. Sentinel-2 Band Extraction (Nominal center and SRF-equivalent FWHM window average)
    s2_bands = {
        'B1': {'center': 443, 'fwhm': (433, 453), 'desc': 'Coastal aerosol'},
        'B2': {'center': 490, 'fwhm': (458, 523), 'desc': 'Blue'},
        'B3': {'center': 560, 'fwhm': (543, 578), 'desc': 'Green'},
        'B4': {'center': 665, 'fwhm': (650, 680), 'desc': 'Red'},
        'B5': {'center': 705, 'fwhm': (698, 713), 'desc': 'Red Edge 1'},
        'B6': {'center': 740, 'fwhm': (733, 748), 'desc': 'Red Edge 2'},
        'B7': {'center': 783, 'fwhm': (773, 793), 'desc': 'Red Edge 3'},
        'B8': {'center': 842, 'fwhm': (785, 899), 'desc': 'NIR broad'},
        'B8A': {'center': 865, 'fwhm': (855, 875), 'desc': 'NIR narrow'},
    }

    for b, info in s2_bands.items():
        center_col = f"Rrs_{info['center']}"
        df_clean[f"Rrs_{b}"] = df_clean[center_col]

        w1, w2 = info['fwhm']
        cols_in_window = [f"Rrs_{w}" for w in range(w1, w2 + 1) if f"Rrs_{w}" in df_clean.columns]
        df_clean[f"Rrs_{b}_srf"] = df_clean[cols_in_window].mean(axis=1)

    # 4. Reflectance conversion: rho = pi * Rrs
    for b in s2_bands.keys():
        df_clean[f"rho_{b}"] = np.pi * df_clean[f"Rrs_{b}"]
        df_clean[f"rho_{b}_srf"] = np.pi * df_clean[f"Rrs_{b}_srf"]

    # 5. Calculate three spectral indices:
    # Index 1: NDCI (Normalized Difference Chlorophyll Index) - Mishra & Mishra 2012
    # Specifically tuned for Sentinel-2 Red Edge (B5) and Red (B4)
    df_clean['NDCI'] = (df_clean['rho_B5'] - df_clean['rho_B4']) / (df_clean['rho_B5'] + df_clean['rho_B4'] + 1e-8)
    df_clean['NDCI_srf'] = (df_clean['rho_B5_srf'] - df_clean['rho_B4_srf']) / (df_clean['rho_B5_srf'] + df_clean['rho_B4_srf'] + 1e-8)

    # Index 2: NDTI (Normalized Difference Turbidity Index) - Lacaux et al. 2007
    # For suspended sediments and turbidity using Red (B4) and Green (B3)
    df_clean['NDTI'] = (df_clean['rho_B4'] - df_clean['rho_B3']) / (df_clean['rho_B4'] + df_clean['rho_B3'] + 1e-8)
    df_clean['NDTI_srf'] = (df_clean['rho_B4_srf'] - df_clean['rho_B3_srf']) / (df_clean['rho_B4_srf'] + df_clean['rho_B3_srf'] + 1e-8)

    # Index 3: NDWI (Normalized Difference Water Index) - McFeeters 1996
    # Water delineation using Green (B3) and NIR (B8A / B8)
    df_clean['NDWI_B8A'] = (df_clean['rho_B3'] - df_clean['rho_B8A']) / (df_clean['rho_B3'] + df_clean['rho_B8A'] + 1e-8)
    df_clean['NDWI_B8'] = (df_clean['rho_B3'] - df_clean['rho_B8']) / (df_clean['rho_B3'] + df_clean['rho_B8'] + 1e-8)

    # Auxiliary Chlorophyll ratio: 2BDA (rho_705 / rho_665)
    df_clean['2BDA'] = df_clean['rho_B5'] / (df_clean['rho_B4'] + 1e-8)

    # Export structured clean dataset
    meta_cols = [
        'GLORIA_ID', 'Dataset_ID', 'Site_name', 'Country', 'Country_code',
        'Latitude', 'Longitude', 'Date_Time_UTC', 'Water_body_type', 'Water_type'
    ]
    target_cols = [
        'Chla', 'TSS', 'Turbidity', 'Secchi_depth', 'aCDOM440',
        'has_valid_chla', 'has_valid_tss', 'has_both_targets'
    ]
    rrs_s2_cols = [f"Rrs_{b}" for b in s2_bands.keys()]
    rho_s2_cols = [f"rho_{b}" for b in s2_bands.keys()]
    rrs_srf_cols = [f"Rrs_{b}_srf" for b in s2_bands.keys()]
    rho_srf_cols = [f"rho_{b}_srf" for b in s2_bands.keys()]
    index_cols = ['NDCI', 'NDCI_srf', 'NDTI', 'NDTI_srf', 'NDWI_B8A', 'NDWI_B8', '2BDA']

    final_cols = meta_cols + target_cols + rrs_s2_cols + rho_s2_cols + rrs_srf_cols + rho_srf_cols + index_cols
    export_df = df_clean[final_cols].copy()

    out_path = 'data/processed/gloria_cleaned.csv'
    os.makedirs('data/processed', exist_ok=True)
    export_df.to_csv(out_path, index=False)
    print(f"Exported clean dataset to {out_path} with shape {export_df.shape}")
    print(f"File size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

    # Correlations
    print("\nPearson Correlation with log1p(Chla):")
    valid_chla_mask = export_df['has_valid_chla']
    log_chla = np.log1p(export_df.loc[valid_chla_mask, 'Chla'])
    for col in ['NDCI', 'NDTI', '2BDA', 'rho_B3', 'rho_B4', 'rho_B5']:
        corr = export_df.loc[valid_chla_mask, col].corr(log_chla)
        print(f"  {col}: {corr:.3f}")

    print("\nPearson Correlation with log1p(TSS):")
    valid_tss_mask = export_df['has_valid_tss']
    log_tss = np.log1p(export_df.loc[valid_tss_mask, 'TSS'])
    for col in ['NDCI', 'NDTI', '2BDA', 'rho_B3', 'rho_B4', 'rho_B5']:
        corr = export_df.loc[valid_tss_mask, col].corr(log_tss)
        print(f"  {col}: {corr:.3f}")

if __name__ == '__main__':
    run_preprocessing()
