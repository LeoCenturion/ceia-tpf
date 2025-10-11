import pandas as pd
import os
import glob
from tqdm import tqdm # Para seguimiento de progreso

# --- Configuration ---
# Directorio donde se encuentran los 2000+ archivos CSV.
DATA_DIR = "/home/leocenturion/Documents/postgrados/ia/tp-final/Tp Final/data/binance/python/data/spot/daily/klines/BTCUSDT/1m"
FILE_PATTERN = "*.csv"
OUTPUT_FILE_NAME = "BTCUSDT_consolidated_klines.csv"
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, OUTPUT_FILE_NAME)

# Define el encabezado esperado (usado para asignar nombres a las columnas)
COLUMNS = ['unix', 'open', 'high', 'low', 'close', 'Volume BTC', 'close_time', 'Volume USDT', 'tradeCount', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

# --- Processing Functions ---
def load_single_csv(filepath):
    """Carga un solo CSV, omite la fila de encabezado redundante y prepara la columna de fecha."""
    # 1. Cargar el CSV: Se usa skiprows=1 para saltar la fila de encabezado del archivo original
    df = pd.read_csv(filepath, header=None)
    
    # 2. Asignar los nombres de columna correctos (definidos en COLUMNS)
    df.columns = COLUMNS
    
    # 3. Convertir la columna 'date' a datetime y establecer como índice
    # Use 'unix' timestamp to create the 'date' column for a reliable index,
    # mirroring the logic from fetch_historical_data.
    df['date'] = pd.to_datetime(df['unix'], unit='us')
    return df.set_index('date')

# --- Main Execution ---

# 1. Usar glob para encontrar todos los archivos CSV
all_files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
print(f"Found {len(all_files)} CSV files to process.")

# 2. Leer todos los archivos en una lista de DataFrames
list_of_dfs = []
for file in tqdm(all_files, desc="Loading and cleaning files"):
    try:
        # Nota: Se añade una verificación para evitar leer el archivo de salida consolidado si ya existe
        if os.path.basename(file) != OUTPUT_FILE_NAME:
            df_chunk = load_single_csv(file)
            list_of_dfs.append(df_chunk)
    except Exception as e:
        print(f"Skipping file {file} due to error: {e}")
        continue

# 3. Concatenar y finalizar el DataFrame
if list_of_dfs:
    # Concatenar todos los DataFrames
    final_df = pd.concat(list_of_dfs)
    
    # Ordenar por índice (fecha) para garantizar el orden cronológico
    final_df = final_df.sort_index()

    # Opcional: Mostrar información del DataFrame final
    print(f"\n--- Final Data Summary ---")
    print(f"Total rows after concat: {len(final_df)}")
    print(f"Number of unique timestamps: {len(final_df.index.unique())}")
    final_df.info(memory_usage='deep')

    # 4. Guardar el DataFrame consolidado en un solo archivo CSV
    print(f"\nSaving consolidated data to: {OUTPUT_FILE_PATH}")
    # index=True asegura que la columna 'date' (el índice) se guarde en el CSV
    final_df.to_csv(OUTPUT_FILE_PATH, index=True) 

    print("Consolidation complete.")
else:
    print("No data loaded.")
