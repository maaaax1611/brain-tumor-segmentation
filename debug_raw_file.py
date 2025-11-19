import os
import glob
import nibabel as nib
import numpy as np

def inspect_raw_data():
    # Pfad anpassen falls nötig
    root_dir = os.path.join('data', 'MICCAI_BraTS_2019_Data_Training')
    
    print(f"1. Suche in: {os.path.abspath(root_dir)}")
    
    # Suche nach irgendeinem Patienten-Ordner (HGG oder LGG)
    search_pattern = os.path.join(root_dir, '**', '*flair.nii')
    # recursive=True braucht man bei glob für **
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print("❌ FEHLER: Keine einzige .nii.gz Datei gefunden!")
        print("Bitte überprüfe die Ordnerstruktur in 'data/'.")
        return

    first_file = found_files[0]
    print(f"✅ Datei gefunden: {first_file}")
    
    try:
        # Lade das ganze Volumen
        img = nib.load(first_file)
        data = img.get_fdata()
        
        print(f"   Shape des Volumens: {data.shape}")
        print(f"   Maximaler Wert (Rohdaten): {data.max()}")
        print(f"   Mittlerer Wert (Rohdaten): {data.mean()}")
        
        if data.max() == 0:
            print("⚠️ WARNUNG: Die Datei scheint komplett leer (schwarz) zu sein.")
        else:
            print("✅ Die Datei enthält Daten!")
            
            # Suche nach einer Slice, die NICHT leer ist
            middle_slice_idx = data.shape[2] // 2
            middle_slice = data[:, :, middle_slice_idx]
            print(f"   Test Mitte (Slice {middle_slice_idx}) Max Wert: {middle_slice.max()}")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR beim Laden: {e}")

if __name__ == "__main__":
    inspect_raw_data()