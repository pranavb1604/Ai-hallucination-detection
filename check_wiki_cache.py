import os
import joblib

cache_path = "data/processed/wiki_cache.pkl"
print("Cache exists:", os.path.exists(cache_path))
if os.path.exists(cache_path):
    print("File size:", os.path.getsize(cache_path), "bytes")
    try:
        data = joblib.load(cache_path)
        print("Data type:", type(data))
        print("Number of entries:", len(data))
        if len(data) > 0:
            print("First 3 keys:")
            for k in list(data.keys())[:3]:
                print(f"  - {k}: {str(data[k])[:100]}...")
    except Exception as e:
        print("Error loading:", e)
