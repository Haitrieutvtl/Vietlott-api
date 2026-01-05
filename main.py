from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import numpy as np
import random
import time
import hashlib
import os  # Nhớ import os

app = FastAPI(
    title="Vietlott Number Generator API",
    description="API gợi ý số Vietlott dựa trên thuật toán thống kê (No AI/ML)",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def calculate_frequency(df, max_num):
    all_numbers = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6']].values.flatten()
    counts = pd.Series(all_numbers).value_counts().reindex(
        range(1, max_num + 1), fill_value=0
    )
    min_v, max_v = counts.min(), counts.max()
    if max_v - min_v == 0:
        norm_freq = pd.Series(0.5, index=counts.index)
    else:
        norm_freq = (counts - min_v) / (max_v - min_v)
    return counts, norm_freq

def calculate_gap(df, max_num):
    total = len(df)
    gap = {}
    for num in range(1, max_num + 1):
        found = False
        for i in range(total - 1, -1, -1):
            row = df.iloc[i][['Num1','Num2','Num3','Num4','Num5','Num6']].values
            if num in row:
                gap[num] = (total - 1) - i
                found = True
                break
        if not found:
            gap[num] = total
    gap = pd.Series(gap)
    min_g, max_g = gap.min(), gap.max()
    norm_gap = (gap - min_g) / (max_g - min_g) if max_g > min_g else pd.Series(0.5, index=gap.index)
    return norm_gap

def calculate_hot_cold(raw_counts):
    max_v = raw_counts.max()
    if max_v == 0:
        return pd.Series(0.5, index=raw_counts.index)
    score = raw_counts / max_v
    return score ** 0.7 

def assign_weight(mode):
    if mode == 'safe':
        return 0.5, 0.3, 0.2
    elif mode == 'risky':
        return 0.2, 0.2, 0.6
    else: # balanced
        return 0.33, 0.33, 0.34

def split_number_tiers(final_scores):
    scores = final_scores.sort_values(ascending=False)
    n = len(scores)
    hot = scores.iloc[:int(n * 0.3)]
    warm = scores.iloc[int(n * 0.3):int(n * 0.7)]
    cold = scores.iloc[int(n * 0.7):]
    return hot, warm, cold

def get_set_composition(mode):
    if mode == 'safe':
        return {'hot': 3, 'warm': 2, 'cold': 1}
    elif mode == 'risky':
        return {'hot': 1, 'warm': 2, 'cold': 3}
    else:
        return {'hot': 2, 'warm': 2, 'cold': 2}

def weighted_pick(nums, weights, k):
    if weights.sum() == 0:
        probs = None 
    else:
        probs = weights / weights.sum()
    return np.random.choice(nums, size=k, replace=False, p=probs)

def generate_number_sets_v2(final_scores, risk_mode, n_sets, seed=None):
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))

    hot, warm, cold = split_number_tiers(final_scores)
    comp = get_set_composition(risk_mode)

    results = []
    for _ in range(n_sets):
        chosen = []
        chosen += list(weighted_pick(hot.index.values, hot.values, comp['hot']))
        chosen += list(weighted_pick(warm.index.values, warm.values, comp['warm']))
        chosen += list(weighted_pick(cold.index.values, cold.values, comp['cold']))
        results.append(sorted(chosen))
    return results

# ==============================================================================
# API ENDPOINT
# ==============================================================================

@app.get("/generate")
def generate_numbers(
    game: str = Query(..., regex="^(645|655)$", description="Loại vé: 645 hoặc 655"),
    mode: str = Query(..., regex="^(safe|balanced|risky)$", description="Chiến thuật: safe, balanced, risky"),
    sets: int = Query(..., gt=0, le=50, description="Số lượng bộ số cần tạo (1-50)"),
    seed: Optional[str] = Query(None, description="Seed cá nhân hóa (tùy chọn)")
):
    try:
        # 1. Config Game & Load Data
        DATA_FILES = {
            "645": "data/vietlott_645.csv",
            "655": "data/vietlott_655.csv"
        }
        
        if game == "655":
            max_num = 55
            game_label = "6/55"
        else:
            max_num = 45
            game_label = "6/45"

        # --- ĐOẠN CODE ĐỌC CSV ĐƯỢC CHUYỂN VÀO ĐÂY ---
        file_path = DATA_FILES.get(game)
        
        if not file_path or not os.path.exists(file_path):
            # Nếu chạy trên máy local chưa có file data, fallback về dữ liệu giả để không crash
            # Bạn có thể xóa đoạn fallback này nếu chắc chắn đã upload file
            print(f"[WARN] Không tìm thấy file {file_path}, dùng Mock Data.")
            df = generate_dummy_data_fallback(max_num) 
        else:
            # Đọc file CSV thật
            df = pd.read_csv(file_path)
            # Đảm bảo chỉ lấy đúng 6 cột số
            required_cols = ['Num1','Num2','Num3','Num4','Num5','Num6']
            if not all(col in df.columns for col in required_cols):
                 raise HTTPException(status_code=500, detail=f"File CSV thiếu cột dữ liệu. Cần: {required_cols}")
            df = df[required_cols]

        # 2. Xử lý Seed
        seed_int = None
        if seed:
            seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)

        # 3. Pipeline thuật toán
        raw_counts, freq = calculate_frequency(df, max_num)
        gap = calculate_gap(df, max_num)
        hc = calculate_hot_cold(raw_counts)

        w1, w2, w3 = assign_weight(mode)
        
        # Tính điểm tổng hợp
        final_scores = (w1 * freq) + (w2 * hc) + (w3 * gap)
        
        if seed_int:
            np.random.seed(seed_int) 
        
        final_scores += np.random.normal(0, 0.03, size=len(final_scores))
        final_scores = final_scores.clip(lower=0.001)

        # 4. Sinh kết quả
        raw_results = generate_number_sets_v2(final_scores, mode, sets, seed_int)

        clean_results = []
        for subset in raw_results:
            clean_results.append([int(x) for x in subset])

        return {
            "game": game_label,
            "mode": mode,
            "results": clean_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Hàm dự phòng nếu không tìm thấy file csv (để tránh lỗi Server Error 500 khi test)
def generate_dummy_data_fallback(max_num, rows=100):
    dummy = []
    for _ in range(rows):
        dummy.append(sorted(random.sample(range(1, max_num + 1), 6)))
    return pd.DataFrame(dummy, columns=['Num1','Num2','Num3','Num4','Num5','Num6'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
