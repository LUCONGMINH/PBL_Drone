import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_fix_final():
    # --- CẤU HÌNH ---
    file_no = "training_log_NO_LATENCY.csv"
    file_with = "training_log_WITH_LATENCY.csv"
    
    # Tăng độ làm mượt lên 30 để đường vẽ bớt bị gai góc do các điểm crash quá lớn
    window_size = 30 
    
    plt.figure(figsize=(10, 6))
    
    # Biến kiểm tra
    has_data = False

    # --- VẼ ĐƯỜNG XANH (NO LATENCY) ---
    if os.path.exists(file_no):
        try:
            df = pd.read_csv(file_no)
            # Lấy 2000 tập đầu
            df = df.iloc[:200] 
            
            # Làm mượt
            smooth = df["Total_Reward"].rolling(window=window_size, min_periods=1).mean()
            episodes = [(i + 1) * 10 for i in range(len(df))]
            
            plt.plot(episodes, smooth, label='No Latency (Lý tưởng)', color='blue', linewidth=2)
            has_data = True
        except: pass

    # --- VẼ ĐƯỜNG ĐỎ (WITH LATENCY) ---
    if os.path.exists(file_with):
        try:
            df = pd.read_csv(file_with)
            # Lấy 2000 tập đầu (cắt cho bằng nhau)
            df = df.iloc[:200]
            
            smooth = df["Total_Reward"].rolling(window=window_size, min_periods=1).mean()
            episodes = [(i + 1) * 10 for i in range(len(df))]
            
            plt.plot(episodes, smooth, label='With Latency (Có trễ)', color='red', linestyle='--', linewidth=2)
            has_data = True
        except: pass

    if not has_data:
        print("Không tìm thấy file dữ liệu!")
        return

    # --- TRANG TRÍ ---
    plt.title('PPO Algorithm: No Latency vs With Latency', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    
    # QUAN TRỌNG: Để tự động scale (không ép giới hạn trục Y nữa)
    plt.autoscale(enable=True, axis='y', tight=False)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Lưu ảnh
    plt.savefig('bieu_do_final_fix.png', dpi=300, bbox_inches='tight')
    print("=> Đã lưu ảnh mới: bieu_do_final_fix.png")
    plt.show()

if __name__ == "__main__":
    plot_fix_final()