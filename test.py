import numpy as np
import torch
import time
import os
from multi_uav_env import MultiUavEnv
from td3_agent import TD3

def evaluate():
    # 1. Cấu hình môi trường
    # Lúc test thì nên test ở cả 2 chế độ để quay video so sánh
    env = MultiUavEnv(latency_mode=False) 
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # 2. Khởi tạo Agent
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    policy = TD3(**kwargs)

    # --- 3. LOAD MODEL ĐÃ TRAIN ---
    # Em hãy vào thư mục 'models', xem file nào có số to nhất (ví dụ 2000)
    # Copy tên file đó paste vào dòng dưới đây:
    model_path = "./models/td3_actor_NO_LATENCY_2000.pth" 
    
    if os.path.exists(model_path):
        policy.actor.load_state_dict(torch.load(model_path))
        print(f"Đã load model thành công: {model_path}")
    else:
        print(f"KHÔNG TÌM THẤY FILE: {model_path}")
        print("Hãy kiểm tra lại tên file trong thư mục models!")
        return

    print("---------------------------------------")
    print("BẮT ĐẦU BAY THỬ NGHIỆM (TESTING MODE)")
    print("Nhấn Ctrl+C để dừng.")
    print("---------------------------------------")

    # Chạy thử 5 lần (5 Episodes)
    for episode in range(5):
        state, _ = env.reset(seed=0)
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Chọn hành động từ Model đã học
            # Lưu ý: KHÔNG cộng thêm noise (np.random) nữa
            action = policy.select_action(np.array(state))
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            step_count += 1
            
            print(f"Step: {step_count} | Action: {action[:2]}...") # In thử hành động

        print(f"KẾT THÚC Episode {episode+1} | Tổng điểm: {episode_reward:.2f}")
        time.sleep(2) # Nghỉ chút trước khi sang ván mới

if __name__ == "__main__":
    evaluate()