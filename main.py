import os
import torch
import numpy as np
import pandas as pd
import time
from multi_uav_env import MultiUavEnv

# Import PPO Agent
from ppo_agent import PPO

def train(is_latency=False):
    # --- 1. CẤU HÌNH ---
    mode_name = "WITH_LATENCY" if is_latency else "NO_LATENCY"
    log_filename = f"training_log_{mode_name}.csv"
    model_folder = f"./models/{mode_name}"
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    print("=======================================")
    print(f"BẮT ĐẦU TRAINING PPO - MODE: {mode_name}")
    print(f" -> Latency: {is_latency}")
    print(f" -> Mục tiêu: 2000 Episodes")
    print("=======================================")

    # Khởi tạo môi trường
    env = MultiUavEnv(latency_mode=is_latency)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # --- HYPERPARAMETERS ---
    MAX_EPISODES = 2000         # <--- CHẠY ĐÚNG 2000 TẬP THÌ DỪNG
    update_timestep = 2000      # Update mạng PPO sau mỗi 2000 bước (steps)
    save_model_freq = 500       # Lưu model mỗi 500 tập (Episodes)
    
    # Khởi tạo Agent PPO
    ppo_agent = PPO(state_dim, action_dim, max_action=1.0)
    
    # Biến theo dõi
    time_step = 0
    i_episode = 0
    reward_history = []

    # --- VÒNG LẶP HUẤN LUYỆN THEO EPISODE ---
    # Sửa điều kiện lặp: Chạy cho đến khi đủ 2000 tập
    while i_episode < MAX_EPISODES:
        
        state, _ = env.reset()
        current_ep_reward = 0
        done = False
        
        while not done:
            # 1. Chọn hành động (PPO)
            action, log_prob, val = ppo_agent.select_action(state)
            
            # 2. Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated

            # 3. Lưu vào buffer
            ppo_agent.store_transition(state, action, log_prob, reward, val, done)
            
            state = next_state
            current_ep_reward += reward
            time_step += 1
            
            # 4. Update mạng PPO (Dựa trên tổng số bước đi)
            if time_step % update_timestep == 0:
                ppo_agent.update()
                print(f"--> [UPDATE] Đã cập nhật mạng PPO tại bước tổng {time_step}")

            if done:
                break
        
        # Kết thúc 1 tập
        i_episode += 1
        reward_history.append(current_ep_reward)
        print(f"Mode: {mode_name} | Ep: {i_episode}/{MAX_EPISODES} | Reward: {current_ep_reward:.2f} | Total Steps: {time_step}")
        
        # 5. Lưu Log CSV
        if i_episode % 10 == 0:
            df = pd.DataFrame(reward_history, columns=["Total_Reward"])
            try:
                df.to_csv(log_filename, index=True)
            except PermissionError:
                print(f"⚠️ Cảnh báo: File CSV đang mở, không lưu được (Code vẫn chạy tiếp...)")

        # 6. Lưu Model định kỳ (theo số tập)
        if i_episode % save_model_freq == 0:
            checkpoint_path = f"{model_folder}/ppo_actor_ep{i_episode}.pth"
            torch.save(ppo_agent.actor.state_dict(), checkpoint_path)
            print(f"--> [SAVE] Đã lưu model tại tập {i_episode}: {checkpoint_path}")

    # --- KẾT THÚC ---
    print("\n=======================================")
    print(f"ĐÃ HOÀN THÀNH {MAX_EPISODES} TẬP HUẤN LUYỆN!")
    print("=======================================")
    
    # Lưu model cuối cùng
    torch.save(ppo_agent.actor.state_dict(), f"{model_folder}/ppo_actor_final.pth")
    env.close()

# --- MENU CHỌN CHẾ ĐỘ ---
if __name__ == "__main__":
    print("Chọn chế độ chạy PPO:")
    print("1. Chạy trường hợp KHÔNG có độ trễ (No Latency)")
    print("2. Chạy trường hợp CÓ độ trễ (With Latency)")
    
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    
    if choice == '1':
        train(is_latency=False)
    elif choice == '2':
        train(is_latency=True)
    else:
        print("Lựa chọn không hợp lệ! Mặc định chạy No Latency.")
        train(is_latency=False)