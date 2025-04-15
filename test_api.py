from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import datetime
import os
import time

load_dotenv()

# Initialize the client with your bot token
slack_token = os.getenv("SLACK_TOKEN")
client = WebClient(token=slack_token)

# Tên kênh hoặc ID kênh
channel_name = "dtm_japan_it_week"
channel_id = None

# Lấy ID kênh từ tên kênh
try:
    result = client.conversations_list()
    for channel in result["channels"]:
        if channel["name"] == channel_name:
            channel_id = channel["id"]
            print(f"Đã tìm thấy ID kênh: {channel_id}")
            break
            
    if not channel_id:
        # Thử tìm trong private channels
        result = client.conversations_list(types="private_channel")
        for channel in result["channels"]:
            if channel["name"] == channel_name:
                channel_id = channel["id"]
                print(f"Đã tìm thấy ID kênh (private): {channel_id}")
                break
except SlackApiError as e:
    print(f"Lỗi khi liệt kê kênh: {e}")

# Nếu không tìm thấy ID kênh, có thể đó là direct message hoặc channel name chính là ID
if not channel_id:
    channel_id = channel_name

# Lấy danh sách tin nhắn
try:
    # Số tin nhắn muốn lấy (tối đa 1000)
    limit = 10
    
    result = client.conversations_history(
        channel=channel_id,
        limit=limit
    )
    
    messages = result["messages"]
    print(f"Đã tìm thấy {len(messages)} tin nhắn")
    
    # Lọc tin nhắn từ bot
    bot_messages = []
    all_messages = []
    
    # Hiển thị thông tin của mỗi tin nhắn
    for i, message in enumerate(messages):
        # Lấy timestamp và chuyển đổi thành thời gian có thể đọc được
        ts = float(message["ts"])
        readable_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        
        # Lấy nội dung tin nhắn
        text = message.get("text", "Không có nội dung văn bản")
        
        # Lấy thông tin người gửi nếu có
        user_id = message.get("user", "Không rõ")
        bot_id = message.get("bot_id", "Không phải bot")
        
        # Lưu thông tin tin nhắn
        all_messages.append({
            "index": i+1,
            "ts": message["ts"],
            "user_id": user_id,
            "bot_id": bot_id,
            "text": text,
            "readable_time": readable_time
        })
        
        # Nếu là tin nhắn bot
        if bot_id != "Không phải bot":
            bot_messages.append({
                "index": i+1,
                "ts": message["ts"],
                "user_id": user_id,
                "bot_id": bot_id,
                "text": text,
                "readable_time": readable_time
            })
        
        print(f"Tin nhắn {i+1}:")
        print(f"  Thời gian: {readable_time}")
        print(f"  Timestamp: {message['ts']}")
        print(f"  Người gửi: {user_id}")
        print(f"  Bot ID: {bot_id}")
        print(f"  Nội dung: {text[:100]}..." if len(text) > 100 else f"  Nội dung: {text}")
        print("----------------------")
    
    if messages:
        print("\nLựa chọn:")
        print("1. Xóa một tin nhắn cụ thể")
        print("2. Xóa nhiều tin nhắn (theo số thứ tự)")
        print("3. Xóa tất cả tin nhắn của bot")
        print("4. Thoát")
        
        choice = input("\nChọn hành động (1-4): ")
        
        if choice == "1":
            # Xóa một tin nhắn cụ thể
            message_index = int(input("Nhập số thứ tự tin nhắn muốn xóa (1, 2, 3...): ")) - 1
            if 0 <= message_index < len(messages):
                message_to_delete = messages[message_index]
                try:
                    delete_result = client.chat_delete(
                        channel=channel_id,
                        ts=message_to_delete["ts"]
                    )
                    print(f"Đã xóa tin nhắn: {delete_result}")
                except SlackApiError as e:
                    print(f"Lỗi khi xóa tin nhắn: {e}")
            else:
                print("Số thứ tự không hợp lệ!")
                
        elif choice == "2":
            # Xóa nhiều tin nhắn
            indices_input = input("Nhập các số thứ tự tin nhắn muốn xóa, cách nhau bởi dấu phẩy (ví dụ: 1,2,3): ")
            indices = [int(idx.strip()) for idx in indices_input.split(",") if idx.strip().isdigit()]
            
            for idx in indices:
                if 1 <= idx <= len(messages):
                    message_to_delete = messages[idx-1]
                    try:
                        delete_result = client.chat_delete(
                            channel=channel_id,
                            ts=message_to_delete["ts"]
                        )
                        print(f"Đã xóa tin nhắn {idx}: {delete_result}")
                        # Thêm độ trễ nhỏ để tránh vượt quá giới hạn tốc độ API
                        time.sleep(0.5)
                    except SlackApiError as e:
                        print(f"Lỗi khi xóa tin nhắn {idx}: {e}")
                else:
                    print(f"Bỏ qua số thứ tự không hợp lệ: {idx}")
                    
        elif choice == "3":
            # Xóa tất cả tin nhắn của bot
            if not bot_messages:
                print("Không tìm thấy tin nhắn nào của bot!")
            else:
                confirm = input(f"Bạn có chắc muốn xóa tất cả {len(bot_messages)} tin nhắn của bot? (y/n): ")
                if confirm.lower() == "y":
                    success_count = 0
                    for msg in bot_messages:
                        try:
                            delete_result = client.chat_delete(
                                channel=channel_id,
                                ts=msg["ts"]
                            )
                            success_count += 1
                            print(f"Đã xóa tin nhắn {msg['index']}: {delete_result}")
                            # Thêm độ trễ nhỏ để tránh vượt quá giới hạn tốc độ API
                            time.sleep(0.5)
                        except SlackApiError as e:
                            print(f"Lỗi khi xóa tin nhắn {msg['index']}: {e}")
                    
                    print(f"\nĐã xóa thành công {success_count}/{len(bot_messages)} tin nhắn của bot.")
        
        elif choice == "4":
            print("Thoát chương trình.")
        
        else:
            print("Lựa chọn không hợp lệ!")
            
except SlackApiError as e:
    print(f"Lỗi khi lấy lịch sử tin nhắn: {e}")