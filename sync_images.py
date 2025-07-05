import os
import shutil

# --- Configuration ---
# C# project မှ ဓာတ်ပုံ folder များ၏ တည်နေရာ
CSHARP_OFFICIAL_PHOTOS_PATH = 'F:\AMS_2\studentsphotos/'
CSHARP_TRAINING_PHOTOS_PATH = 'F:\AMS_2\TrainingImages/'

# Python AI server က အသုံးပြုမည့် folder
AI_IMAGES_PATH = './AI_Images/'


def sync_all_images():
    """f# folders မှ ဓာတ်ပုံအားလုံးကို AI folder ထဲသို့ copy ကူးပေးသည်"""

    # 1. AI folder အဟောင်းကို ဖျက်ပြီး အသစ်ပြန်ဆောက်ပါ (သန့်ရှင်းစေရန်)
    if os.path.exists(AI_IMAGES_PATH):
        shutil.rmtree(AI_IMAGES_PATH)
    os.makedirs(AI_IMAGES_PATH)

    copied_count = 0

    # 2. တရားဝင်ဓာတ်ပုံများကို copy ကူးပါ
    print(f"Copying official photos from {CSHARP_OFFICIAL_PHOTOS_PATH}...")
    for filename in os.listdir(CSHARP_OFFICIAL_PHOTOS_PATH):
        source_file = os.path.join(CSHARP_OFFICIAL_PHOTOS_PATH, filename)
        destination_file = os.path.join(AI_IMAGES_PATH, filename)
        shutil.copy2(source_file, destination_file)
        copied_count += 1

    # 3. Training ဓာတ်ပုံများကို copy ကူးပါ
    # မှတ်ချက်: လက်ရှိတွင် training photo အားလုံးကို ကူးထည့်ပါမည်။
    # Admin approval system ပါလာပါက DB ကို query လုပ်ပြီး 'Approved' ဖြစ်သောပုံများကိုသာ ကူးရမည်။
    print(f"Copying training photos from {CSHARP_TRAINING_PHOTOS_PATH}...")
    for filename in os.listdir(CSHARP_TRAINING_PHOTOS_PATH):
        source_file = os.path.join(CSHARP_TRAINING_PHOTOS_PATH, filename)
        destination_file = os.path.join(AI_IMAGES_PATH, filename)
        shutil.copy2(source_file, destination_file)
        copied_count += 1

    print(f"Sync complete. Total {copied_count} images copied to {AI_IMAGES_PATH}")


if __name__ == '__main__':
    sync_all_images()