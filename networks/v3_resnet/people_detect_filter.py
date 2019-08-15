from PIL import Image
import os
import face_recognition

root = "../../.."
new_dataset = "./FilteredDataset/"
os.chdir(root)
# Load the jpg file into a numpy array
train = "Dataset/train/"
count = 0
saved_count = 0
for influencer in os.listdir(train):
    inf_path = os.path.join(train, influencer)
    path_i_new = os.path.join(new_dataset, train, influencer)
    for month in os.listdir(inf_path):
        mon_path = os.path.join(inf_path, month)
        path_m_new = os.path.join(path_i_new, month)
        os.makedirs(path_m_new, exist_ok=True)
        for img in os.listdir(mon_path):
            if img == "avg_likes.txt":
                continue
            else:
                count += 1
                print("train: ", count)
                path_img_new = os.path.join(path_m_new, img)
                img_path = os.path.join(mon_path, img)
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
                if len(face_locations) >= 1:
                    saved_count += 1
                    image = Image.fromarray(image)
                    image.save(path_img_new)
                
print("Found ", str(saved_count), " in ", str(count))
 
val = "Dataset/val/"
count = 0
saved_count = 0
for influencer in os.listdir(val):
    inf_path = os.path.join(val, influencer)
    path_i_new = os.path.join(new_dataset, val, influencer)
    for month in os.listdir(inf_path):
        mon_path = os.path.join(inf_path, month)
        path_m_new = os.path.join(path_i_new, month)
        os.makedirs(path_m_new, exist_ok=True)
        for img in os.listdir(mon_path):
            if img == "avg_likes.txt":
                continue
            else:
                count += 1
                print("val: ", count)
                path_img_new = os.path.join(path_m_new, img)
                img_path = os.path.join(mon_path, img)
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
                if len(face_locations) >= 1:
                    saved_count += 1
                    image = Image.fromarray(image)
                    image.save(path_img_new)

print("Found ", str(saved_count), " in ", str(count))
