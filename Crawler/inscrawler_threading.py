from queue import Queue, Empty
from threading import Thread
import csv
import os
import sys
from data_preprocessing import preprocess, read_file
import urllib.request
import json

sys.path.insert(0, './instagram-crawler-master')
import crawler

proxy_list = ["134.119.214.201:8080", "75.151.213.85:8080", "202.147.173.10:80", "58.58.213.55:8888"]
users = ["selenagomez", "cristiano", "beyonce", "arianagrande"]

## Crawler Threading

def threaded_crawler():
    results = dict()
    threads = []
    q = Queue()
    num_threads = min(20, len(users))
    for i in range(len(users)):
        q.put((users[i]))
    
    def crawl_wrapper(q, n, full_post, debug):
        proxy = proxy_list[0]
        while not q.empty():
            user = q.get()
            try:
                result = crawler.get_posts_by_user(user, n, full_post, debug, proxy)
                results[user] = result
            except:
                print("user " + user + " failed")
            q.task_done()
        return True
        
    for i in range(num_threads):
        print("starting thread " + str(i))
        process = Thread(target=crawl_wrapper, args=[q, None, True, True])
        process.start()

    q.join()
    return results

## Get Images

def save_images(result):
    os.chdir("/Volumes/My Passport")
    for user in result:
        user_content = result[user]
        total_imgs = len(user_content)
        null_imgs = 0
        folder_path = "Influencers" + "/" + user
        dl_q = Queue()
        num_threads = min(total_imgs, 5)
        for i in range(total_imgs):
            post = user_content[i]
            url = post['img_url']
            if url == None:
                null_imgs += 1
            likes = post['likes']
            date = post['date']
            dl_q.put((url, likes, date, i))
        for i in range(num_threads):
            process = Thread(target=get_image, args=[dl_q, folder_path])
            process.start()
    return total_imgs, null_imgs
    
def get_image(dl_q, folder_path):
    while not dl_q.empty():
        url, likes, date, i = dl_q.get()
        path = folder_path + "/" + str(likes) + "_" + str(date) + "_" + str(i) + ".jpg"
        if url != None:
            urllib.request.urlretrieve(url, path)
        dl_q.task_done()
    return True
    
## Save into folder

def generate_folders(result):
    os.chdir("/Volumes/My Passport")
    path = "Influencers"
    for name in result:
        path_name = path + "/" + name
        try:
            os.mkdir(path_name)
        except:
            print(path_name + " already made!")
            
## Main


def inscrawler_to_file():
    print("Starting process...")
    result = threaded_crawler()
    print("Inscrawler Done...")
    total, ignored, processed_result = preprocess(result)
    print("Processed Results...")
    print(str(total) + " total results, " + str(ignored) + " ignored results.")
    generate_folders(processed_result)
    print("Generated Folders for Influencers...")
    f= open("user.txt","w+")
    f.write(json.dumps(processed_result))
    print("Saved Images to txt file")
    save_images(processed_result)
    print("Downloaded Images...")
    print("Done!")
    
def backup_download():
    os.chdir("/Volumes/My Passport")
    f = open("user.txt", "r")
    result = json.loads(f.read())
    print(save_images(result))
