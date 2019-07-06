from queue import Queue, Empty
from threading import Thread
import csv
import os
import sys
from data_preprocessing import preprocess
import urllib.request

sys.path.insert(0, './instagram-crawler-master')
import crawler


users = ["selenagomez", "cristiano", "leomessi", "beyonce", "arianagrande", "kyliejenner"]

## Crawler Threading

def threaded_crawler():
    results = dict()
    threads = []
    q = Queue()
    num_threads = min(20, len(users))
    for i in range(len(users)):
        q.put((users[i]))
    
    def crawl_wrapper(q, n, full_post, debug):
        while not q.empty():
            user = q.get()
            try:
                result = crawler.get_posts_by_user(user, n, full_post, debug)
                results[user] = result
            except:
                result[user] = None
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
        folder_path = "Influencers" + "/" + user
        dl_q = Queue()
        num_threads = min(20, len(user_content))
        for i in range(len(user_content)):
            post = user_content[i]
            url = post['img_url']
            likes = post['likes']
            date = post['date']
            dl_q.put((url, likes, date, i))
        for i in range(num_threads):
            process = Thread(target=get_image, args=[dl_q, folder_path])
            process.start()

def get_image(dl_q, folder_path):
    while not dl_q.empty():
        url, likes, date, i = dl_q.get()
        path = folder_path + "/" + str(likes) + "_" + str(date) + "_" + str(i) + ".jpg"
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
    save_images(processed_result)
    print("Downloaded Images...")
    print("Done!")
    