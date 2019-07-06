import ast

#write filename of return from huaying instagram crawler
file = "Data/test1.txt"

## Reading File

#reads a string .txt file and returns a list of dictionaries
#printing return value may crash system
def read_file(file_name):
    f = open(file_name)
    content_string = f.read()
    content_literal = ast.literal_eval(content_string)
    return content_literal

## Filters

#returns True if it is a video
def video_filter(input_dict):
    if "views" in input_dict:
        return True
    else:
        return False

#returns True if there are multiple photos
def multi_photo_filter(input_dict):
    if "img_urls" in input_dict:
        if len(input_dict["img_urls"]) == 1:
            return False
    return True

#returns True if there are people in the image
#use facial recognition here
def people_filter(input_image):
    pass

## Processing

#input_datetime format is "YYYY-MM-DD"+"T"+"HH:MM:SS"+".000Z"
def datetime_format(input_datetime):
    year = input_datetime[:4]
    month = input_datetime[5:7]
    day = input_datetime[8:10]
    return year, month, day

#processes a txt data of influencer information
#returns total posts and ignored posts count
def preprocess(result_dict):
    influencers = result_dict
    user_filtered = dict()
    ignored_count = 0
    total_count = 0
    #loops through every user
    for user in influencers:
        user_content = []
        content = influencers[user]
        total_count += len(content)
        for post in content:
            user_post = dict()
            #filters out videos and multi photos
            if video_filter(post) or multi_photo_filter(post):
                ignored_count += 1
                continue
            user_post['likes'] = post['likes']
            user_post['img_url'] = post['img_urls'][0]
            year, month, day = datetime_format(post['datetime'])
            # in YYYYMMDD format
            user_post['date'] = year+month+day
            user_content.append(user_post)
        user_filtered[user] = user_content
    return total_count, ignored_count, user_filtered