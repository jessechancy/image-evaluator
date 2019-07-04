import ast

#write filename of return from huaying instagram crawler
file = "test1.txt"

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

#processes files
#returns total posts and ignored posts count
def preprocess_file(file_name):
    content = read_file(file_name)
    total_count = len(content)
    ignored_count = 0
    for post in content:
        #filters out videos and multi photos
        if video_filter(post) or multi_photo_filter(post):
            ignored_count += 1
            continue
        likes = post['likes']
        img_url = post['img_urls'][0]
        year, month, day = datetime_format(post['datetime'])
        
        """do something with this information"""
        
    return total_count, ignored_count