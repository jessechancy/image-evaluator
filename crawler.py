import requests
from bs4 import BeautifulSoup
import re
import json
import hashlib
import urllib
from datetime import datetime


url_list = []
name_list = []
url='http://instagram.com'
who='/selenagomez'
query = {'id': '', 'first': 12, 'hash': '', 'after': ''}

def getSession(rhx_gis, variables):
    """ Get session preconfigured with required headers & cookies. """
    #"rhx_gis:csfr_token:user_agent:variables"
    values = "%s:%s" % (
            rhx_gis,
            variables)
    x_instagram_gis = hashlib.md5(values.encode()).hexdigest()

    session = requests.Session()
    session.headers = {
            'x-instagram-gis': x_instagram_gis
            }

    return session
    
def coba_lagi(r, first):
    global session, url
    has_next = False
    # get queryID
    if first:
        soup = BeautifulSoup(r.content, "html.parser")
        script_tag = soup.find('script', text=re.compile('window\._sharedData'))
        shared_data = script_tag.string.partition('=')[-1].strip(' ;')
        result = json.loads(shared_data)
        query['after'] = result["entry_data"]["ProfilePage"][0]["graphql"]["user"]["edge_owner_to_timeline_media"]["page_info"]["end_cursor"]
        # get query_hash
        link_tag = soup.select("link[href*=ProfilePageContainer.js]") #find link with the query_hash
        js_file = session.get(url+link_tag[0]["href"]) #get the JS file
        query['id'] = result["entry_data"]["ProfilePage"][0]["graphql"]["user"]["id"]
        hash = re.search(re.compile('s.pagination},queryId:"(.*)"'), js_file.text)
        if hash:
            query['hash'] = hash.group(1)
        has_next = result["entry_data"]["ProfilePage"][0]["graphql"]["user"]["edge_owner_to_timeline_media"]["page_info"]["has_next_page"]
    else:
        result = json.loads(r.text)
        query['after'] = result["data"]["user"]["edge_owner_to_timeline_media"]["page_info"]["end_cursor"]
        has_next = result["data"]["user"]["edge_owner_to_timeline_media"]["page_info"]["has_next_page"]
    if not has_next:
        return ''
    variables = '{"id":"'+query['id']+'","first":12,"after":"'+query['after']+'"}'
    rhx_gis = ''
    session = getSession(rhx_gis, variables)
    encoded_vars = urllib.parse.quote(variables)
    next_url = 'https://www.instagram.com/graphql/query/?query_hash=%s&variables=%s' % (query['hash'], encoded_vars)
    return next_url

def get_image_links(r):
    data = json.loads(r.text)
    pics_url_list = []
    pics_name_list = []
    for i in data['data']['user']['edge_owner_to_timeline_media']['edges']:
        typename = str(i['node']['__typename'])
        if typename == "GraphImage":
            pic_url = str(i['node']['display_url'])
            # print(ctr, pic_url)
            pics_url_list.append(pic_url)
            num_likes = i['node']["edge_media_preview_like"]['count']
            num_comments = i['node']["edge_media_to_comment"]['count']
            date = i['node']['taken_at_timestamp']
            # print(datetime.fromtimestamp(date))
            name = str(num_likes)+'-'+str(num_comments)+'-'+str(date)+'.jpg'
            pics_name_list.append(name)
    return pics_url_list, pics_name_list

def download_images(url_list, name_list):
    for i, url in enumerate(url_list):
        urllib.request.urlretrieve(url,'./try/'+str(i)+'-'+name_list[i])

session = requests.Session()
# session.headers = { 'user-agent': CHROME_UA }
r = session.get(url+who)

next = coba_lagi(r, True)
while next is not '':
    response = session.get(next)
    pics_url_list, pics_name_list = get_image_links(response)
    url_list.extend(pics_url_list)
    name_list.extend(pics_name_list)
    next = coba_lagi(response, False)
print(len(url_list))
download_images(url_list, name_list)
