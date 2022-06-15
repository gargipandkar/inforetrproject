import requests
from bs4 import BeautifulSoup

import json

LINKS = []
with open("links.txt", "r") as f:
    LINKS = f.read().splitlines(keepends=False)
f.close()

CONTENT_AREAS = ['featured-content--wrapper', 
                 'main-content column',  
                 'signpost--contents', 
                 'content-listing parbase section', 
                 'sidebar-content parbase section']

def extract_text(text_w_tags):
    replace_chars = ['\n', '\t', '\r']
    text=""
    for t in text_w_tags:
        clean_text = t.text
        for ch in replace_chars:
            clean_text = clean_text.replace(ch, ' ').strip()
        clean_text = clean_text.strip().rstrip('.')
        if len(clean_text)>0:
            text += f" {clean_text}."
        
    return text

def get_content_text(soup, content_areas=CONTENT_AREAS):
    '''Returns single text block for a webpage'''
    for linebreak in soup.find_all('br'):
        linebreak.extract()
    
    text_w_tags = []
        
    for area in content_areas:
        area_results = soup.find_all('div', {'class': area})
        results = []
        recurse = True
        if area_results is not None:
            for occ in area_results:
                if area=='signpost--contents': recurse = False
                results += occ.find_all(['p', 'h5', 'small'], recursive=recurse)
        if results != []:
            text_w_tags += results
            
    text = extract_text(text_w_tags)
    text = text.strip()
    return text

def save_as_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    file.close()

for url in LINKS:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    title = soup.find('h1').text
    data = get_content_text(soup)
    if len(data)<1:
        continue
    page_json = {'title': title, 
             'data': data,
             'uri': url} 
    
    path = "./documents/" + url.split("/")[-2] + ".json"
    save_as_json(page_json, path)
    

