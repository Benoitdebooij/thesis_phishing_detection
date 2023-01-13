#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
from datetime import date, datetime
import time
import pandas as pd


# In[3]:


data = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_urls_2.csv')
data


# In[6]:


data.drop("phish_id", inplace=True, axis=1)
data.drop("phish_detail_url", inplace=True, axis=1)
data.drop("submission_time", inplace=True, axis=1)
data.drop("verified", inplace=True, axis=1)
data.drop("online", inplace=True, axis=1)
data.drop("target", inplace=True, axis=1)
data.drop("verification_time", inplace=True, axis=1)


# In[7]:


url_list = data['url'].tolist()


# In[8]:


def assemble_data(url):
    url = str(url)
    data_features = []

    if not re.match(r"^https?", url):
        url = "http://" + url

    try:
        response = requests.get(url, verify=True, timeout=4)
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        response = ""
        soup = -999


    # IPAddress
    try:
        ipaddress.ip_address(url)
        data_features.append(0)
    except:
        data_features.append(1)

    # LinkLength
    if len(url) < 60:
        data_features.append(0)
    else:
        data_features.append(1)

    # ShortiningServices
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net', url)
    if match:
        data_features.append(1)
    else:
        data_features.append(0)

    # AtSymbol
    if re.findall("@", url):
        data_features.append(1)
    else:
        data_features.append(0)

    # DoubleDash
    list = [x.start(0) for x in re.finditer('//', url)]
    if list[len(list)-1] > 6:
        data_features.append(1)
    else:
        data_features.append(0)

    # Prefix
    if re.findall(r"https?://[^\-]+-[^\-]+/", url):
        data_features.append(1)
    else:
        data_features.append(0)

    domain = re.findall(r"://([^/]+)/?", url)[0]
    if re.match(r"^www.", domain):
        domain = domain.replace("www.", "")

    # SubDomain
    if len(re.findall("\.", url)) == 2:
        data_features.append(0)
    else:
        data_features.append(1)

    # SSL
    try:
        if response.text:
            data_features.append(0)
    except:
        data_features.append(1)

    # Favicon
    if soup == -999:
        data_features.append(1)
    else:
        try:
            for head in soup.find_all('head'):
                for head.link in soup.find_all('link', href=True):
                    dots = [x.start(0)
                            for x in re.finditer('\.', head.link['href'])]
                    if url in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                        data_features.append(0)
                        raise StopIteration
                    else:
                        data_features.append(1)
                        raise StopIteration
        except StopIteration:
            pass

    # Port
    try:
        port = domain.split(":")[1]
        if port:
            data_features.append(1)
        else:
            data_features.append(0)
    except:
        data_features.append(0)

    # HTTPS
    if re.findall(r"^https://", url):
        data_features.append(0)
    else:
        data_features.append(1)

    # Request
    i = 0
    success = 0
    if soup == -999:
        data_features.append(1)
    else:
        for img in soup.find_all('img', src=True):
            dots = [x.start(0) for x in re.finditer('\.', img['src'])]
            if url in img['src'] or domain in img['src'] or len(dots) == 1:
                success = success + 1
            i = i+1

        for audio in soup.find_all('audio', src=True):
            dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
            if url in audio['src'] or domain in audio['src'] or len(dots) == 1:
                success = success + 1
            i = i+1

        for embed in soup.find_all('embed', src=True):
            dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
            if url in embed['src'] or domain in embed['src'] or len(dots) == 1:
                success = success + 1
            i = i+1

        for iframe in soup.find_all('iframe', src=True):
            dots = [x.start(0) for x in re.finditer('\.', iframe['src'])]
            if url in iframe['src'] or domain in iframe['src'] or len(dots) == 1:
                success = success + 1
            i = i+1

        try:
            percentage = success/float(i) * 100
            if percentage < 30.0:
                data_features.append(0)
            else:
                data_features.append(1)
        except:
            data_features.append(0)

    # Anchor
    percentage = 0
    i = 0
    unsafe = 0
    if soup == -999:
        data_features.append(1)
    else:
        for a in soup.find_all('a', href=True):
            # 2nd condition was 'JavaScript ::void(0)' but we put JavaScript because the space between javascript and :: might not be
                # there in the actual a['href']
            if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (url in a['href'] or domain in a['href']):
                unsafe = unsafe + 1
            i = i + 1

        try:
            percentage = unsafe / float(i) * 100
        except:
            print("")

        if percentage < 36.0:
            data_features.append(0)
        else:
            data_features.append(1)

    # TagLinks
    i = 0
    success = 0
    if soup == -999:
        data_features.append(1)
    else:
        for link in soup.find_all('link', href=True):
            dots = [x.start(0) for x in re.finditer('\.', link['href'])]
            if url in link['href'] or domain in link['href'] or len(dots) == 1:
                success = success + 1
            i = i+1

        for script in soup.find_all('script', src=True):
            dots = [x.start(0) for x in re.finditer('\.', script['src'])]
            if url in script['src'] or domain in script['src'] or len(dots) == 1:
                success = success + 1
            i = i+1
        try:
            percentage = success / float(i) * 100
        except:
            data_features.append(0)

        if percentage < 30.0:
            data_features.append(0)
        else:
            data_features.append(1)

        # SFH
        if len(soup.find_all('form', action=True))==0:
            data_features.append(1)
        else :
            for form in soup.find_all('form', action=True):
                if form['action'] == "" or form['action'] == "about:blank":
                    data_features.append(1)
                    break
                else:
                    data_features.append(0)
                    break

    # Email
    if response == "":
        data_features.append(1)
    else:
        if re.findall(r"[mail\(\)|mailto:?]", response.text):
            data_features.append(1)
        else:
            data_features.append(0)


    # Redirect
    if response == "":
        data_features.append(1)
    else:
        if len(response.history) <= 4:
            data_features.append(1)
        else:
            data_features.append(0)

    # Mouseover
    if response == "":
        data_features.append(1)
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            data_features.append(0)
        else:
            data_features.append(1)

    # RightClick
    if response == "":
        data_features.append(0)
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            data_features.append(0)
        else:
            data_features.append(1)

    # PopUpWidnow
    if response == "":
        data_features.append(1)
    else:
        if re.findall(r"alert\(", response.text):
            data_features.append(0)
        else:
            data_features.append(1)

    # Iframe
    if response == "":
        data_features.append(1)
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            data_features.append(0)
        else:
            data_features.append(1)

    rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
        "name": domain
    })

    try:
        global_rank = int(re.findall(
            r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        global_rank = -1
    # PageRank
    try:
        if global_rank > 0 and global_rank < 100000:
            data_features.append(0)
        else:
            data_features.append(1)
    except:
        data_features.append(1)


    # Links_pointing_to_page
    if response == "":
        data_features.append(1)
    else:
        number_of_links = len(re.findall(r"<a href=", response.text))
        if number_of_links == 0 or number_of_links <= 1:
            data_features.append(0)
        else:
            data_features.append(1)

    return data_features


# In[9]:


begin = 0
end = 300
collection_list = url_list[begin:end]


# In[10]:


def feature_extraction(url_list):
  data_list = []
  for url in url_list:
    data_list.append(assemble_data(url))
    print("completed", url)
  return data_list


# In[12]:


phishing_data = feature_extraction(collection_list)


# In[11]:


columns = ["HavingIP",
    'LinkLength',
    'ShorteningServices',
    'AtSymbol',
    'DoubleDash',
    'Prefix',
    'SubDomain',
    'SSL',
    'Favicon',
    'Port',
    'HTTPS',
    'Request',
    'Anchor',
    'TagLinks',
    'SFH',
    "Email",
    'Redirect',
    'Mouseover',
    'RightClick',
    'popUpWidnow',
    'Iframe',
    'PageRank',
    "LinkToPage"
]


# In[13]:


df1 = pd.DataFrame(data=phishing_data, columns=columns)


# In[ ]:


df1.to_csv("phishing_feature_extraction_300_final.csv")


# In[32]:


phishing_data_v1 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_300_final.csv')
phishing_data_v1.drop(phishing_data_v1.filter(regex="Unname"),axis=1, inplace=True)
phishing_data_v1


# In[25]:


phishing_data_v1['Links'].isnull().sum()


# In[26]:


del phishing_data_v1["Links"]


# In[27]:


phishing_data_v1.isnull().sum().sum()


# In[29]:


phishing_data_v1["Label"] = 0


# In[31]:


phishing_data_v1.to_csv("phishing_feature_extraction_300_final.csv")


# In[33]:


#This has been done multiple times for 8 different batches on different days to ensure the websites were online. Now these are merged together


# In[35]:


data_phishing_1 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_300_final.csv')
data_phishing_2 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_700_final.csv')
data_phishing_3 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_1100_final.csv')
data_phishing_4 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_1500_final.csv')
data_phishing_5 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_1800_final.csv')
data_phishing_6 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_2100_final.csv')
data_phishing_7 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_2600_final.csv')
data_phishing_8 = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\phishing_feature_extraction_3000_final.csv')


# In[36]:


frames = [data_phishing_1, data_phishing_2, data_phishing_3, data_phishing_4, data_phishing_5, data_phishing_6, data_phishing_7, data_phishing_8]
final_dataset_phishing = pd.concat(frames)


# In[37]:


final_dataset_phishing['Label'] = "1"


# In[ ]:


final_dataset_phishing.to_csv("dataset_phishing_final.csv")


# In[ ]:




