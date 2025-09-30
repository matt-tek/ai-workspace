import mechanicalsoup
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

URL = "https://www.marmiton.org/"
browser = mechanicalsoup.Browser()


def get_navbar_link() -> pd.DataFrame:
    desc, links_arr = [], []
    main_page = browser.get(URL)
    main_page_html = main_page.soup
    links = main_page.soup.find_all('a', href=True)

    for link in links:
        address = link['href']
        text = link.text.strip()
        desc.append(text)
        links_arr.append(address)
    data = {'description': desc, 'links': links_arr}
    df = pd.DataFrame(data=data)
    df.replace('', {'description': np.nan}, regex=True, inplace=True)
    df.dropna(inplace=True)
    return df

def get_recipe_info():
    pass

def get_data_from_recipe_url(route: str):
    url = URL.rstrip("/")
    print(f"url to fetch = {url + route}")
    page = browser.get(f"{url + route}")
    div = page.soup.find("div", class_="recipe-step-list")
    if div:
        text_content = div.get_text(strip=True, separator="\n")
    else:
        return "rien n'a été trouvé"
    return text_content
    

def get_search_bar(search: str):
    main_page = browser.get(URL)
    main_page_html = main_page.soup
    search_bar = main_page_html.select('form')[0]
    search_bar.select('input')[0]['value'] = search
    result = browser.submit(form=search_bar, url=main_page.url)
    return result

def main():
    recipe_addr = []
    search_result = get_search_bar('sardine, pain')
    recipe_div = search_result.soup.find(id='content')
    links = recipe_div.find_all('a', href=True)
    for link in links:
        address = link['href']
        text = link.text.strip()
        print(f"{text} : {address}")
        recipe_addr.append(address)
    data = get_data_from_recipe_url(recipe_addr[0])
    print(data)

if __name__ == "__main__":
    main()