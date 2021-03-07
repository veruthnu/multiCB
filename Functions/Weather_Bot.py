from selenium import webdriver
import re
from bs4 import BeautifulSoup
import unicodedata
import time


def weather_information():

    try:
       
        driver = webdriver.Chrome("chromedriver.exe")
        time.sleep(1)
        # 네이버 날씨정보 

        url = 'https://weather.naver.com/today'
        driver.get(url)
        time.sleep(1)

        # 특정지역 날씨 검색을 위해 검색창 클릭

        searchingarea = driver.find_element_by_css_selector('#header > div.gnb_area > div > div.button_group > button')
        searchingarea.click()
        time.sleep(1)
        # 지역입력

        area = input('지역을 입력하세요(ex 온수동, 청천동): ')
        time.sleep(0.5)
        input_area = driver.find_element_by_css_selector('#_idSearchInput')
        time.sleep(0.5)
        input_area.send_keys(area)
        time.sleep(0.5)
        input_area.submit()
        time.sleep(0.5)

        #input_finalarea = driver.find_element_by_css_selector('#_idsearchResultContainer > ul > li > a')
        #input_finalarea.click()
        driver.find_element_by_css_selector('#_idsearchResultContainer > ul > li > a').click()
        time.sleep(0.5)
        # 현재페이지의 정보를 변수에 저장

        html = driver.page_source
        time.sleep(0.5)
        soup = BeautifulSoup(html, 'lxml')

        # 종합날씨정보

        weather = soup.select('#content > div > div.card.card_today > div.today_weather > div.weather_area > p')[0].text.split('요')

        degree = soup.select('#nation > div > div.nation_map > a.zone.z1 > span.text > em')[0].text


        print('오늘 {}의 날씨는 {}'.format(area,weather[1].strip()))
        #if weather[1].strip() == '맑음':
            # print(' ')
        print(weather[0].strip()+'요'+'(현재기온 {})'.format(degree))

        driver.close()
        
    except:

        driver.close()

def Particulate_Matter():

    try:
       
        driver = webdriver.Chrome("chromedriver.exe")
        time.sleep(1)
        # 네이버 날씨정보 

        url = 'https://weather.naver.com/today'
        driver.get(url)
        time.sleep(1)

        # 특정지역 날씨 검색을 위해 검색창 클릭

        searchingarea = driver.find_element_by_css_selector('#header > div.gnb_area > div > div.button_group > button')
        searchingarea.click()
        time.sleep(1)
        # 지역입력

        area = input('지역을 입력하세요(ex 온수동, 청천동): ')
        time.sleep(0.5)
        input_area = driver.find_element_by_css_selector('#_idSearchInput')
        time.sleep(0.5)
        input_area.send_keys(area)
        time.sleep(0.5)
        input_area.submit()
        time.sleep(0.5)

        #input_finalarea = driver.find_element_by_css_selector('#_idsearchResultContainer > ul > li > a')
        #input_finalarea.click()
        driver.find_element_by_css_selector('#_idsearchResultContainer > ul > li > a').click()
        time.sleep(0.5)
        # 현재페이지의 정보를 변수에 저장

        html = driver.page_source
        time.sleep(0.5)
        soup = BeautifulSoup(html, 'lxml')

        # 미세먼지 정보 저장
        # #content > div > div.card.card_today > div.today_weather > ul
        PM = soup.select('#content > div > div.card.card_today > div.today_weather > ul')
        PM = PM[0].text.strip().split('\n\n\n\n')
        pm_normal = PM[0].strip()
        pm_micro = PM[1].strip()
        pm_uv = PM[2].strip()

        print("오늘 {}의 미세먼지 정보입니다".format(area))
        print(pm_normal)
        print('--------')
        print(pm_normal)
        print('--------')
        print(pm_uv)
        
        driver.close()
        
    except:

        driver.close()



