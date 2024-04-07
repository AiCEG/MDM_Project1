#Imports
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_review_data():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    reviews = []

    url = 'https://www.google.com/maps/search/Restaurants/@40.6919479,-74.104705,11.26z/data=!4m2!2m1!6e5?entry=ttu'
    driver.get(url)
    try:
        # Accept cookies
        WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,"//span[contains(@class, 'VfPpkd-vQzf8d') and text()='Accept all']"))).click()

        time.sleep(1)

        #scroll down the list so we have more restaurants to scrape
        scroll_duration = 30  # Duration in seconds for which you want to scroll down
        scroll_increment = 1000  # The amount to scroll on each iteration (adjust as needed)

        scrollable_div = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd")  # Replace spaces with dots for CSS selector

        start_time = time.time()
        while time.time() - start_time < scroll_duration:
            driver.execute_script('arguments[0].scrollBy(0, arguments[1]);', scrollable_div, scroll_increment)
            time.sleep(0.3)  # Adjust sleep time as needed to control scroll speed

        # scrape the urls of the restaurants
        css_selector = ".Nv2PK.THOPZb.CpccDe a.hfpxzc"  # This targets <a> tags within elements with the specified classes
        elements = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, css_selector)))

        # Extracting the 'href' attribute of each element
        urls = [element.get_attribute('href') for element in elements]

        # Printing out all URLs

        for url in urls:
            driver.get(url)
            time.sleep(1.5)
            # Click on the 'Reviews' button
            button_xpath = "//div[contains(@class, 'RWPxGd')]//button[contains(@class, 'hh2c6') and starts-with(@aria-label, 'Reviews')]"
            button = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
            button.click()

            time.sleep(1)
            #scroll down the list so we have more reviews to scrape
            scroll_duration = 45  # Duration in seconds for which you want to scroll down
            scroll_increment = 1000  # The amount to scroll on each iteration (adjust as needed)

            scrollable_div = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")  # Replace spaces with dots for CSS selector

            start_time = time.time()
            while time.time() - start_time < scroll_duration:
                driver.execute_script('arguments[0].scrollBy(0, arguments[1]);', scrollable_div, scroll_increment)
                time.sleep(0.3)  # Adjust sleep time as needed to control scroll speed

            #scrape the reviews
            review_containers = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.jJc9Ad")))

            for container in review_containers:
                try:
                    # Find the review text
                    review_text_span = container.find_element(By.CSS_SELECTOR, "div.MyEned > span")
                    review_text = review_text_span.text

                    # Find the star rating from the aria-label attribute of the span within the div.DU9Pgb
                    star_rating_span = container.find_element(By.CSS_SELECTOR, "div.DU9Pgb > span")
                    star_rating = star_rating_span.get_attribute("aria-label")

                    reviews.append({"text": review_text, "stars": star_rating})
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(e)
                    continue

    except Exception as e:
        print(e)

    finally:
        # Save the reviews as a json
        df = pd.DataFrame(reviews)
        df.to_json("reviews.json", orient="records")

    driver.quit()

if __name__ == '__main__':
    scrape_review_data()