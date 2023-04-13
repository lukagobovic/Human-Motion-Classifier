
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome('C:\\Users\\Luka\\Documents\\chromedriver.exe')
driver.get('http://192.168.2.69/')

# Wait for element to load
element_present = EC.presence_of_element_located((By.ID, 'element6'))
WebDriverWait(driver, timeout=10).until(element_present)

# Switch to iframe
iframe = driver.find_element_by_id('appframe')
driver.switch_to.frame(iframe)

# Extract value from element
value_x = float(driver.find_element_by_id('element6').text.strip())

# Switch back to main frame
driver.switch_to.default_content()

# Close webdriver
driver.quit()