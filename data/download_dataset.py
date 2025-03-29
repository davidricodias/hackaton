from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import csv

# Configurar Selenium en modo headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Iniciar el navegador
browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# URL de iStockPhoto
BASE_URL = "https://www.istockphoto.com/es/fotos/damp-mould?page={}"
PAGES = 10  # Número de páginas a scrapear

# Carpeta para guardar imágenes
IMAGES_DIR = "C:/Users/maria/Downloads/humidities/scrapper"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Lista para almacenar datos
data = []

def download_image(img_url, img_name):
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        img_path = os.path.join(IMAGES_DIR, img_name)
        with open(img_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"✅ Imagen guardada: {img_path}")
    else:
        print(f"❌ Error al descargar: {img_url}")

for page in range(1, PAGES + 1):
    url = BASE_URL.format(page)
    print(f"📄 Procesando página {page}...")
    browser.get(url)

    # Esperar hasta que las imágenes estén visibles
    try:
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img._BZ9oiIzKJoKWjkJDof_"))
        )
    except Exception as e:
        print(f"⚠️ No se encontraron imágenes en la página {page}: {e}")
        continue

    # Extraer imágenes y títulos
    items = browser.find_elements(By.CLASS_NAME, "vnBlOzAX1i2xs1n1TAo1")

    for item in items:
        try:
            title_element = item.find_element(By.CSS_SELECTOR, "[data-testid='gateway-asset-title'] a")
            title = title_element.text.strip() if title_element else "Sin título"

            img_element = item.find_element(By.CSS_SELECTOR, "img._BZ9oiIzKJoKWjkJDof_")
            img_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")

            page_url = title_element.get_attribute("href") if title_element else "Sin enlace"

            if img_url:
                img_name = img_url.split("/")[-1].split("?")[0]  # Nombre del archivo sin parámetros
                download_image(img_url, img_name)

            data.append([title, img_url, page_url])
        except Exception as e:
            print(f"⚠️ Error al extraer datos de un elemento: {e}")

# Cerrar el navegador
browser.quit()

# Guardar datos en un CSV
with open(f"{IMAGES_DIR}/scraped_images.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Título", "URL de la Imagen", "URL de la Página"])
    writer.writerows(data)

print("🎉 Scraping completado. Datos guardados en 'scraped_images.csv'")

"""import os

url = "data/real"
list_img = os.listdir(url)
for i, img in enumerate(list_img):
    img_path = os.path.join(url, img)

    ruta_antigua = os.path.join(url, img)
    ruta_nueva = os.path.join(url, f"real_{i}.jpg")

    os.rename(ruta_antigua, ruta_nueva)"""

