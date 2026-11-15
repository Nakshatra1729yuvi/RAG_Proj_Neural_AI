import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def capture():
    print("Initializing headless Chrome for UI capture...")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,950")
    options.add_argument("--disable-gpu")
    
    out_path = Path("data/eval_results/medi_rag_app_ui.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        driver = webdriver.Chrome(options=options)
        driver.get("http://localhost:8501")
        time.sleep(5)  # Wait for Streamlit components to mount
        driver.save_screenshot(str(out_path))
        driver.quit()
        print(f"Streamlit UI screenshot successfully saved to: {out_path.resolve()}")
    except Exception as e:
        print(f"Chrome capture failed: {e}. Trying Edge...")
        from selenium.webdriver.edge.options import Options as EdgeOptions
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--window-size=1400,950")
        driver = webdriver.Edge(options=edge_options)
        driver.get("http://localhost:8501")
        time.sleep(5)
        driver.save_screenshot(str(out_path))
        driver.quit()
        print(f"Edge capture saved to: {out_path.resolve()}")

if __name__ == "__main__":
    capture()
