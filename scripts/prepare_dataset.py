import sys
from pathlib import Path
import urllib.request
import re
from bs4 import BeautifulSoup
import pymupdf as fitz
import docx
from docx.shared import Pt, Inches

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from backend.pipeline import MediRAGPipeline

SOURCES = [
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "title": "WHO Clinical Fact Sheet: Hypertension",
        "filename": "who_hypertension_guideline.pdf",
        "type": "pdf"
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "title": "WHO Clinical Fact Sheet: Diabetes",
        "filename": "who_diabetes_factsheet.docx",
        "type": "docx"
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/asthma",
        "title": "WHO Clinical Fact Sheet: Asthma",
        "filename": "who_asthma_guidance.pdf",
        "type": "pdf"
    },
    {
        "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
        "title": "WHO Fact Sheet: Cardiovascular Diseases (CVDs)",
        "filename": "who_cardiovascular_diseases.txt",
        "type": "txt"
    }
]

def fetch_and_clean_who_article(url: str, fallback_title: str) -> dict:
    print(f"Fetching: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            html = res.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching {url}: {e}. Using cached clinical knowledge representation.")
        return get_fallback_article(fallback_title)

    soup = BeautifulSoup(html, "html.parser")
    
    # Title
    title_elem = soup.find("h1")
    title = title_elem.get_text().strip() if title_elem else fallback_title

    # Article content
    content_div = soup.find("article") or soup.find("div", class_="sf-content-block") or soup.find("main")
    if not content_div:
        content_div = soup.body

    sections = []
    current_heading = "Overview"
    current_paragraphs = []

    for elem in content_div.find_all(["h2", "h3", "p", "ul"]):
        if elem.name in ["h2", "h3"]:
            h_text = elem.get_text().strip()
            if h_text and len(h_text) < 100:
                if current_paragraphs:
                    sections.append((current_heading, "\n\n".join(current_paragraphs)))
                    current_paragraphs = []
                current_heading = h_text
        elif elem.name == "p":
            p_text = elem.get_text().strip()
            if p_text and len(p_text) > 20:
                current_paragraphs.append(p_text)
        elif elem.name == "ul":
            items = [li.get_text().strip() for li in elem.find_all("li") if li.get_text().strip()]
            if items:
                bullet_str = "\n".join(f"• {it}" for it in items)
                current_paragraphs.append(bullet_str)

    if current_paragraphs:
        sections.append((current_heading, "\n\n".join(current_paragraphs)))

    if not sections or len(sections) < 2:
        return get_fallback_article(fallback_title)

    return {"title": title, "sections": sections}

def get_fallback_article(title: str) -> dict:
    if "Hypertension" in title:
        return {
            "title": "WHO Clinical Fact Sheet: Hypertension",
            "sections": [
                ("Overview", "Hypertension (high blood pressure) is when the pressure in your blood vessels is too high (140/90 mmHg or higher). It is common but can be serious if not treated. An estimated 1.28 billion adults aged 30-79 years worldwide have hypertension, most living in low- and middle-income countries. An estimated 46% of adults with hypertension are unaware that they have the condition. Only 1 in 5 adults with hypertension have it under control."),
                ("Symptoms and Diagnosis", "Most people with hypertension do not feel any symptoms. Very high blood pressure can cause headaches, blurred vision, chest pain and other symptoms. Checking your blood pressure is the best way to know if you have high blood pressure. If untreated, hypertension can cause health problems like kidney failure, heart disease and stroke. Symptoms of very high blood pressure (usually 180/120 or higher) can include severe headaches, chest pain, dizziness, difficulty breathing, nausea, vomiting, blurred vision or other vision changes, anxiety, confusion, buzzing in the ears, nosebleeds, and abnormal heart rhythm."),
                ("Risk Factors", "Modifiable risk factors include unhealthy diets (excessive salt consumption, a diet high in saturated fat and trans fats, low intake of fruits and vegetables), physical inactivity, consumption of tobacco and alcohol, and being overweight or obese. Non-modifiable risk factors include a family history of hypertension, age over 65 years and co-existing diseases such as diabetes or kidney disease."),
                ("Prevention and Management", "Lifestyle changes can help lower high blood pressure. These include eating a healthy, low-salt diet, losing weight, being physically active, and quitting tobacco. Salt intake should be reduced to less than 5g daily. If lifestyle changes are not enough, medications are needed. Common classes of blood pressure medications include ACE inhibitors, angiotensin receptor blockers (ARBs), calcium channel blockers, and diuretics. Blood pressure should be maintained below 130/80 mmHg in high-risk patients.")
            ]
        }
    elif "Diabetes" in title:
        return {
            "title": "WHO Clinical Fact Sheet: Diabetes",
            "sections": [
                ("Overview", "Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose. Hyperglycaemia, also called raised blood glucose or raised blood sugar, is a common effect of uncontrolled diabetes and over time leads to serious damage to many of the body's systems, especially the nerves and blood vessels. In 2019, diabetes was the direct cause of 1.5 million deaths."),
                ("Type 1 versus Type 2 Diabetes", "Type 1 diabetes (previously known as insulin-dependent or childhood-onset) is characterized by deficient insulin production and requires daily administration of insulin. The cause of type 1 diabetes is not known and it is currently not preventable. Type 2 diabetes affects how your body uses sugar (glucose) for energy. It stops the body from using insulin properly, which can lead to high levels of blood sugar if not treated. More than 95% of people with diabetes have type 2 diabetes. Type 2 diabetes was formerly called non-insulin-dependent, or adult-onset."),
                ("Symptoms", "Symptoms of diabetes may occur suddenly. In type 2 diabetes, the symptoms can be mild and may take many years to be noticed. Common symptoms include feeling very thirsty, needing to urinate more often than usual, blurred vision, feeling tired, and losing weight unintentionally. Over time, diabetes can damage blood vessels in the heart, eyes, kidneys and nerves. People with diabetes have a higher risk of health problems including heart attack, stroke and kidney failure."),
                ("Prevention and Treatment", "Diet and physical activity are important parts of a healthy lifestyle when you have diabetes. Healthy eating includes reducing sugar and saturated fats. Physical activity helps control weight and uses glucose as energy, making cells more sensitive to insulin. Adults should aim for at least 150 minutes of moderate-intensity aerobic physical activity weekly. Treatment involves lowering blood glucose and the levels of other known risk factors that damage blood vessels. People with type 1 diabetes require insulin; people with type 2 diabetes can be treated with oral medication such as metformin, although insulin may also be prescribed.")
            ]
        }
    elif "Asthma" in title:
        return {
            "title": "WHO Clinical Fact Sheet: Asthma",
            "sections": [
                ("Overview", "Asthma is a major noncommunicable disease (NCD), affecting both children and adults, and is the most common chronic disease among children. Asthma affected an estimated 262 million people in 2019 and caused 455,000 deaths. Inhaled medication can control asthma symptoms and allow people with asthma to lead a normal, active life. Asthma is characterized by recurrent attacks of breathlessness and wheezing, which vary in severity and frequency from person to person."),
                ("Symptoms and Causes", "Symptoms of asthma include coughing, wheezing (a whistling sound when breathing), shortness of breath, and chest tightness. Symptoms are often worse at night or during exercise. The exact causes of asthma are not completely understood, although risk factors include a genetic predisposition, allergic conditions such as eczema and rhinitis, and environmental triggers including indoor allergens (house dust mites, pet dander), outdoor allergens (pollens and moulds), tobacco smoke, chemical irritants in the workplace, and air pollution."),
                ("Management and Medications", "Asthma cannot be cured, but good management with inhaled medications can control the disease and enable people to enjoy a normal, active life. There are two main types of inhalers: bronchodilators (such as salbutamol), which open the air passages and relieve symptoms during an acute attack, and steroids (such as beclometasone), which reduce inflammation in the air passages, preventing asthma attacks from occurring. People with asthma may need to use an inhaler every day. Their treatment will depend on the frequency of symptoms and the different types of inhalers available."),
                ("Action Plan and Triggers", "Avoiding asthma triggers can also help reduce asthma symptoms. People with asthma should work with a healthcare professional to establish an asthma action plan. An action plan outlines daily treatment, how to adjust medications when symptoms worsen, and when to seek emergency medical attention. Severe asthma attacks can be life-threatening and require immediate medical intervention.")
            ]
        }
    else:
        return {
            "title": "WHO Fact Sheet: Cardiovascular Diseases (CVDs)",
            "sections": [
                ("Overview", "Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age."),
                ("Warning Signs of Heart Attack and Stroke", "Symptoms of a heart attack include severe pain or discomfort in the center of the chest; and pain or discomfort in the arms, the left shoulder, elbows, jaw, or back. In addition, the person may experience difficulty in breathing or shortness of breath; feeling sick or vomiting; feeling light-headed or faint; breaking into a cold sweat; and becoming pale. The most common symptom of a stroke is sudden weakness or numbness of the face, arm or leg, most often on one side of the body. Other symptoms include confusion, difficulty speaking or understanding speech; difficulty seeing with one or both eyes; difficulty walking, dizziness, loss of balance or coordination; severe headache with no known cause; and fainting or unconsciousness. People experiencing these symptoms must seek medical care immediately."),
                ("Risk Factors and Prevention", "The most important behavioral risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. The effects of behavioral risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease.")
            ]
        }

def create_pdf(doc_data: dict, output_path: Path):
    doc = fitz.open()
    
    # We create clean multi-page document
    page_width, page_height = 595, 842  # A4 points
    margin = 50
    usable_width = page_width - 2 * margin

    # We distribute sections across distinct pages to test page-level extraction and citation verification
    for page_num, (heading, body) in enumerate(doc_data["sections"], start=1):
        page = doc.new_page(width=page_width, height=page_height)
        
        # Header banner
        page.draw_rect(fitz.Rect(margin, 35, page_width - margin, 37), color=(0.2, 0.4, 0.7), fill=(0.2, 0.4, 0.7))
        page.insert_text(fitz.Point(margin, 30), doc_data["title"], fontsize=9, color=(0.4, 0.4, 0.4))
        
        # Section title
        page.insert_text(fitz.Point(margin, 70), f"Section {page_num}: {heading}", fontsize=16, color=(0.1, 0.2, 0.5))
        
        # Body text wrapped in rect
        rect = fitz.Rect(margin, 95, page_width - margin, page_height - 60)
        page.insert_textbox(rect, body, fontsize=11, fontname="helv", color=(0.15, 0.15, 0.15))
        
        # Page footer
        page.draw_line(fitz.Point(margin, page_height - 45), fitz.Point(page_width - margin, page_height - 45), color=(0.8, 0.8, 0.8))
        page.insert_text(fitz.Point(page_width / 2 - 20, page_height - 30), f"Page {page_num} of {len(doc_data['sections'])}", fontsize=9, color=(0.5, 0.5, 0.5))

    doc.save(str(output_path))
    doc.close()
    print(f"Generated PDF: {output_path} ({len(doc_data['sections'])} pages)")

def create_docx(doc_data: dict, output_path: Path):
    doc = docx.Document()
    
    # Title
    title_p = doc.add_heading(doc_data["title"], level=0)
    
    for heading, body in doc_data["sections"]:
        doc.add_heading(heading, level=1)
        for para in body.split("\n\n"):
            if para.startswith("• "):
                for item in para.split("\n"):
                    clean_item = item.lstrip("• ").strip()
                    if clean_item:
                        doc.add_paragraph(clean_item, style="List Bullet")
            else:
                doc.add_paragraph(para)
        doc.add_page_break()

    doc.save(str(output_path))
    print(f"Generated DOCX: {output_path}")

def create_txt(doc_data: dict, output_path: Path):
    lines = [doc_data["title"], "=" * len(doc_data["title"]), ""]
    for heading, body in doc_data["sections"]:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated TXT: {output_path}")

def main():
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for src in SOURCES:
        data = fetch_and_clean_who_article(src["url"], src["title"])
        out_path = settings.RAW_DIR / src["filename"]
        
        if src["type"] == "pdf":
            create_pdf(data, out_path)
        elif src["type"] == "docx":
            create_docx(data, out_path)
        elif src["type"] == "txt":
            create_txt(data, out_path)

        generated_files.append(out_path)

    # Now index all files into MediRAG pipeline
    print("\n--- Indexing Documents into MediRAG Pipeline ---")
    pipeline = MediRAGPipeline.get_instance()
    for f in generated_files:
        print(f"Indexing: {f.name}...")
        rec = pipeline.ingest_file(f)
        print(f"Indexed {rec.filename}: {rec.num_pages} pages, {rec.num_chunks} chunks.")

    print("\nDataset preparation & indexing complete!")
    print(f"Total documents: {len(pipeline.get_documents())}")
    print(f"Total chunks: {pipeline.document_store.total_chunks()}")

if __name__ == "__main__":
    main()
