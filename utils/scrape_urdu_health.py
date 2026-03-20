import bz2
import xml.etree.ElementTree as ET
import mwparserfromhell
import json
import re
import os

# 1. EXACT FOLDER PATHS
# This guarantees the JSON file saves exactly where this Python script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
dump_path = os.path.join(script_directory, "urwiki-latest-pages-articles.xml.bz2")
output_path = os.path.join(script_directory, "urdu_medical_rag_FINAL.json")

MAX_ARTICLES = 1000

# 2. THE EXACT WORD MATCHING SET 
# We are looking for these exact standalone words, not substrings!
medical_category_words = {
    "طب", "صحت", "امراض", "ادویات", "وائرس", 
    "بیماریاں", "بیماری", "طبی", "کینسر", "ذیابیطس", 
    "ویکسین", "علاج", "جراحی", "ہسپتال", "وبائی"
}

# 3. REJECT KEYWORDS (The Ultimate Noise Filter)
reject_category_keywords = [
    # The Originals (History, Physics, Basic Bios, Safety)
    "مابعدالطبیعیات", "تاریخ", "سیاست", "ممالک", 
    "اموات", "شخصیات", "پیدائشیں", "وفیات", "افراد",
    "سلامتی", "پیشہ وری",
    
    # New: People & Professions (Blocks actors who played doctors, etc.)
    "کھلاڑی", "سیاستدان", "اداکار", "مصنفین", "خاندان",
    
    # New: Media & Entertainment (Blocks medical TV shows, movies, and novels)
    "فلمیں", "ڈرامے", "ٹیلی_ویژن", "ناول", "افسانے",
    
    # New: Places, Admin & Institutions (Blocks medical colleges, companies, cities)
    "شہر", "علاقے", "مقامات", "جامعات", "کمپنیاں", "تنظیمیں", "ادارے",
    
    # New: Other unrelated domains (Law, Economy, Sports, Religion/Mythology)
    "قانون", "معیشت", "کھیل", "مذاہب", "دیومالا",
    
    # New: Wikipedia Meta-pages (Blocks raw lists and incomplete "stub" articles)
    "فہرستیں", "نامکمل"
]

# Skip Wikipedia backend pages
ignore_prefixes = ("ماڈیول:", "زمرہ:", "سانچہ:", "ویکیپیڈیا:", "تبادلہ خیال:", "باب:", "صارف:")

medical_articles = []
article_count = 0
pages_scanned = 0

print("--- STARTING EXACT-WORD CATEGORY EXTRACTION ---")

# Verify the dump file exists before starting
if not os.path.exists(dump_path):
    print(f"❌ ERROR: Could not find {dump_path}")
    print("Please make sure the .bz2 file is in the exact same folder as this script.")
    exit()

with bz2.open(dump_path, "rt", encoding="utf-8") as file:
    context = ET.iterparse(file, events=("end",))
    
    for event, elem in context:
        tag = elem.tag.split('}', 1)[-1]
        
        if tag == "page":
            pages_scanned += 1
            
            # Print a status update every 20,000 pages so you know it hasn't frozen
            if pages_scanned % 20000 == 0:
                print(f"... scanned {pages_scanned} pages, found {article_count} medical articles...")

            title_elem = elem.find(".//{*}title")
            text_elem = elem.find(".//{*}text")
            
            if title_elem is not None and text_elem is not None and text_elem.text:
                title = title_elem.text
                wikitext = text_elem.text
                
                # Filter out Wikipedia backend code files
                if title.startswith(ignore_prefixes):
                    elem.clear()
                    continue
                
                # EXTRACT CATEGORIES
                extracted_cats = re.findall(r'\[\[(?:زمرہ|Category):([^\]|]+)', wikitext)
                
                is_medical = False
                found_cat = ""
                
                # Check every category the article belongs to
                for cat in extracted_cats:
                    # 1. Skip if it contains ANY reject words (like "اموات" or "شخصیات")
                    if any(reject in cat for reject in reject_category_keywords):
                        continue
                        
                    # 2. Clean the string (replace underscores with spaces)
                    cat_clean = cat.replace('_', ' ')
                    
                    # 3. Split the category into an array of exact, individual words
                    cat_words = set(cat_clean.split())
                    
                    # 4. Check if there is an EXACT match between the category's words and our medical words
                    if cat_words.intersection(medical_category_words):
                        is_medical = True
                        found_cat = cat
                        break
                
                # ONLY save it if we got an exact word match and NO reject words
                if is_medical:
                    parsed = mwparserfromhell.parse(wikitext)
                    clean_text = parsed.strip_code()
                    
                    medical_articles.append({
                        "title": title,
                        "category": found_cat,
                        "text": clean_text.strip()
                    })
                    article_count += 1
                    print(f"✅ Added #{article_count}: {title} (Category: {found_cat})")
                    
                    # Stop if we hit our target limit
                    if article_count >= MAX_ARTICLES:
                        break
            
            # Clear element from memory to prevent RAM crashes
            elem.clear()
        
        # Double break to exit the outer loop when finished
        if article_count >= MAX_ARTICLES:
            break

print(f"\n🎉 SUCCESS! Saved {len(medical_articles)} verified medical articles to:")
print(output_path)

# Save to JSON
with open(output_path, 'w', encoding='utf-8') as out_f:
    json.dump(medical_articles, out_f, ensure_ascii=False, indent=4)

print("Ready for RAG chunking!")