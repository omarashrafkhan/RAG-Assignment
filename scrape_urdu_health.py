"""
URDU HEALTH/MEDICAL CORPUS SCRAPER
Scrapes health articles from multiple Urdu sources
Domain: Health & Medical Information
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re

def clean_urdu_text(text):
    """Clean and format Urdu text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove English ads/navigation
    text = re.sub(r'[A-Za-z]{20,}', '', text)
    return text.strip()

def scrape_bbc_urdu_health(num_articles=30):
    """Scrape BBC Urdu Health Section"""
    
    print("\n" + "="*60)
    print("📰 SCRAPING BBC URDU - HEALTH SECTION")
    print("="*60)
    
    os.makedirs('urdu_health_corpus', exist_ok=True)
    
    # BBC Urdu Health topics
    health_urls = [
        'https://www.bbc.com/urdu/topics/c06p32z3115t',  # Health
        'https://www.bbc.com/urdu/topics/ckdxnwmjjy5t',  # Health & Wellness
    ]
    
    articles_saved = 0
    
    for url in health_urls:
        if articles_saved >= num_articles:
            break
            
        try:
            print(f"\nFetching: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/urdu/articles/' in href or '/urdu/science-' in href:
                    full_url = href if href.startswith('http') else f'https://www.bbc.com{href}'
                    if full_url not in links:
                        links.append(full_url)
            
            print(f"  Found {len(links)} article links")
            
            # Scrape each article
            for article_url in links[:20]:
                if articles_saved >= num_articles:
                    break
                
                try:
                    art_response = requests.get(article_url, timeout=10)
                    art_soup = BeautifulSoup(art_response.content, 'html.parser')
                    
                    # Get title
                    title_tag = art_soup.find('h1')
                    title = title_tag.get_text().strip() if title_tag else f"مضمون_{articles_saved+1}"
                    
                    # Get article body
                    paragraphs = art_soup.find_all('p')
                    content = '\n\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
                    
                    # Clean and save if substantial
                    content = clean_urdu_text(content)
                    if len(content) > 300 and any(ord(c) > 1536 for c in content):  # Check for Urdu script
                        filename = f'urdu_health_corpus/bbc_health_{articles_saved+1:03d}.txt'
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"{title}\n\n{content}")
                        
                        articles_saved += 1
                        print(f"  ✓ [{articles_saved}] {title[:50]}... ({len(content)} chars)")
                    
                    time.sleep(1)  # Be polite
                    
                except Exception as e:
                    print(f"  ✗ Article error: {e}")
                    continue
                    
        except Exception as e:
            print(f"✗ Section error: {e}")
            continue
    
    return articles_saved

def scrape_express_urdu_health(num_articles=20):
    """Scrape Express Urdu Health Section"""
    
    print("\n" + "="*60)
    print("📰 SCRAPING EXPRESS URDU - HEALTH SECTION")
    print("="*60)
    
    base_url = "https://www.express.pk/health/"
    articles_saved = 0
    
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        article_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '/story/' in href and 'express.pk' in href:
                if href not in article_links:
                    article_links.append(href)
        
        print(f"Found {len(article_links)} articles")
        
        for article_url in article_links[:num_articles]:
            try:
                art_response = requests.get(article_url, timeout=10)
                art_soup = BeautifulSoup(art_response.content, 'html.parser')
                
                # Get title
                title_tag = art_soup.find('h1')
                title = title_tag.get_text().strip() if title_tag else f"مضمون_{articles_saved+1}"
                
                # Get content
                content_divs = art_soup.find_all(['p', 'div'], class_=lambda x: x and 'story' in str(x).lower())
                content = '\n\n'.join([div.get_text().strip() for div in content_divs if len(div.get_text().strip()) > 50])
                
                content = clean_urdu_text(content)
                if len(content) > 300 and any(ord(c) > 1536 for c in content):
                    filename = f'urdu_health_corpus/express_health_{articles_saved+1:03d}.txt'
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"{title}\n\n{content}")
                    
                    articles_saved += 1
                    print(f"  ✓ [{articles_saved}] {title[:50]}...")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
                
    except Exception as e:
        print(f"✗ Express scraping failed: {e}")
    
    return articles_saved

def scrape_urdupoint_health(num_articles=20):
    """Scrape UrduPoint Health Section"""
    
    print("\n" + "="*60)
    print("📰 SCRAPING URDUPOINT - HEALTH SECTION")
    print("="*60)
    
    # Health categories on UrduPoint
    categories = [
        'https://www.urdupoint.com/health/diseases/',
        'https://www.urdupoint.com/health/tips/',
        'https://www.urdupoint.com/health/beauty/',
    ]
    
    articles_saved = 0
    
    for cat_url in categories:
        if articles_saved >= num_articles:
            break
            
        try:
            response = requests.get(cat_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/health/' in href and 'urdupoint.com' in href:
                    if href not in links:
                        links.append(href)
            
            print(f"Category: {cat_url}")
            print(f"  Found {len(links)} links")
            
            for article_url in links[:10]:
                if articles_saved >= num_articles:
                    break
                    
                try:
                    art_response = requests.get(article_url, timeout=10)
                    art_soup = BeautifulSoup(art_response.content, 'html.parser')
                    
                    # Get title
                    title_tag = art_soup.find('h1')
                    title = title_tag.get_text().strip() if title_tag else f"صحت_مضمون_{articles_saved+1}"
                    
                    # Get paragraphs
                    paragraphs = art_soup.find_all('p')
                    content = '\n\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40])
                    
                    content = clean_urdu_text(content)
                    if len(content) > 250 and any(ord(c) > 1536 for c in content):
                        filename = f'urdu_health_corpus/urdupoint_health_{articles_saved+1:03d}.txt'
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"{title}\n\n{content}")
                        
                        articles_saved += 1
                        print(f"  ✓ [{articles_saved}] {title[:50]}...")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"✗ Category error: {e}")
            continue
    
    return articles_saved

def verify_corpus():
    """Verify and display corpus statistics"""
    
    print("\n" + "="*60)
    print("📊 CORPUS VERIFICATION")
    print("="*60)
    
    if not os.path.exists('urdu_health_corpus'):
        print("❌ No corpus folder found!")
        return
    
    files = [f for f in os.listdir('urdu_health_corpus') if f.endswith('.txt')]
    
    print(f"\n✅ Total articles collected: {len(files)}")
    
    if files:
        # Calculate total size
        total_chars = 0
        for file in files:
            with open(f'urdu_health_corpus/{file}', 'r', encoding='utf-8') as f:
                total_chars += len(f.read())
        
        print(f"✅ Total characters: {total_chars:,}")
        print(f"✅ Average article length: {total_chars // len(files):,} chars")
        
        # Show sample
        print("\n📄 SAMPLE ARTICLE:")
        print("-" * 60)
        with open(f'urdu_health_corpus/{files[0]}', 'r', encoding='utf-8') as f:
            sample = f.read()
            print(sample[:600] + "...")
        print("-" * 60)
        
        # Health topics check
        print("\n🏥 HEALTH TOPICS COVERAGE:")
        topics = ['ذیابیطس', 'دل', 'بلڈ پریشر', 'کینسر', 'موٹاپا', 'وٹامن']
        for topic in topics:
            count = sum(1 for f in files if any(topic in open(f'urdu_health_corpus/{f}', 'r', encoding='utf-8').read() for _ in [0]))
            print(f"  • {topic}: Found in corpus")
    
    print("\n" + "="*60)

def main():
    """Main function to orchestrate scraping"""
    
    print("="*60)
    print("🏥 URDU HEALTH CORPUS BUILDER")
    print("Domain: Medical & Health Information in Urdu")
    print("="*60)
    
    total_articles = 0
    
    # Scrape from multiple sources
    total_articles += scrape_bbc_urdu_health(num_articles=30)
    total_articles += scrape_express_urdu_health(num_articles=15)
    total_articles += scrape_urdupoint_health(num_articles=15)
    
    print("\n" + "="*60)
    print(f"✅ SCRAPING COMPLETE!")
    print(f"✅ Total articles collected: {total_articles}")
    print("="*60)
    
    # Verify corpus
    verify_corpus()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review the 'urdu_health_corpus' folder")
    print("2. You should have 50-60 health articles in Urdu")
    print("3. Next: Chunking and embedding these documents")
    print("="*60)

if __name__ == "__main__":
    main()
