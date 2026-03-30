#!/usr/bin/env python3
"""
Quick summary statistics for scraped Senate HTML data.

Analyzes file sizes, content patterns, and HTML structure
to identify data quality issues or malformed files.
"""
import os

def analyze_senate_scrape():
    """Analyze HTML scrape data for quality and completeness."""
    path = "senate_html"
    
    if not os.path.exists(path):
        print(f"Error: {path} directory not found")
        return
    
    files = sorted([(f, os.path.getsize(f"senate_html/{f}")) 
                    for f in os.listdir(path) if f.endswith('.html')], 
                   key=lambda x: x[1])

    print("=== FILE SIZE DISTRIBUTION ===")
    print(f"Total files: {len(files)}")
    print(f"Smallest: {files[0][0]} ({files[0][1]} bytes)")
    print(f"Largest: {files[-1][0]} ({files[-1][1]} bytes)")

    sizes = [f[1] for f in files]
    avg_size = sum(sizes) / len(sizes)
    print(f"Average: {avg_size:.0f} bytes ({avg_size/1024:.1f} KB)")

    # Check for suspiciously small files (< 10KB)
    small = [f for f in files if f[1] < 10000]
    print(f"\nFiles < 10KB: {len(small)}")
    if small:
        for fname, size in small:
            print(f"  {fname}: {size}")

    # Sample content analysis
    print("\n=== CONTENT ANALYSIS ===")
    for fname in ["John_Boozman_AR.html", "Tina_Smith_MN.html", "Bernard_Sanders_VT.html"]:
        fpath = f"senate_html/{fname}"
        try:
            with open(fpath) as f:
                content = f.read()
            
            bio_keywords = sum(1 for kw in ['biography', 'born', 'education', 'career', 'served', 'elected'] 
                              if kw.lower() in content.lower())
            is_redirect = 'window.location' in content or ('meta' in content and 'refresh' in content.lower())
            
            print(f"\n{fname}:")
            print(f"  Size: {os.path.getsize(fpath)} bytes")
            print(f"  Bio keywords found: {bio_keywords}")
            print(f"  Likely redirect: {is_redirect}")
            
            # Check HTML structure
            html_tags = {}
            for tag in ['div', 'section', 'article', 'main', 'aside', 'nav']:
                count = content.lower().count(f"<{tag}")
                html_tags[tag] = count
            print(f"  Main tags: {dict((k,v) for k,v in html_tags.items() if v > 0)}")
        except FileNotFoundError:
            print(f"{fname}: NOT FOUND")

    # Structural heterogeneity analysis
    print("\n=== STRUCTURAL HETEROGENEITY ===")
    structures = {}
    for fname in files[:10]:  # Sample first 10 files
        fpath = f"senate_html/{fname[0]}"
        with open(fpath) as f:
            content = f.read()
        
        # Count key structural elements
        sig = tuple(sorted([
            ('div_count', content.lower().count('<div')),
            ('section_count', content.lower().count('<section')),
            ('article_count', content.lower().count('<article')),
            ('main_count', content.lower().count('<main')),
        ]))
        
        if sig not in structures:
            structures[sig] = []
        structures[sig].append(fname[0])

    print(f"Number of distinct HTML structure patterns: {len(structures)}")
    for i, (sig, files_list) in enumerate(structures.items()):
        print(f"\nPattern {i+1}: {dict(sig)}")
        print(f"  Files: {', '.join(files_list[:3])}")

if __name__ == '__main__':
    analyze_senate_scrape()
