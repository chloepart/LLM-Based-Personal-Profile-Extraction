"""
HTML processing utilities
Consolidates BeautifulSoup operations used throughout the pipeline
"""

import re
from bs4 import BeautifulSoup


class HTMLProcessor:
    """Centralized HTML processing functions"""
    
    # Default tags to remove during preprocessing
    DEFAULT_REMOVAL_TAGS = ["script", "style", "nav", "footer", "noscript"]
    
    @staticmethod
    def extract_readable_text(html, separator=" ", strip_tags=None, max_length=None):
        """
        Extract readable text from HTML, removing script/style/nav/footer
        
        Args:
            html: HTML string
            separator: String to use between extracted text elements (default: " ")
            strip_tags: List of tags to remove (default: script, style, nav, footer, noscript)
            max_length: If set, truncate to this many characters
            
        Returns:
            Cleaned text string
        """
        if not html or (isinstance(html, str) and len(html) == 0):
            return ""
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove specified tags
        if strip_tags is None:
            strip_tags = HTMLProcessor.DEFAULT_REMOVAL_TAGS
        
        for tag in soup(strip_tags):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator=separator, strip=True)
        
        # Normalize whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def extract_infobox(html, field_mappings=None):
        """
        Extract data from Wikipedia infobox
        
        Args:
            html: Wikipedia HTML
            field_mappings: Dict mapping field names to extract to label patterns
                           e.g., {"birth_date": "born", "gender": "gender"}
            
        Returns:
            Dict with extracted field values
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Find infobox table (usually has class containing "infobox")
        infobox = soup.find("table", {"class": lambda x: x and "infobox" in x})
        
        if not infobox:
            return {}
        
        result = {}
        rows = infobox.find_all("tr")
        
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            
            # Store raw value for any label
            if not value:
                continue
            
            # If field_mappings provided, only store mapped fields
            if field_mappings:
                for field_name, label_pattern in field_mappings.items():
                    if label_pattern.lower() in label:
                        result[field_name] = value
                        break
            else:
                result[label] = value
        
        return result
    
    @staticmethod
    def extract_page_title(html):
        """Extract page title from <h1> tag"""
        soup = BeautifulSoup(html, "html.parser")
        h1_tag = soup.find("h1")
        if h1_tag:
            title = h1_tag.get_text(strip=True)
            # Remove disambiguation suffix
            title = re.sub(r'\s*\(.*?\)\s*$', '', title).strip()
            return title
        return None
    
    @staticmethod
    def extract_text_around_header(html, header_text, context_lines=10):
        """
        Extract text around a specific header
        
        Args:
            html: HTML string
            header_text: Text to search for in headers (h2, h3, h4)
            context_lines: Number of lines of text to extract after header
            
        Returns:
            Extracted text block
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Find header containing text
        header = None
        for h in soup.find_all(['h2', 'h3', 'h4']):
            if header_text.lower() in h.get_text().lower():
                header = h
                break
        
        if not header:
            return None
        
        # Extract text until next header
        text_parts = []
        sibling = header.find_next()
        
        while sibling and len(text_parts) < context_lines:
            if sibling.name and sibling.name.startswith('h'):
                break
            
            if sibling.name in ['p', 'li', 'td']:
                text = sibling.get_text(strip=True)
                if text:
                    text_parts.append(text)
            
            sibling = sibling.find_next_sibling() or sibling.find_next()
        
        return " ".join(text_parts)
    
    @staticmethod
    def extract_links(html, filter_text=None):
        """
        Extract all links from HTML
        
        Args:
            html: HTML string
            filter_text: If provided, only return links containing this text
            
        Returns:
            List of tuples: (link_text, href)
        """
        soup = BeautifulSoup(html, "html.parser")
        
        links = []
        for a in soup.find_all("a", href=True):
            link_text = a.get_text(strip=True)
            href = a['href']
            
            if filter_text is None or filter_text.lower() in link_text.lower():
                links.append((link_text, href))
        
        return links


class WikipediaExtractor(HTMLProcessor):
    """Specialized extractor for Wikipedia pages"""
    
    INFOBOX_FIELD_MAPPINGS = {
        "full_name": "name",
        "birthdate": "born",
        "birth_place": "birth place",
        "gender": "gender",
        "race_ethnicity": "ethnicity",
        "residence": "residence",
    }
    
    @classmethod
    def extract_profile(cls, html):
        """
        Extract full senator profile from Wikipedia
        
        Returns:
            Dict with: full_name, birthdate, gender, race_ethnicity, education_text, siblings
        """
        result = {
            "full_name": cls.extract_page_title(html),
            "infobox": cls.extract_infobox(html, cls.INFOBOX_FIELD_MAPPINGS),
        }
        
        return result


# Convenience functions for backward compatibility
def extract_readable_text(html, **kwargs):
    """Extract readable text (function wrapper)"""
    return HTMLProcessor.extract_readable_text(html, **kwargs)

def extract_infobox(html, **kwargs):
    """Extract infobox data (function wrapper)"""
    return HTMLProcessor.extract_infobox(html, **kwargs)
