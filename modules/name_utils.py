"""
Name normalization and slug generation utilities
Consolidates duplicate name processing logic from multiple sections
"""

import re
import unicodedata


class NameNormalizer:
    """Unified name normalization and slug generation"""
    
    # Common abbreviations mapped to full names
    ABBREVIATIONS = {
        r'\bdan\b': 'daniel',
        r'\btom\b': 'thomas',
        r'\bjon\b': 'jonathan',
        r'\bjim\b': 'james',
        r'\bbob\b': 'robert',
        r'\bwill\b': 'william',
        r'\bliz\b': 'elizabeth',
        r'\bpat\b': 'patrick',
        r'\bbert\b': 'albert',
        r'\bted\b': 'edward',
        r'\bamy\b': 'amelia',
        r'\bkatie\b': 'katherine',
        r'\bcat\b': 'catherine',
        r'\btimothy\b': 'tim',
        r'\bchristopher\b': 'chris',
        r'\banthony\b': 'tony',
    }
    
    # Known name variations for URL generation
    OVERRIDES = {
        "Bernard_Sanders": "Bernie_Sanders",
        "Dan_Sullivan": "Daniel_Sullivan",
        "Tom_Cotton": "Thomas_Cotton",
        "Tommy_Tuberville": "Thomas_Tuberville",
        "Jon_Ossoff": "Jonathan_Ossoff",
        "Alan_Armstrong": "Alan_S._Armstrong",
    }
    
    @staticmethod
    def remove_accents(text):
        """Remove diacritical marks from Unicode string"""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    @classmethod
    def normalize(cls, name, lowercase=True, remove_middle_initial=True):
        """
        Normalize a name string
        
        Args:
            name: Input name string
            lowercase: Convert to lowercase
            remove_middle_initial: Remove middle initial (e.g., "John F. Smith" → "John Smith")
            
        Returns:
            Normalized name string
        """
        if not name or pd.isna(name):
            return ""
        
        # Convert to string if needed
        norm_name = str(name).strip()
        
        # Lowercase
        if lowercase:
            norm_name = norm_name.lower()
        
        # Remove accents
        norm_name = cls.remove_accents(norm_name)
        
        # Remove middle initial: "John F. Smith" → "John Smith"
        if remove_middle_initial:
            norm_name = re.sub(r'\s+[A-Za-z]\.\s*', ' ', norm_name).strip()
        
        # Expand abbreviations
        for pattern, replacement in cls.ABBREVIATIONS.items():
            norm_name = re.sub(pattern, replacement, norm_name)
        
        return norm_name.strip()
    
    @classmethod
    def create_slug(cls, name, wiki_style=True):
        """
        Create a URL slug from a senator name
        
        Args:
            name: Senator's name
            wiki_style: If True, use Wikipedia format; if False, use Ballotpedia
            
        Returns:
            URL slug (e.g., "Bernie_Sanders" for Wikipedia)
        """
        # Normalize name
        norm_name = cls.normalize(name, lowercase=True)
        
        # Replace spaces with underscores
        slug = norm_name.replace(" ", "_")
        
        # Apply hardcoded overrides
        override_key = None
        if wiki_style:
            # Try to find in overrides by checking normalized version
            for override_pattern, override_value in cls.OVERRIDES.items():
                if override_pattern.lower() in slug.lower():
                    return override_value
        
        return slug
    
    @classmethod
    def create_wikipedia_url(cls, name):
        """Create Wikipedia URL for a senator"""
        slug = cls.create_slug(name, wiki_style=True)
        return f"https://en.wikipedia.org/wiki/{slug}"
    
    @classmethod
    def create_ballotpedia_url(cls, name):
        """Create Ballotpedia URL for a senator"""
        slug = cls.create_slug(name, wiki_style=False)
        return f"https://ballotpedia.org/{slug}"
    
    @classmethod
    def create_senator_id(cls, name, state):
        """
        Create standardized senator ID from name and state
        
        Args:
            name: Senator's name
            state: Two-letter state abbreviation (e.g., "CA")
            
        Returns:
            Standardized ID (e.g., "john_smith_ca.html")
        """
        normalized = cls.normalize(name, lowercase=True)
        name_slug = "_".join(w for w in normalized.split() if w)
        return f"{name_slug}_{state.upper()}"


# Optional: Import pandas for NaN checking (done in try/except to avoid hard dependency)
try:
    import pandas as pd
except ImportError:
    # Fallback for environments without pandas
    class _PandasStub:
        @staticmethod
        def isna(x):
            return x is None or (isinstance(x, float) and x != x)  # NaN check
    pd = _PandasStub()
