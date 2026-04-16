import pandas as pd
import json
import ast
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class SenatorDataStandardizer:
    """Comprehensive standardization for US Senate biographical data."""
    
    ETHNICITY_MAPPINGS = {
        'white': 'White', 'caucasian': 'White', 'european': 'White',
        'black': 'Black', 'african american': 'Black', 'african-american': 'Black',
        'hispanic': 'Hispanic', 'latino': 'Hispanic', 'latino/a': 'Hispanic', 'latinx': 'Hispanic',
        'asian': 'Asian', 'east asian': 'Asian', 'south asian': 'Asian', 'southeast asian': 'Asian',
        'chinese': 'Asian', 'japanese': 'Asian', 'korean': 'Asian', 'indian': 'Asian',
        'vietnamese': 'Asian', 'thai': 'Asian',
        'native american': 'Native American', 'american indian': 'Native American', 'indigenous': 'Native American',
        'pacific islander': 'Pacific Islander', 'native hawaiian': 'Pacific Islander',
        'middle eastern': 'Middle Eastern/North African', 'arab': 'Middle Eastern/North African', 'persian': 'Middle Eastern/North African',
        'mixed': 'Mixed', 'multiracial': 'Mixed', 'biracial': 'Mixed',
    }
    
    DEGREE_ABBREV = {
        'b.a.': 'BA', 'ba': 'BA', 'b.s.': 'BS', 'bs': 'BS', 'b.e.': 'BE', 'be': 'BE',
        'm.a.': 'MA', 'ma': 'MA', 'm.s.': 'MS', 'ms': 'MS', 'm.b.a.': 'MBA', 'mba': 'MBA',
        'j.d.': 'JD', 'jd': 'JD', 'l.l.b.': 'LLB', 'llb': 'LLB', 'l.l.m.': 'LLM', 'llm': 'LLM',
        'm.d.': 'MD', 'md': 'MD', 'ph.d.': 'PhD', 'phd': 'PhD', 'ed.d.': 'EdD', 'edd': 'EdD',
        'd.d.s.': 'DDS', 'dds': 'DDS',
    }
    
    INSTITUTION_CLEAN = {
        'mit': 'Massachusetts Institute of Technology', 'stanford': 'Stanford University',
        'harvard': 'Harvard University', 'yale': 'Yale University', 'columbia': 'Columbia University',
        'princeton': 'Princeton University', 'berkeley': 'University of California, Berkeley',
        'caltech': 'California Institute of Technology',
    }
    
    RELIGION_MAPPINGS = {
        'christian': 'Christian', 'catholic': 'Catholic', 'protestant': 'Protestant',
        'baptist': 'Baptist', 'methodist': 'Methodist', 'episcopal': 'Episcopal',
        'presbyterian': 'Presbyterian', 'lutheran': 'Lutheran', 'mormon': 'LDS (Mormon)',
        'lds': 'LDS (Mormon)', 'latter-day saints': 'LDS (Mormon)', 'jewish': 'Jewish',
        'jew': 'Jewish', 'jewish faith': 'Jewish', 'jewish heritage': 'Jewish',
        'muslim': 'Muslim', 'islam': 'Muslim', 'buddhist': 'Buddhist', 'buddhism': 'Buddhist',
        'hindu': 'Hindu', 'hinduism': 'Hindu', 'atheist': 'Atheist', 'atheism': 'Atheist',
        'agnostic': 'Agnostic', 'agnosticism': 'Agnostic', 'unitarian': 'Unitarian',
        'congregationalist': 'Congregationalist', 'orthodox': 'Orthodox Christian',
        'greek orthodox': 'Orthodox Christian', 'seventh-day adventist': 'Seventh-day Adventist',
        'pentecostal': 'Pentecostal', 'evangelical': 'Evangelical Christian', 'quaker': 'Quaker',
    }
    
    @staticmethod
    def standardize_full_name(name: Any) -> Optional[str]:
        if pd.isna(name) or name == '':
            return None
        name = str(name).strip()
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            name = ' '.join(reversed(parts))
        name = re.sub(r'\s+', ' ', name)
        titles = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sen.', 'senator']
        name_lower = name.lower()
        for title in titles:
            name_lower = name_lower.replace(title, '').strip()
        name = ' '.join(word.capitalize() for word in name_lower.split())
        return name if name else None
    
    @staticmethod
    def validate_full_name(name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        parts = name.split()
        return len(parts) >= 2
    
    @staticmethod
    def standardize_gender(gender: Any) -> Optional[str]:
        if pd.isna(gender) or gender == '':
            return None
        gender_str = str(gender).strip().lower()
        male_variants = ['m', 'male', 'man', 'mr', 'mister']
        female_variants = ['f', 'female', 'woman', 'mrs', 'miss', 'ms']
        if gender_str in male_variants:
            return 'Male'
        elif gender_str in female_variants:
            return 'Female'
        return None
    
    @staticmethod
    def validate_gender(gender: str) -> bool:
        return gender in ['Male', 'Female']
    
    @classmethod
    def standardize_race_ethnicity(cls, ethnicity: Any) -> Optional[str]:
        if pd.isna(ethnicity) or ethnicity == '':
            return None
        ethnicity_str = str(ethnicity).strip().lower()
        if ethnicity_str in cls.ETHNICITY_MAPPINGS:
            return cls.ETHNICITY_MAPPINGS[ethnicity_str]
        for key, value in cls.ETHNICITY_MAPPINGS.items():
            if key in ethnicity_str:
                return value
        return None
    
    @staticmethod
    def validate_race_ethnicity(ethnicity: str) -> bool:
        valid_categories = ['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Pacific Islander', 'Middle Eastern/North African', 'Mixed']
        return ethnicity in valid_categories
    
    @staticmethod
    def standardize_birthdate(birthdate: Any) -> Optional[str]:
        if pd.isna(birthdate) or birthdate == '':
            return None
        birthdate_str = str(birthdate).strip()
        if birthdate_str.isdigit():
            try:
                serial = int(birthdate_str)
                date_obj = datetime(1900, 1, 1) + pd.Timedelta(days=serial - 2)
                return date_obj.strftime('%Y-%m-%d')
            except:
                pass
        date_formats = ['%m/%d/%y', '%m/%d/%Y', '%m-%d-%y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%B %d, %Y', '%b %d, %Y', '%Y%m%d']
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(birthdate_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except:
                continue
        return None
    
    @staticmethod
    def validate_birthdate(birthdate: str) -> bool:
        if not birthdate:
            return False
        try:
            date_obj = datetime.strptime(birthdate, '%Y-%m-%d')
            year = date_obj.year
            return 1900 <= year <= 2010
        except:
            return False
    
    @classmethod
    def standardize_education(cls, education: Any) -> Optional[List[Dict[str, Any]]]:
        if pd.isna(education) or education == '' or education == '[]':
            return None
        
        education_list = []
        
        if isinstance(education, str):
            education = education.strip()
            if education.startswith('['):
                try:
                    education = json.loads(education)
                except:
                    try:
                        education = ast.literal_eval(education)
                    except:
                        return None
            else:
                if '|' in education:
                    education = education.split('|')
                else:
                    education = [education]
        
        if not isinstance(education, list):
            education = [education]
        
        for entry in education:
            if isinstance(entry, dict):
                standardized = cls._standardize_education_entry(entry)
            else:
                standardized = cls._parse_education_text(str(entry))
            if standardized:
                education_list.append(standardized)
        
        return education_list if education_list else None
    
    @classmethod
    def _standardize_education_entry(cls, entry: Dict) -> Optional[Dict]:
        if not isinstance(entry, dict):
            return None
        standardized = {}
        if 'degree' in entry and entry['degree']:
            degree = str(entry['degree']).strip().lower()
            standardized['degree'] = cls.DEGREE_ABBREV.get(degree, degree.upper())
        else:
            standardized['degree'] = None
        if 'institution' in entry and entry['institution']:
            inst = str(entry['institution']).strip().lower()
            inst = re.sub(r'university of ', '', inst)
            inst = re.sub(r'the ', '', inst)
            standardized['institution'] = cls.INSTITUTION_CLEAN.get(inst, inst.title())
        else:
            standardized['institution'] = None
        if 'year' in entry and entry['year']:
            try:
                standardized['year'] = int(entry['year'])
            except:
                standardized['year'] = None
        else:
            standardized['year'] = None
        return standardized if (standardized.get('degree') or standardized.get('institution')) else None
    
    @classmethod
    def _parse_education_text(cls, text: str) -> Optional[Dict]:
        text = text.strip()
        if not text:
            return None
        entry = {'degree': None, 'institution': None, 'year': None}
        for abbrev in cls.DEGREE_ABBREV.keys():
            if abbrev in text.lower():
                entry['degree'] = cls.DEGREE_ABBREV[abbrev]
                text = re.sub(rf'\b{abbrev}\b', '', text, flags=re.IGNORECASE)
                break
        year_match = re.search(r'(\d{4})', text)
        if year_match:
            entry['year'] = int(year_match.group(1))
            text = re.sub(r'\d{4}', '', text)
        entry['institution'] = text.strip().title() if text.strip() else None
        return entry if (entry['degree'] or entry['institution']) else None
    
    @staticmethod
    def validate_education(education: List[Dict]) -> bool:
        if not isinstance(education, list):
            return False
        return all(isinstance(e, dict) and ('degree' in e or 'institution' in e) for e in education)
    
    @classmethod
    def standardize_religion(cls, religion: Any) -> Optional[str]:
        if pd.isna(religion) or religion == '':
            return None
        religion_str = str(religion).strip().lower()
        if religion_str in cls.RELIGION_MAPPINGS:
            return cls.RELIGION_MAPPINGS[religion_str]
        for key, value in cls.RELIGION_MAPPINGS.items():
            if key in religion_str:
                return value
        return None
    
    @staticmethod
    def validate_religion(religion: str) -> bool:
        valid = ['Christian', 'Catholic', 'Protestant', 'Baptist', 'Methodist', 'Episcopal',
                 'Presbyterian', 'Lutheran', 'LDS (Mormon)', 'Jewish', 'Muslim', 'Buddhist',
                 'Hindu', 'Atheist', 'Agnostic', 'Unitarian', 'Congregationalist',
                 'Orthodox Christian', 'Seventh-day Adventist', 'Pentecostal',
                 'Evangelical Christian', 'Quaker']
        return religion in valid
    
    @staticmethod
    def standardize_committee_roles(roles: Any) -> Optional[List[str]]:
        if pd.isna(roles) or roles == '' or roles == '[]':
            return None
        if isinstance(roles, str):
            roles = roles.strip()
            if roles.startswith('['):
                try:
                    roles = json.loads(roles)
                except:
                    roles = [r.strip() for r in roles.split('|')]
            elif '|' in roles:
                roles = [r.strip() for r in roles.split('|')]
            else:
                roles = [roles]
        if not isinstance(roles, list):
            roles = [roles]
        standardized = []
        for role in roles:
            role_clean = str(role).strip()
            if role_clean and role_clean not in ['', 'nan', 'None']:
                role_clean = ' '.join(w.capitalize() for w in role_clean.split())
                standardized.append(role_clean)
        seen = set()
        unique = []
        for role in standardized:
            if role.lower() not in seen:
                seen.add(role.lower())
                unique.append(role)
        return unique if unique else None
    
    @staticmethod
    def validate_committee_roles(roles: List[str]) -> bool:
        if not isinstance(roles, list):
            return False
        return all(isinstance(r, str) and len(r) > 0 for r in roles)
    
    def standardize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        standardized = {}
        validation = {}
        
        if 'full_name' in record or 'name' in record:
            name_field = record.get('full_name') or record.get('name')
            std_name = self.standardize_full_name(name_field)
            standardized['full_name'] = std_name
            validation['full_name_valid'] = self.validate_full_name(std_name) if std_name else False
        
        if 'gender' in record:
            std_gender = self.standardize_gender(record['gender'])
            standardized['gender'] = std_gender
            validation['gender_valid'] = self.validate_gender(std_gender) if std_gender else False
        
        if 'race_ethnicity' in record or 'ethnicity' in record:
            eth_field = record.get('race_ethnicity') or record.get('ethnicity')
            std_eth = self.standardize_race_ethnicity(eth_field)
            standardized['race_ethnicity'] = std_eth
            validation['race_ethnicity_valid'] = self.validate_race_ethnicity(std_eth) if std_eth else False
        
        if 'birthdate' in record:
            std_date = self.standardize_birthdate(record['birthdate'])
            standardized['birthdate'] = std_date
            validation['birthdate_valid'] = self.validate_birthdate(std_date) if std_date else False
        
        if 'education' in record:
            std_edu = self.standardize_education(record['education'])
            standardized['education'] = std_edu
            validation['education_valid'] = self.validate_education(std_edu) if std_edu else False
        
        if 'religion' in record or 'religious_affiliation' in record:
            rel_field = record.get('religion') or record.get('religious_affiliation')
            std_rel = self.standardize_religion(rel_field)
            standardized['religion'] = std_rel
            validation['religion_valid'] = self.validate_religion(std_rel) if std_rel else False
        
        if 'committee_roles' in record or 'committees' in record:
            com_field = record.get('committee_roles') or record.get('committees')
            std_com = self.standardize_committee_roles(com_field)
            standardized['committee_roles'] = std_com
            validation['committee_roles_valid'] = self.validate_committee_roles(std_com) if std_com else False
        
        standardized['_validation'] = validation
        return standardized
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in df.iterrows():
            std_record = self.standardize_record(row.to_dict())
            results.append(std_record)
        return pd.DataFrame(results)
