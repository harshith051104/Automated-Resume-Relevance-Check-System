import PyPDF2
import docx2txt
import re
from typing import Dict, Any, List
import spacy
from io import BytesIO

class ResumeParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_text(self, file) -> str:
        """Extract text from uploaded file"""
        text = ""
        
        try:
            if file.type == "application/pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
                    if len(pdf_reader.pages) == 0:
                        return "Error: PDF file appears to be empty or corrupted (no pages found)."
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += page_text + "\n"
                    
                    if not text.strip():
                        return "Error: Could not extract readable text from PDF. The file may be image-based or corrupted."
                        
                except PyPDF2.errors.EmptyFileError:
                    return "Error: PDF file is empty or corrupted."
                except PyPDF2.errors.PdfReadError as e:
                    return f"Error: Unable to read PDF file. {str(e)}"
                except Exception as e:
                    return f"Error: Unexpected error reading PDF. {str(e)}"
            
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    text = docx2txt.process(BytesIO(file.read()))
                    if not text.strip():
                        return "Error: Could not extract text from Word document. The file may be empty or corrupted."
                except Exception as e:
                    return f"Error: Unable to read Word document. {str(e)}"
            
            else:
                return f"Error: Unsupported file type '{file.type}'. Please upload a PDF or Word document."
                
        except Exception as e:
            return f"Error: Failed to process file. {str(e)}"
        
        # Clean the extracted text
        cleaned_text = self.clean_text(text)
        if not cleaned_text.strip():
            return "Error: No readable text found in the file. The document may be image-based or corrupted."
            
        return cleaned_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\@\+\#]', '', text)
        return text.strip()
    
    def extract_email(self, text: str) -> str:
        """Extract email from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else ""
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        phone_pattern = r'[\+KATEX_INLINE_OPEN]?[1-9][0-9 .\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE]{8,}[0-9]'
        phones = re.findall(phone_pattern, text)
        return phones[0] if phones else ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text"""
        # Common technical skills (expand this list)
        skill_patterns = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb',
            'machine learning', 'deep learning', 'data science', 'aws', 'docker',
            'kubernetes', 'git', 'agile', 'scrum', 'tensorflow', 'pytorch'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_patterns:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def parse_sections(self, text: str) -> Dict[str, str]:
        """Parse resume into sections"""
        sections = {
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': ''
        }
        
        # Simple section extraction (can be improved with ML)
        section_headers = {
            'education': ['education', 'academic', 'qualification'],
            'experience': ['experience', 'work history', 'employment'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            for section, headers in section_headers.items():
                if any(header in line_lower for header in headers):
                    current_section = section
                    break
            
            # Add content to current section
            if current_section:
                sections[current_section] += line + '\n'
        
        return sections