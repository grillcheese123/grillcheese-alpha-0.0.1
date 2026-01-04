import logging
from typing import Optional, List, Dict
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

class MultilingualProcessor:
    """Handle multilingual text processing and language detection"""
    
    # ISO 639-1 language codes mapped to full names
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'it': 'Italian',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'hi': 'Hindi'
    }
    
    def __init__(self, primary_language: str = 'en'):
        self.primary_language = primary_language
        self.detector = None
        self._init_detector()
        
    def _init_detector(self):
        """Initialize fast language detector"""
        try:
            # Build detector with common languages
            languages = [
                Language.ENGLISH, Language.SPANISH, Language.FRENCH,
                Language.GERMAN, Language.CHINESE, Language.JAPANESE,
                Language.KOREAN, Language.ARABIC, Language.RUSSIAN,
                Language.PORTUGUESE, Language.ITALIAN
            ]
            
            self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
            logger.info("Language detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize language detector: {e}")
            
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text, return ISO 639-1 code"""
        if not self.detector:
            return self.primary_language
            
        try:
            language = self.detector.detect_language_of(text)
            if language:
                return self._lingua_to_iso(language)
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            
        return self.primary_language
        
    def detect_with_confidence(self, text: str) -> Dict[str, float]:
        """Detect language with confidence scores"""
        if not self.detector:
            return {self.primary_language: 1.0}
            
        try:
            confidences = self.detector.compute_language_confidence_values(text)
            return {
                self._lingua_to_iso(lang): conf 
                for lang, conf in confidences
            }
        except Exception as e:
            logger.debug(f"Language confidence detection failed: {e}")
            return {self.primary_language: 1.0}
            
    def is_multilingual(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text contains multiple languages"""
        confidences = self.detect_with_confidence(text)
        
        # Count languages with confidence > threshold
        high_conf_langs = [lang for lang, conf in confidences.items() 
                          if conf > threshold]
        
        return len(high_conf_langs) > 1
        
    def get_language_name(self, code: str) -> str:
        """Get full language name from ISO code"""
        return self.SUPPORTED_LANGUAGES.get(code, code.upper())
        
    def normalize_text(self, text: str, language: str = None) -> str:
        """Normalize text for better embedding quality"""
        if language is None:
            language = self.detect_language(text)
            
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Language-specific normalization
        if language == 'zh' or language == 'ja':
            # Remove spaces between CJK characters
            text = self._normalize_cjk(text)
        elif language == 'ar':
            # Normalize Arabic diacritics
            text = self._normalize_arabic(text)
            
        return text
        
    def split_by_language(self, text: str) -> List[Dict[str, str]]:
        """Split mixed-language text into segments"""
        # Simple sentence-based splitting
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        segments = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            lang = self.detect_language(sent)
            segments.append({
                'text': sent,
                'language': lang,
                'name': self.get_language_name(lang)
            })
            
        return segments
        
    def _lingua_to_iso(self, language) -> str:
        """Convert Lingua language enum to ISO 639-1 code"""
        lang_map = {
            Language.ENGLISH: 'en',
            Language.SPANISH: 'es',
            Language.FRENCH: 'fr',
            Language.GERMAN: 'de',
            Language.CHINESE: 'zh',
            Language.JAPANESE: 'ja',
            Language.KOREAN: 'ko',
            Language.ARABIC: 'ar',
            Language.RUSSIAN: 'ru',
            Language.PORTUGUESE: 'pt',
            Language.ITALIAN: 'it',
            Language.DUTCH: 'nl',
            Language.POLISH: 'pl',
            Language.TURKISH: 'tr'
        }
        return lang_map.get(language, 'en')
        
    def _normalize_cjk(self, text: str) -> str:
        """Normalize Chinese/Japanese/Korean text"""
        # Remove spaces between CJK characters
        import re
        cjk_pattern = re.compile(r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])\s+([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])')
        
        while cjk_pattern.search(text):
            text = cjk_pattern.sub(r'\1\2', text)
            
        return text
        
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text by removing diacritics"""
        import re
        # Remove Arabic diacritical marks
        arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        return arabic_diacritics.sub('', text)
        
    @staticmethod
    def get_script(text: str) -> str:
        """Detect writing script"""
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return 'Han'
        elif any('\u3040' <= c <= '\u309f' for c in text):
            return 'Hiragana'
        elif any('\u30a0' <= c <= '\u30ff' for c in text):
            return 'Katakana'
        elif any('\uac00' <= c <= '\ud7af' for c in text):
            return 'Hangul'
        elif any('\u0600' <= c <= '\u06ff' for c in text):
            return 'Arabic'
        elif any('\u0400' <= c <= '\u04ff' for c in text):
            return 'Cyrillic'
        else:
            return 'Latin'
