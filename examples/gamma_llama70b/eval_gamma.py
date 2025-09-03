import xml.etree.ElementTree as ET
import re
from typing import Tuple, Optional

def is_xml_format(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a string is valid XML format.
    Args:
        text (str): The string to check
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Empty or invalid input"
    # Strip whitespace
    text = text.strip()
    # Basic XML pattern check (starts with < and ends with >)
    if not (text.startswith('<') and text.endswith('>')):
        return False, "Does not start with '<' and end with '>'"
    try:
        # Attempt to parse with ElementTree
        ET.fromstring(text)
        return True, None
    except ET.ParseError as e:
        return False, f"XML Parse Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def is_well_formed_xml(text: str) -> Tuple[bool, Optional[str]]:
    """
    More lenient check for XML-like format (handles fragments).
    Args:
        text (str): The string to check
    Returns:
        Tuple[bool, Optional[str]]: (is_xml_like, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Empty or invalid input"
    text = text.strip()
    # Check for XML-like tags
    xml_tag_pattern = r'<[^>]+>'
    if not re.search(xml_tag_pattern, text):
        return False, "No XML tags found"
    # Try wrapping in root element for fragments
    try:
        wrapped_text = f"<root>{text}</root>"
        ET.fromstring(wrapped_text)
        return True, None
    except ET.ParseError as e:
        return False, f"XML-like Parse Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def analyze_xml_structure(text: str) -> Dict[str, Any]:
    """
    Analyze the structure of XML content.
    Args:
        text (str): XML string to analyze
    Returns:
        Dict with analysis results
    """
    analysis = {
        'is_valid_xml': False,
        'is_xml_fragment': False,
        'has_xml_declaration': False,
        'root_element': None,
        'tag_count': 0,
        'unique_tags': set(),
        'self_closing_tags': 0,
        'errors': []
    }
    if not text:
        analysis['errors'].append("Empty input")
        return analysis
    text = text.strip()
    # Check for XML declaration
    if text.startswith('<?xml'):
        analysis['has_xml_declaration'] = True
    # Count tags
    tag_pattern = r'<([^/>][^>]*?)/?>'
    tags = re.findall(tag_pattern, text)
    analysis['tag_count'] = len(tags)
    # Extract unique tag names
    tag_name_pattern = r'</?([a-zA-Z][a-zA-Z0-9_-]*)'
    tag_names = re.findall(tag_name_pattern, text)
    analysis['unique_tags'] = set(tag_names)
    # Count self-closing tags
    self_closing_pattern = r'<[^>]+/>'
    analysis['self_closing_tags'] = len(re.findall(self_closing_pattern, text))
    # Try parsing as complete XML
    try:
        root = ET.fromstring(text)
        analysis['is_valid_xml'] = True
        analysis['root_element'] = root.tag
    except ET.ParseError:
        # Try as XML fragment
        try:
            wrapped = f"<root>{text}</root>"
            ET.fromstring(wrapped)
            analysis['is_xml_fragment'] = True
        except ET.ParseError as e:
            analysis['errors'].append(f"Parse error: {str(e)}")
    return analysis

# Example usage and test cases
if __name__ == "__main__":
    # Test cases
    test_strings = [
        # Valid XML
        '<?xml version="1.0"?><root><child>content</child></root>',
        '<book><title>Test</title><author>John</author></book>',
        # XML fragment (like your example)
        '<SECTION n=1><H1>Title</H1><P>Content</P></SECTION>',
        
        # Invalid XML
        '<unclosed>tag',
        'not xml at all',
        '<invalid><nested></invalid></nested>',
        
        # Self-closing tags
        '<IMG src="test.jpg" />',
        
        # Your specific format
        '''<SECTION n=1>
        <H1>인터랙티브 스토리텔링의 세계</H1>
        <IMG provider="aiGenerated" query="test" />
        <P>Content here</P>
        </SECTION>'''
    ]
    
    print("=== XML Format Validation Tests ===\n")
    
    for i, test_string in enumerate(test_strings, 1):
        print(f"Test {i}:")
        print(f"Input: {test_string[:50]}{'...' if len(test_string) > 50 else ''}")
        
        # Check if valid XML
        is_valid, error = is_xml_format(test_string)
        print(f"Valid XML: {is_valid}")
        if error:
            print(f"Error: {error}")
        
        # Check if XML-like (more lenient)
        is_xml_like, error2 = is_well_formed_xml(test_string)
        print(f"XML-like: {is_xml_like}")
        
        # Detailed analysis
        analysis = analyze_xml_structure(test_string)
        print(f"Analysis: {json.dumps({k: list(v) if isinstance(v, set) else v 
                                     for k, v in analysis.items() 
                                     if k != 'errors'}, indent=2)}")
        print("-" * 50)

def quick_xml_check(text: str) -> bool:
    """Simple one-liner to check if string is XML format"""
    try:
        ET.fromstring(text.strip())
        return True
    except:
        return False

def xml_fragment_check(text: str) -> bool:
    """Check if string is XML fragment (can be wrapped to become valid XML)"""
    try:
        ET.fromstring(f"<root>{text.strip()}</root>")
        return True
    except:
        return False