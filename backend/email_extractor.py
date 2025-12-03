"""
Email Extractor Module
Extracts email addresses and identifies providers using regex pattern matching.
"""

import re


def extract_email_and_provider(text: str) -> dict:
    """
    Extract the first valid email address from text and identify its provider.
    
    Args:
        text: Input text containing email address(es)
    
    Returns:
        dict with 'email' and 'provider' keys
    """
    if not text:
        return {"email": None, "provider": "unknown"}
    
    # Regex pattern for email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    
    # Find first email
    match = re.search(email_pattern, text)
    
    if not match:
        return {"email": None, "provider": "unknown"}
    
    email = match.group().lower()
    domain = email.split('@')[1]
    
    # Classify provider
    if domain == 'gmail.com':
        provider = 'Gmail'
    elif domain in ('outlook.com', 'hotmail.com', 'live.com'):
        provider = 'Outlook'
    elif domain in ('yahoo.com', 'yahoo.in'):
        provider = 'Yahoo'
    elif domain == 'zoho.com':
        provider = 'Zoho'
    else:
        provider = 'Company Email'
    
    return {"email": email, "provider": provider}


def extract_all_emails(text: str) -> list:
    """
    Extract all email addresses from text with their providers.
    
    Args:
        text: Input text containing email address(es)
    
    Returns:
        list of dicts with 'email' and 'provider' keys
    """
    if not text:
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    
    results = []
    seen = set()
    
    for email in matches:
        email_lower = email.lower()
        if email_lower in seen:
            continue
        seen.add(email_lower)
        
        domain = email_lower.split('@')[1]
        
        if domain == 'gmail.com':
            provider = 'Gmail'
        elif domain in ('outlook.com', 'hotmail.com', 'live.com'):
            provider = 'Outlook'
        elif domain in ('yahoo.com', 'yahoo.in'):
            provider = 'Yahoo'
        elif domain == 'zoho.com':
            provider = 'Zoho'
        else:
            provider = 'Company Email'
        
        results.append({"email": email_lower, "provider": provider})
    
    return results


# Example usage
if __name__ == "__main__":
    test_texts = [
        "Contact me at john.doe@gmail.com for more info",
        "Send to sarah@outlook.com or backup@hotmail.com",
        "Business email: ceo@acme-corp.com",
        "Reach out: user123@yahoo.in",
        "No email here!",
        "Multiple: test@zoho.com and work@live.com"
    ]
    
    print("=== Single Email Extraction ===")
    for text in test_texts:
        result = extract_email_and_provider(text)
        print(f"Input: {text}")
        print(f"Email: {result['email']}")
        print(f"Provider: {result['provider']}")
        print("---")
    
    print("\n=== All Emails Extraction ===")
    multi_text = "Contact john@gmail.com, sarah@outlook.com, or admin@company.org"
    results = extract_all_emails(multi_text)
    print(f"Input: {multi_text}")
    for r in results:
        print(f"  Email: {r['email']} | Provider: {r['provider']}")

