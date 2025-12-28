"""
Professional error handling for web crawling operations.
"""
import streamlit as st

def display_crawl_error(error_type: str, url: str, details: str = ""):
    """
    Display beautiful, informative error messages for crawl operations.
    
    Args:
        error_type: Type of error (robots_blocked, timeout, not_found, etc.)
        url: The URL that caused the error
        details: Additional error details
    """
    
    error_configs = {
        'robots_blocked': {
            'icon': '🚫',
            'title': 'Crawling Blocked by robots.txt',
            'message': f'The website **{url}** does not allow automated crawling.',
            'suggestion': 'Try:\n- Manually copying the text\n- Using a different URL from the same site\n- Checking if the site has an official API',
            'severity': 'warning'
        },
        'timeout': {
            'icon': '⏱️',
            'title': 'Request Timeout',
            'message': f'The website **{url}** took too long to respond (>30s).',
            'suggestion': 'Try:\n- Reducing crawl depth\n- Trying again in a few minutes\n- Checking if the site is down',
            'severity': 'error'
        },
        'not_found': {
            'icon': '❌',
            'title': 'Page Not Found (404)',
            'message': f'The URL **{url}** does not exist.',
            'suggestion': 'Try:\n- Checking for typos in the URL\n- Visiting the homepage first\n- Using the Wayback Machine for archived versions',
            'severity': 'error'
        },
        'no_content': {
            'icon': '📭',
            'title': 'No Text Content Found',
            'message': f'Could not extract meaningful text from **{url}**.',
            'suggestion': 'The page might be:\n- Image or video heavy\n- Behind a login wall\n- Requiring JavaScript (will try headless browser next)',
            'severity': 'warning'
        },
        'ssl_error': {
            'icon': '🔒',
            'title': 'SSL Certificate Error',
            'message': f'Secure connection to **{url}** failed.',
            'suggestion': 'The site may have:\n- Expired SSL certificate\n- Security issues\n- Strict HTTPS requirements',
            'severity': 'error'
        },
        'rate_limit': {
            'icon': '⚠️',
            'title': 'Rate Limited',
            'message': f'Too many requests to **{url}**. Cooling down...',
            'suggestion': 'Waiting 60 seconds before retrying.',
            'severity': 'warning'
        },
        'javascript_required': {
            'icon': '⚙️',
            'title': 'JavaScript Required',
            'message': f'The page **{url}** requires JavaScript to load content.',
            'suggestion': '🔄 Automatically switching to headless browser...',
            'severity': 'info'
        }
    }
    
    config = error_configs.get(error_type, {
        'icon': '⚠️',
        'title': 'Unknown Error',
        'message': f'An error occurred: {details}',
        'suggestion': 'Please try again or contact support.',
        'severity': 'error'
    })
    
    # Display based on severity
    if config['severity'] == 'error':
        st.error(f"{config['icon']} **{config['title']}**\n\n{config['message']}")
    elif config['severity'] == 'warning':
        st.warning(f"{config['icon']} **{config['title']}**\n\n{config['message']}")
    else:
        st.info(f"{config['icon']} **{config['title']}**\n\n{config['message']}")
    
    st.info(f"💡 **Suggestion:**\n\n{config['suggestion']}")
    
    # Return True if retry is recommended
    return error_type in ['timeout', 'rate_limit', 'javascript_required']

