"""
Security utilities for the MultimodalRAG system.
Provides authentication, authorization, and input validation.
"""

import os
import time
import hashlib
import hmac
import base64
import re
import secrets
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from datetime import datetime, timedelta
import jwt
from flask import request, jsonify, g

# Configure logging
logger = logging.getLogger('multimodal_rag.security')

# Load environment variables
API_SECRET_KEY = os.getenv("API_SECRET_KEY", secrets.token_hex(32))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

class SecurityManager:
    """Manages security for the MultimodalRAG system"""
    
    def __init__(self):
        """Initialize the security manager"""
        self.api_keys = {}
        self.load_api_keys()
    
    def load_api_keys(self) -> None:
        """Load API keys from environment or configuration"""
        # Load from environment variables
        env_api_keys = os.getenv("API_KEYS", "")
        if env_api_keys:
            for key_entry in env_api_keys.split(','):
                if ':' in key_entry:
                    key_id, key_secret = key_entry.split(':', 1)
                    self.api_keys[key_id.strip()] = {
                        "key": key_secret.strip(),
                        "roles": ["user"],
                        "rate_limit": 100
                    }
        
        # Add admin key if specified
        admin_key_id = os.getenv("ADMIN_API_KEY_ID")
        admin_key_secret = os.getenv("ADMIN_API_KEY_SECRET")
        if admin_key_id and admin_key_secret:
            self.api_keys[admin_key_id] = {
                "key": admin_key_secret,
                "roles": ["admin", "user"],
                "rate_limit": 1000
            }
    
    def validate_api_key(self, api_key_id: str, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key_id: API key identifier
            api_key: API key to validate
        
        Returns:
            True if API key is valid, False otherwise
        """
        if api_key_id not in self.api_keys:
            return False
        
        stored_key = self.api_keys[api_key_id]["key"]
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_key, api_key)
    
    def get_user_roles(self, api_key_id: str) -> List[str]:
        """
        Get roles for a user.
        
        Args:
            api_key_id: API key identifier
        
        Returns:
            List of roles
        """
        if api_key_id in self.api_keys:
            return self.api_keys[api_key_id].get("roles", ["user"])
        return []
    
    def get_rate_limit(self, api_key_id: str) -> int:
        """
        Get rate limit for a user.
        
        Args:
            api_key_id: API key identifier
        
        Returns:
            Rate limit (requests per minute)
        """
        if api_key_id in self.api_keys:
            return self.api_keys[api_key_id].get("rate_limit", 100)
        return 60  # Default rate limit
    
    def generate_jwt_token(self, api_key_id: str) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            api_key_id: API key identifier
        
        Returns:
            JWT token
        """
        expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        payload = {
            "sub": api_key_id,
            "roles": self.get_user_roles(api_key_id),
            "exp": expiration
        }
        
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token to validate
        
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
    
    def sanitize_input(self, input_data: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            input_data: Input string to sanitize
        
        Returns:
            Sanitized input string
        """
        # Remove potentially dangerous HTML/script tags
        sanitized = re.sub(r'<script.*?>.*?</script>', '', input_data, flags=re.DOTALL)
        sanitized = re.sub(r'<.*?>', '', sanitized)
        
        # Limit input length
        max_length = 10000  # Adjust as needed
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate_file_upload(self, file_data: bytes, file_name: str, 
                           allowed_extensions: List[str], max_size_mb: int = 10) -> bool:
        """
        Validate file upload.
        
        Args:
            file_data: File data
            file_name: File name
            allowed_extensions: List of allowed file extensions
            max_size_mb: Maximum file size in MB
        
        Returns:
            True if file is valid, False otherwise
        """
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.warning(f"File too large: {file_name} ({file_size_mb:.2f} MB)")
            return False
        
        # Check file extension
        file_ext = os.path.splitext(file_name)[1].lower()
        if not file_ext or file_ext[1:] not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_ext}")
            return False
        
        return True


# Flask decorators for API security

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        security_manager = SecurityManager()
        
        # Get API key from header or query parameter
        api_key_id = request.headers.get('X-API-Key-ID') or request.args.get('api_key_id')
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key_id or not api_key:
            return jsonify({"error": "API key required"}), 401
        
        if not security_manager.validate_api_key(api_key_id, api_key):
            return jsonify({"error": "Invalid API key"}), 401
        
        # Store API key ID for rate limiting
        g.api_key_id = api_key_id
        
        return f(*args, **kwargs)
    return decorated

def require_jwt_auth(f):
    """Decorator to require JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        security_manager = SecurityManager()
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "JWT token required"}), 401
        
        token = auth_header.split(' ')[1]
        payload = security_manager.validate_jwt_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        # Store user info in Flask global context
        g.user = payload
        
        return f(*args, **kwargs)
    return decorated

def require_role(role):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, 'user') or role not in g.user.get('roles', []):
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

def rate_limit(f):
    """Decorator to apply rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        from src.monitoring import RequestTracker
        
        # Get client identifier
        client_id = g.api_key_id if hasattr(g, 'api_key_id') else request.remote_addr
        
        # Get rate limit for client
        security_manager = SecurityManager()
        rate_limit = security_manager.get_rate_limit(client_id) if hasattr(g, 'api_key_id') else 60
        
        # Check rate limit
        request_tracker = RequestTracker(window_size=60, max_requests=rate_limit)
        if not request_tracker.check_rate_limit(client_id):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        return f(*args, **kwargs)
    return decorated

def sanitize_request(f):
    """Decorator to sanitize request data"""
    @wraps(f)
    def decorated(*args, **kwargs):
        security_manager = SecurityManager()
        
        # Sanitize JSON data
        if request.is_json:
            sanitized_json = {}
            for key, value in request.json.items():
                if isinstance(value, str):
                    sanitized_json[key] = security_manager.sanitize_input(value)
                else:
                    sanitized_json[key] = value
            
            # Replace request.json with sanitized version
            request._cached_json = (sanitized_json, request._cached_json[1])
        
        # Sanitize form data
        if request.form:
            for key in request.form:
                if isinstance(request.form[key], str):
                    request.form[key] = security_manager.sanitize_input(request.form[key])
        
        return f(*args, **kwargs)
    return decorated

def validate_content_type(content_types):
    """
    Decorator to validate request content type.
    
    Args:
        content_types: List of allowed content types
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.content_type not in content_types:
                return jsonify({
                    "error": f"Unsupported content type: {request.content_type}. "
                             f"Supported types: {', '.join(content_types)}"
                }), 415
            return f(*args, **kwargs)
        return decorated
    return decorator

def validate_file_upload_decorator(allowed_extensions, max_size_mb=10):
    """
    Decorator to validate file uploads.
    
    Args:
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            security_manager = SecurityManager()
            
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read file data for validation
            file_data = file.read()
            file.seek(0)  # Reset file pointer
            
            if not security_manager.validate_file_upload(
                file_data, file.filename, allowed_extensions, max_size_mb
            ):
                return jsonify({
                    "error": f"Invalid file. Allowed extensions: {', '.join(allowed_extensions)}, "
                             f"max size: {max_size_mb}MB"
                }), 400
            
            return f(*args, **kwargs)
        return decorated
    return decorator

class CSRFProtection:
    """CSRF protection for web applications"""
    
    @staticmethod
    def generate_csrf_token() -> str:
        """
        Generate a CSRF token.
        
        Returns:
            CSRF token
        """
        if 'csrf_token' not in g:
            g.csrf_token = secrets.token_hex(16)
        
        return g.csrf_token
    
    @staticmethod
    def validate_csrf_token(token: str) -> bool:
        """
        Validate a CSRF token.
        
        Args:
            token: CSRF token to validate
        
        Returns:
            True if token is valid, False otherwise
        """
        if not token or not hasattr(g, 'csrf_token'):
            return False
        
        return hmac.compare_digest(token, g.csrf_token)
    
    @staticmethod
    def csrf_protect(f):
        """Decorator to require CSRF token for POST/PUT/DELETE requests"""
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'DELETE']:
                token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
                
                if not CSRFProtection.validate_csrf_token(token):
                    return jsonify({"error": "CSRF token missing or invalid"}), 403
            
            return f(*args, **kwargs)
        return decorated

class ContentSecurityPolicy:
    """Content Security Policy (CSP) management"""
    
    def __init__(self):
        """Initialize CSP with default policies"""
        self.policies = {
            'default-src': ["'self'"],
            'script-src': ["'self'"],
            'style-src': ["'self'"],
            'img-src': ["'self'", "data:"],
            'connect-src': ["'self'"],
            'font-src': ["'self'"],
            'object-src': ["'none'"],
            'media-src': ["'self'"],
            'frame-src': ["'none'"]
        }
    
    def add_source(self, directive: str, source: str) -> None:
        """
        Add a source to a CSP directive.
        
        Args:
            directive: CSP directive
            source: Source to add
        """
        if directive in self.policies:
            if source not in self.policies[directive]:
                self.policies[directive].append(source)
        else:
            self.policies[directive] = [source]
    
    def get_header_value(self) -> str:
        """
        Get CSP header value.
        
        Returns:
            CSP header value
        """
        directives = []
        
        for directive, sources in self.policies.items():
            directives.append(f"{directive} {' '.join(sources)}")
        
        return "; ".join(directives)

class SecureHeaders:
    """Security headers management"""
    
    @staticmethod
    def apply_secure_headers(response):
        """
        Apply security headers to response.
        
        Args:
            response: Flask response object
        
        Returns:
            Response with security headers
        """
        # Content Security Policy
        csp = ContentSecurityPolicy()
        response.headers['Content-Security-Policy'] = csp.get_header_value()
        
        # Other security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response

class PasswordManager:
    """Password hashing and validation"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: Password to hash
        
        Returns:
            Hashed password
        """
        # Generate a random salt
        salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
        
        # Hash password with salt
        pwdhash = hashlib.pbkdf2_hmac(
            'sha512', 
            password.encode('utf-8'), 
            salt, 
            100000,
            dklen=128
        )
        
        # Encode for storage
        pwdhash = base64.b64encode(pwdhash).decode('ascii')
        
        # Return salt and hash
        return f"{salt.decode('ascii')}${pwdhash}"
    
    @staticmethod
    def verify_password(stored_password: str, provided_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            stored_password: Stored password hash
            provided_password: Password to verify
        
        Returns:
            True if password matches, False otherwise
        """
        # Split salt and hash
        salt, stored_hash = stored_password.split('$')
        
        # Hash provided password with same salt
        pwdhash = hashlib.pbkdf2_hmac(
            'sha512', 
            provided_password.encode('utf-8'), 
            salt.encode('ascii'), 
            100000,
            dklen=128
        )
        
        # Encode for comparison
        pwdhash = base64.b64encode(pwdhash).decode('ascii')
        
        # Compare hashes using constant-time comparison
        return hmac.compare_digest(pwdhash, stored_hash)

# Initialize security components
security_manager = SecurityManager()
csrf_protection = CSRFProtection()
