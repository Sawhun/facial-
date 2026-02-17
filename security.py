"""
Security Module for MediCare Attendance System
Implements comprehensive security measures to prevent OWASP Top 10 vulnerabilities
"""

import os
import re
import hashlib
import hmac
import secrets
from functools import wraps
from datetime import datetime, timedelta
from flask import request, jsonify, session
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import bleach

# =====================
# ENCRYPTION AT REST
# =====================

class EncryptionManager:
    """Manages encryption/decryption of sensitive data (face embeddings)"""

    def __init__(self, master_key=None):
        """
        Initialize encryption manager with a master key.
        Master key should be stored securely (environment variable or key management service)
        """
        if master_key is None:
            # Get from environment or generate (store this securely!)
            master_key = os.environ.get('ENCRYPTION_MASTER_KEY')
            if not master_key:
                # Generate a new key (ONLY for first-time setup)
                # In production, this should be stored in a secure key vault
                master_key = Fernet.generate_key().decode()
                print("=" * 60)
                print("WARNING: NEW ENCRYPTION KEY GENERATED")
                print("=" * 60)
                print(f"Master Key: {master_key}")
                print("\nIMPORTANT: Store this key securely!")
                print("Set it as environment variable: ENCRYPTION_MASTER_KEY")
                print("=" * 60)

        if isinstance(master_key, str):
            master_key = master_key.encode()

        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'medicare_attendance_2025',  # Static salt for deterministic key
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(master_key)

        # Use Fernet (symmetric encryption) for data at rest
        from base64 import urlsafe_b64encode
        self.cipher = Fernet(urlsafe_b64encode(key))

    def encrypt_embedding(self, embedding_bytes):
        """Encrypt face embedding data"""
        if not embedding_bytes:
            return None
        return self.cipher.encrypt(embedding_bytes)

    def decrypt_embedding(self, encrypted_bytes):
        """Decrypt face embedding data"""
        if not encrypted_bytes:
            return None
        try:
            return self.cipher.decrypt(encrypted_bytes)
        except Exception as e:
            print(f"Decryption error: {e}")
            return None


# Global encryption manager instance
_encryption_manager = None

def get_encryption_manager():
    """Get or create the global encryption manager"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


# =====================
# RATE LIMITING & BRUTE FORCE PROTECTION
# =====================

class RateLimiter:
    """In-memory rate limiter to prevent brute force attacks"""

    def __init__(self):
        # Store: {ip_or_key: [(timestamp, count), ...]}
        self.attempts = {}
        self.lockouts = {}  # {ip_or_key: lockout_until_timestamp}

    def is_locked_out(self, key):
        """Check if key is currently locked out"""
        if key in self.lockouts:
            if datetime.now() < self.lockouts[key]:
                return True
            else:
                # Lockout expired
                del self.lockouts[key]
        return False

    def record_attempt(self, key, max_attempts=5, window_seconds=300, lockout_seconds=900):
        """
        Record an attempt and check if rate limit exceeded
        Returns: (is_allowed, remaining_attempts, lockout_time)
        """
        now = datetime.now()

        # Check if locked out
        if self.is_locked_out(key):
            remaining_time = (self.lockouts[key] - now).seconds
            return False, 0, remaining_time

        # Clean old attempts outside the window
        if key in self.attempts:
            self.attempts[key] = [
                (ts, count) for ts, count in self.attempts[key]
                if (now - ts).seconds < window_seconds
            ]
        else:
            self.attempts[key] = []

        # Add current attempt
        self.attempts[key].append((now, 1))

        # Count total attempts in window
        total_attempts = sum(count for ts, count in self.attempts[key])

        # Check if limit exceeded
        if total_attempts >= max_attempts:
            # Lock out the key
            self.lockouts[key] = now + timedelta(seconds=lockout_seconds)
            return False, 0, lockout_seconds

        remaining = max_attempts - total_attempts
        return True, remaining, 0

    def reset(self, key):
        """Reset attempts for a key (on successful auth)"""
        if key in self.attempts:
            del self.attempts[key]
        if key in self.lockouts:
            del self.lockouts[key]


# Global rate limiter
_rate_limiter = RateLimiter()

def get_rate_limiter():
    """Get the global rate limiter"""
    return _rate_limiter


# =====================
# INPUT VALIDATION & SANITIZATION
# =====================

class InputValidator:
    """Validates and sanitizes user input to prevent injection attacks"""

    @staticmethod
    def sanitize_string(text, max_length=255):
        """Sanitize string input (remove HTML/scripts)"""
        if not text:
            return ""
        # Remove any HTML tags and scripts
        clean = bleach.clean(str(text), tags=[], strip=True)
        return clean[:max_length]

    @staticmethod
    def validate_employee_id(employee_id):
        """Validate employee ID format (alphanumeric, dashes, underscores)"""
        if not employee_id:
            return False, "Employee ID is required"

        # Allow alphanumeric, dashes, underscores only (prevent SQL injection)
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', employee_id):
            return False, "Employee ID must be alphanumeric (1-50 characters)"

        return True, None

    @staticmethod
    def validate_name(name):
        """Validate name (letters, spaces, basic punctuation)"""
        if not name:
            return False, "Name is required"

        # Allow letters, spaces, hyphens, apostrophes
        if not re.match(r"^[a-zA-Z\s\-']{2,100}$", name):
            return False, "Name must contain only letters (2-100 characters)"

        return True, None

    @staticmethod
    def validate_username(username):
        """Validate username for admin login"""
        if not username:
            return False, "Username is required"

        # Alphanumeric and underscore only
        if not re.match(r'^[a-zA-Z0-9_]{3,80}$', username):
            return False, "Username must be alphanumeric (3-80 characters)"

        return True, None

    @staticmethod
    def validate_password(password):
        """Validate password strength"""
        if not password:
            return False, "Password is required"

        if len(password) < 8:
            return False, "Password must be at least 8 characters"

        # Check for complexity (at least one uppercase, lowercase, digit)
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"

        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"

        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one digit"

        return True, None

    @staticmethod
    def validate_department(department):
        """Validate department selection"""
        valid_departments = [
            'Cardiology', 'Neurology', 'Pediatrics', 'Radiology',
            'Emergency', 'Surgery', 'Nursing', 'Pharmacy',
            'Laboratory', 'Administration', 'IT Support'
        ]

        if department not in valid_departments:
            return False, "Invalid department selected"

        return True, None


# =====================
# SECURITY HEADERS
# =====================

def add_security_headers(response, strict_mode=False):
    """
    Add comprehensive security headers to prevent common attacks

    Args:
        strict_mode: If True, applies very strict CSP and HSTS headers
                    If False (default), uses relaxed headers for development
    """

    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Changed from DENY to allow iframes from same origin

    # Prevent MIME sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # Enable XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # Content Security Policy (relaxed for CDN resources)
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com https://code.jquery.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com https://unpkg.com; "
        "img-src 'self' data: blob: https:; "
        "font-src 'self' data: https://fonts.gstatic.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "connect-src 'self'; "
        "media-src 'self' blob:; "
        "frame-ancestors 'none'; "
    )
    # Only apply CSP in strict mode to avoid breaking development
    if strict_mode:
        response.headers['Content-Security-Policy'] = csp

    # Strict Transport Security (HTTPS only) - only in strict mode
    if strict_mode:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions Policy (disable unnecessary features)
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), payment=()'

    return response


# =====================
# SESSION SECURITY
# =====================

def configure_secure_session(app):
    """Configure secure session settings"""

    # Use a strong secret key from environment
    app.config['SECRET_KEY'] = os.environ.get(
        'SESSION_SECRET',
        secrets.token_hex(32)  # Generate if not set
    )

    # Session cookie security
    app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only (disable in dev if no HTTPS)
    app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

    # Session timeout (30 minutes)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

    # Regenerate session on login to prevent fixation
    return app


# =====================
# AUDIT LOGGING
# =====================

class AuditLogger:
    """Log security-relevant events for monitoring and compliance"""

    def __init__(self, log_file='logs/security_audit.log'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_event(self, event_type, user=None, ip=None, details=None, severity='INFO'):
        """Log a security event"""
        timestamp = datetime.now().isoformat()
        user_str = user or 'anonymous'
        ip_str = ip or 'unknown'
        details_str = details or ''

        log_entry = f"[{timestamp}] [{severity}] {event_type} | User: {user_str} | IP: {ip_str} | {details_str}\n"

        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Audit log error: {e}")

    def log_login_attempt(self, username, success, ip):
        """Log login attempt"""
        event = "LOGIN_SUCCESS" if success else "LOGIN_FAILED"
        severity = "INFO" if success else "WARNING"
        self.log_event(event, user=username, ip=ip, severity=severity)

    def log_auth_failure(self, username, reason, ip):
        """Log authentication failure"""
        self.log_event("AUTH_FAILURE", user=username, ip=ip,
                      details=reason, severity="WARNING")

    def log_lockout(self, identifier, ip):
        """Log account/IP lockout"""
        self.log_event("LOCKOUT", user=identifier, ip=ip,
                      details="Rate limit exceeded", severity="CRITICAL")

    def log_suspicious_activity(self, activity, user, ip):
        """Log suspicious activity"""
        self.log_event("SUSPICIOUS", user=user, ip=ip,
                      details=activity, severity="CRITICAL")

    def log_data_access(self, resource, user, ip):
        """Log access to sensitive data"""
        self.log_event("DATA_ACCESS", user=user, ip=ip,
                      details=resource, severity="INFO")


# Global audit logger
_audit_logger = None

def get_audit_logger():
    """Get the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# =====================
# CSRF PROTECTION
# =====================

def generate_csrf_token():
    """Generate a CSRF token for forms"""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']

def validate_csrf_token(token):
    """Validate CSRF token"""
    session_token = session.get('csrf_token')
    if not session_token:
        return False

    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(session_token, token)


# =====================
# HELPER FUNCTIONS
# =====================

def get_client_ip():
    """Get client IP address (handling proxies)"""
    # Check for forwarded IP (if behind proxy)
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr


def rate_limit_key(prefix=''):
    """Generate a rate limit key based on IP and optional prefix"""
    ip = get_client_ip()
    return f"{prefix}:{ip}"
