"""
Authentication & RBAC for SupplyGuard AI.
Uses streamlit-authenticator for session management.
Roles: admin, analyst, viewer — defined in config.py.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional
import streamlit as st
import yaml

from config import cfg, ROLES, DEMO_USERS


# ─────────────────────────────────────────────────────────────────
# Password utilities
# ─────────────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """Simple HMAC-SHA256 hash (in production use bcrypt via streamlit-authenticator)."""
    return hmac.new(cfg.secret_key.encode(), password.encode(), hashlib.sha256).hexdigest()


# Simple in-memory user store for demo (replace with DB in production)
_DEMO_CREDENTIALS = {
    "admin": {"password_plain": "Admin@1234", "name": "Admin User", "role": "admin"},
    "analyst": {"password_plain": "Analyst@1234", "name": "Alice Analyst", "role": "analyst"},
    "viewer": {"password_plain": "Viewer@1234", "name": "Victor Viewer", "role": "viewer"},
}


def _verify_credentials(username: str, password: str) -> Optional[dict]:
    """Verify username/password. Returns user dict or None."""
    user = _DEMO_CREDENTIALS.get(username)
    if user and password == user["password_plain"]:
        return {"username": username, "name": user["name"], "role": user["role"]}
    return None


# ─────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────
SESSION_USER_KEY = "_auth_user"
SESSION_LOGIN_TIME_KEY = "_auth_login_time"
SESSION_TIMEOUT_SECONDS = 3600 * 8   # 8 hours


def init_auth_state():
    """Initialise auth keys in session_state if not present."""
    if SESSION_USER_KEY not in st.session_state:
        st.session_state[SESSION_USER_KEY] = None
    if SESSION_LOGIN_TIME_KEY not in st.session_state:
        st.session_state[SESSION_LOGIN_TIME_KEY] = None


def is_authenticated() -> bool:
    """Check if a valid session exists (not expired)."""
    user = st.session_state.get(SESSION_USER_KEY)
    login_time = st.session_state.get(SESSION_LOGIN_TIME_KEY)
    if not user or not login_time:
        return False
    if time.time() - login_time > SESSION_TIMEOUT_SECONDS:
        logout()
        return False
    return True


def get_current_user() -> Optional[dict]:
    """Return current user dict or None."""
    if is_authenticated():
        return st.session_state.get(SESSION_USER_KEY)
    return None


def get_user_role() -> Optional[str]:
    user = get_current_user()
    return user["role"] if user else None


def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission."""
    role = get_user_role()
    if not role:
        return False
    return permission in ROLES.get(role, {}).get("permissions", [])


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Attempt login. Returns (success, message).
    Implements brute-force protection via session-level attempt tracking.
    """
    # Brute-force protection
    attempts_key = "_login_attempts"
    lockout_key = "_login_lockout_until"
    max_attempts = 5
    lockout_seconds = 300  # 5 minutes

    now = time.time()
    lockout_until = st.session_state.get(lockout_key, 0)
    if now < lockout_until:
        remaining = int(lockout_until - now)
        return False, f"Too many failed attempts. Try again in {remaining}s."

    # Validate input
    if not username or not password:
        return False, "Username and password are required."
    if len(username) > 64 or len(password) > 128:
        return False, "Invalid credentials."

    # Strip whitespace (input sanitisation)
    username = username.strip().lower()

    user = _verify_credentials(username, password)
    if user:
        st.session_state[SESSION_USER_KEY] = user
        st.session_state[SESSION_LOGIN_TIME_KEY] = now
        st.session_state[attempts_key] = 0
        return True, f"Welcome, {user['name']}!"
    else:
        attempts = st.session_state.get(attempts_key, 0) + 1
        st.session_state[attempts_key] = attempts
        if attempts >= max_attempts:
            st.session_state[lockout_key] = now + lockout_seconds
            return False, f"Account locked for {lockout_seconds // 60} minutes."
        remaining = max_attempts - attempts
        return False, f"Invalid credentials. {remaining} attempts remaining."


def logout():
    """Clear session."""
    st.session_state[SESSION_USER_KEY] = None
    st.session_state[SESSION_LOGIN_TIME_KEY] = None


# ─────────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────────
def render_login_page():
    """Render the full-page login UI."""
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <div style="font-size:3rem;">🛡️</div>
        <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.2rem; 
                   color:#F97316; margin:0.5rem 0 0.2rem 0;">SupplyGuard AI</h1>
        <p style="color:#94A3B8; font-size:1rem; margin:0;">
            Supply Chain Disruption Risk Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="_login_username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="_login_password"
        )

        if st.button("Sign In →", type="primary", use_container_width=True):
            success, msg = login(username, password)
            if success:
                st.success(msg)
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

        st.markdown("""
        <div style="background:#0F172A; border:1px solid #1E293B; border-radius:8px; 
                    padding:1rem; margin-top:1rem;">
            <p style="color:#64748B; font-size:0.8rem; margin:0 0 0.5rem 0;">
                <strong style="color:#94A3B8;">Demo Credentials</strong>
            </p>
            <p style="color:#64748B; font-size:0.78rem; margin:0; font-family:monospace;">
                admin / Admin@1234 &nbsp;|&nbsp; analyst / Analyst@1234 &nbsp;|&nbsp; viewer / Viewer@1234
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_user_badge():
    """Render user info + logout in sidebar."""
    user = get_current_user()
    if not user:
        return
    role_label = ROLES.get(user["role"], {}).get("label", user["role"])
    role_colors = {"admin": "#EF4444", "analyst": "#F97316", "viewer": "#22C55E"}
    color = role_colors.get(user["role"], "#94A3B8")

    st.sidebar.markdown(f"""
    <div style="background:#1E293B; border-radius:8px; padding:0.75rem; margin-bottom:1rem;">
        <div style="color:#F1F5F9; font-weight:600; font-size:0.9rem;">👤 {user['name']}</div>
        <div style="color:{color}; font-size:0.75rem; margin-top:2px;">
            ● {role_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()
        st.rerun()


def require_permission(permission: str):
    """Decorator/guard — call at top of page to enforce RBAC."""
    if not has_permission(permission):
        st.error(
            f"🚫 Access Denied: Your role ({get_user_role()}) does not have "
            f"'{permission}' permission."
        )
        st.stop()
