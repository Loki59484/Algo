import argparse
import getpass
import http.server
import json
import logging
import socketserver
import sys
import threading
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from playwright.sync_api import Error, sync_playwright
from rich.console import Console
from rich.panel import Panel

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import required config variables from your core methods
from core.upstox_methods import (
    API_KEY,
    API_SECRET,
    MOBILE_NUM,
    SANDBOX_TOKEN_FILE,
    SECRETS_PATH,
    TOKEN_FILE,
)

console = Console()

class OAuthRedirectHandler(http.server.SimpleHTTPRequestHandler):
    """Handles the local redirect from Upstox and captures the auth code."""
    auth_state = {"code": None}

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if parsed_url.path == "/callback" and "code" in query_params:
            OAuthRedirectHandler.auth_state["code"] = query_params.get("code")[0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family: sans-serif; text-align: center; margin-top: 50px;'>"
                b"<h1 style='color: green;'>Authentication Successful!</h1>"
                b"<p>You can safely close this window and return to your terminal.</p>"
                b"</body></html>"
            )
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard HTTP server logs to keep terminal clean
        pass


class UpstoxAuthenticator:
    """Manages the generation of Production and Sandbox tokens."""

    PORT = 5000
    REDIRECT_URI = f"http://localhost:{PORT}/callback"

    @classmethod
    def generate_production_token(cls, show_browser: bool = False):
        """Automates the Upstox login flow via Playwright to fetch an access token."""
        console.print(Panel.fit("[bold cyan]🚀 Initiating Upstox Production Login[/bold cyan]"))
        
        server_ready_event = threading.Event()

        def run_server(httpd):
            socketserver.TCPServer.allow_reuse_address = True
            server_ready_event.set()
            httpd.serve_forever()

        # Start Local Server
        try:
            httpd = socketserver.TCPServer(("", cls.PORT), OAuthRedirectHandler)
        except OSError as e:
            console.print(f"[bold red]Failed to start local server on port {cls.PORT}: {e}[/bold red]")
            sys.exit(1)

        server_thread = threading.Thread(target=run_server, args=(httpd,), daemon=True)
        server_thread.start()

        server_ready_event.wait(timeout=5)
        if not server_ready_event.is_set():
            console.print("[bold red]Error: Local server did not start in time. Exiting.[/bold red]")
            sys.exit(1)

        try:
            login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={cls.REDIRECT_URI}"
            
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=not show_browser)
                except Error:
                    # Fallback for Ubuntu/Snap
                    browser = p.chromium.launch(executable_path="/usr/bin/chromium-browser", headless=not show_browser)

                context = browser.new_context(ignore_https_errors=True)
                page = context.new_page()
                
                console.print("[dim]Navigating to Upstox login portal...[/dim]")
                page.goto(login_url)
                
                console.print(f"[yellow]Sending OTP to registered mobile:[/yellow] {MOBILE_NUM}")
                page.fill("#mobileNum", MOBILE_NUM)
                page.click("#getOtp")
                
                # Interactive Input
                otp = console.input("[bold green]Enter the 6-digit OTP received: [/bold green]")
                page.fill("#otpNum", otp)
                page.click("#continueBtn")
                
                pin = getpass.getpass("Enter your 6-digit Upstox PIN: ")
                page.fill("#pinCode", pin)
                page.click("#pinContinueBtn")
                
                console.print("[dim]Waiting for OAuth callback...[/dim]")
                page.wait_for_url(f"{cls.REDIRECT_URI}*", timeout=150000)
                browser.close()

            # Retrieve code from HTTP handler
            auth_code = OAuthRedirectHandler.auth_state.get("code")
            if not auth_code:
                raise ValueError("Auth code was not successfully captured by the local callback server.")

            # Exchange code for token
            console.print("[dim]Exchanging Auth Code for Access Token...[/dim]")
            token_url = "https://api.upstox.com/v2/login/authorization/token"
            headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "code": auth_code,
                "client_id": API_KEY,
                "client_secret": API_SECRET,
                "redirect_uri": cls.REDIRECT_URI,
                "grant_type": "authorization_code",
            }

            response = requests.post(token_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()

            # Calculate precise Upstox expiry (3:30 AM next valid day)
            now = datetime.now()
            if 0 <= now.hour < 3 or (now.hour == 3 and 0 <= now.minute <= 30):
                expiry_date = now.replace(hour=3, minute=30, second=0, microsecond=0)
            else:
                expiry_date = (now + timedelta(days=1)).replace(hour=3, minute=30, second=0, microsecond=0)
            
            token_data["expiry"] = expiry_date.strftime("%Y-%m-%d %H:%M:%S")

            # Save to disk securely
            SECRETS_PATH.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_FILE, "w") as f:
                json.dump(token_data, f, indent=4)

            console.print(f"[bold green]✅ Production Login Successful![/bold green]")
            console.print(f"Token is valid until: [yellow]{token_data['expiry']}[/yellow]")

        except Exception as e:
            console.print(f"[bold red]❌ Authentication Failed:[/bold red] {e}")
            sys.exit(1)
        finally:
            # Always clean up the server
            httpd.shutdown()
            httpd.server_close()

    @classmethod
    def generate_sandbox_token(cls):
        """Prompts for manual entry of a multiline Sandbox token."""
        console.print(Panel.fit("[bold magenta]🛠️ Upstox Sandbox Configuration[/bold magenta]"))
        console.print(
            "1. Login to [link=https://account.upstox.com/developer/apps#sandbox]Upstox Developer Portal[/link]\n"
            "2. Generate a Sandbox Access Token\n"
            "3. Paste the entire token below (Press [bold]Enter[/bold] on an empty line to finish):"
        )
        
        lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except EOFError:
                break
                
        access_token = "\n".join(lines).strip()
        
        if not access_token:
            console.print("[bold red]❌ No token provided. Aborting.[/bold red]")
            sys.exit(1)

        sandbox_expiry = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        payload = {"sandbox_access_token": access_token, "expiry": sandbox_expiry}
        
        SECRETS_PATH.mkdir(parents=True, exist_ok=True)
        with open(SANDBOX_TOKEN_FILE, "w") as f:
            json.dump(payload, f, indent=4)
            
        console.print("[bold green]✅ Sandbox Token Saved Successfully![/bold green]")
        console.print(f"Token expires in 30 days on: [yellow]{sandbox_expiry}[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upstox Authentication Manager")
    parser.add_argument("--sandbox", action="store_true", help="Generate a Sandbox token instead of a Production token.")
    parser.add_argument("--show-browser", action="store_true", help="Launch Playwright without headless mode (useful for debugging).")
    
    args = parser.parse_args()

    try:
        if args.sandbox:
            UpstoxAuthenticator.generate_sandbox_token()
        else:
            UpstoxAuthenticator.generate_production_token(show_browser=args.show_browser)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Login aborted by user.[/bold yellow]")
        sys.exit(0)