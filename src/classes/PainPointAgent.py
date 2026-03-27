import os
import re
import json
import requests

from status import *
from config import *
import llm_provider


CACHE_FILE = os.path.join(ROOT_DIR, ".mp", "pain_points.json")


def _load_cache() -> list:
    if not os.path.exists(CACHE_FILE):
        return []
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def _save_cache(data: list) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


class PainPointAgent:
    """
    Multi-step LLM agent that researches a company, identifies workflow pain
    points, generates solutions, and drafts a personalised outreach email.

    Pipeline:
        1. Scrape company website (homepage + /about + /services)
        2. Identify industry/business type via LLM
        3. Surface top 3 workflow pain points via LLM
        4. Generate a concrete solution for each pain point via LLM
        5. Draft a short, personalised cold-outreach email via LLM
        6. (Optional) Send the email via SMTP
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_html(self, html: str) -> str:
        """Remove HTML tags and collapse whitespace."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_page(self, url: str, timeout: int = 10) -> str:
        """Return stripped text from a URL, or empty string on failure."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; PainPointAgent/1.0)"}
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return self._strip_html(r.text)
        except Exception:
            pass
        return ""

    def _parse_numbered_list(self, text: str, limit: int = 3) -> list:
        """Extract lines from a numbered list, stripping the leading number."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        results = []
        for line in lines:
            clean = re.sub(r"^[\d]+[\.\):\-]\s*", "", line).strip()
            if clean and len(clean) > 10:
                results.append(clean)
            if len(results) == limit:
                break
        return results

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def scrape_website(self, url: str) -> str:
        """
        Attempt to scrape the homepage and common sub-pages to gather
        enough context for the LLM analysis.
        """
        base = url.rstrip("/")
        pages = [base, base + "/about", base + "/services", base + "/about-us"]
        content = ""
        for page in pages:
            chunk = self._fetch_page(page)
            if chunk:
                content += chunk + " "
            if len(content) > 5000:
                break
        return content[:5000]

    def identify_industry(self, company_name: str, content: str) -> str:
        """Ask the LLM to describe the company's industry in 1–2 sentences."""
        prompt = (
            f"Based on the following website content from '{company_name}', "
            "describe the company's industry and what they do in 1-2 sentences. "
            "Be specific.\n\n"
            f"Content: {content[:2000]}\n\n"
            "Industry description:"
        )
        return llm_provider.generate_text(prompt)

    def find_pain_points(
        self, company_name: str, industry: str, content: str
    ) -> list:
        """
        Ask the LLM for the top 3 workflow/operational pain points this
        company likely faces.
        """
        prompt = (
            f"You are a business consultant analysing '{company_name}', "
            f"a {industry} business.\n\n"
            "Based on the website content below and your knowledge of this industry, "
            "identify the top 3 workflow or operational pain points this company "
            "most likely struggles with. Focus on concrete, fixable problems "
            "(e.g. manual processes, poor lead follow-up, lack of automation, "
            "slow invoicing, no online booking, etc.).\n\n"
            f"Website content: {content[:2000]}\n\n"
            "List exactly 3 pain points, one per line, numbered 1–3. "
            "Keep each under 20 words:"
        )
        raw = llm_provider.generate_text(prompt)
        points = self._parse_numbered_list(raw, limit=3)
        # Fallback: if parsing failed, return the raw text split by lines
        if not points:
            points = [l.strip() for l in raw.split("\n") if l.strip()][:3]
        return points

    def generate_solutions(self, company_name: str, pain_points: list) -> list:
        """Generate one concrete freelance/consultant solution per pain point."""
        solutions = []
        for pain_point in pain_points:
            prompt = (
                f"You are pitching your freelance services to '{company_name}'.\n\n"
                f"Their pain point: {pain_point}\n\n"
                "Propose ONE specific, actionable solution you can deliver as a "
                "freelancer or consultant. Be concrete — name the tool, service, "
                "or approach. Keep it to 2 sentences maximum:"
            )
            solution = llm_provider.generate_text(prompt)
            solutions.append(solution)
        return solutions

    def generate_outreach_email(
        self, company_name: str, pain_points: list, solutions: list
    ) -> str:
        """Draft a short, personalised cold-outreach email body."""
        # Pick the most impactful pain point / solution pair for the hook
        primary_pain = pain_points[0] if pain_points else "inefficient workflows"
        primary_solution = solutions[0] if solutions else "a custom automation solution"

        prompt = (
            f"Write a short, professional cold-outreach email body to '{company_name}'.\n\n"
            f"Lead with this specific pain point: {primary_pain}\n"
            f"Offer this solution: {primary_solution}\n\n"
            "Rules:\n"
            "- Under 120 words\n"
            "- Conversational, not corporate\n"
            "- Do NOT write a subject line\n"
            "- End with a clear call to action (15-minute call)\n"
            "- Sign off as 'Alex'\n\n"
            "Email body:"
        )
        return llm_provider.generate_text(prompt)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def research_company(
        self,
        company_name: str,
        website_url: str = "",
        on_progress=None,
    ) -> dict:
        """
        Run the full pipeline for one company and return a result dict.

        Args:
            company_name: Name of the company to research.
            website_url:  Optional URL to scrape for context.
            on_progress:  Optional callable(message: str, stage: str) invoked at
                          each pipeline step — used by the web UI for SSE streaming.
        """
        def _emit(message: str, stage: str = "info"):
            if on_progress:
                on_progress(message, stage)
            else:
                info(f" => {message}")

        _emit(f"Researching '{company_name}'...", "start")

        # Step 1 – scrape
        content = ""
        if website_url:
            _emit(f"Scraping {website_url}...", "scraping")
            content = self.scrape_website(website_url)
            if content:
                _emit(f"Scraped {len(content)} characters.", "scraping")
            else:
                _emit("Could not scrape website. Continuing with name only.", "warning")

        # Step 2 – identify industry
        _emit("Identifying industry...", "industry")
        industry = self.identify_industry(company_name, content)
        _emit(f"Industry identified: {industry[:120]}", "industry_done")

        # Step 3 – pain points
        _emit("Finding pain points...", "pain_points")
        pain_points = self.find_pain_points(company_name, industry, content)
        for i, pp in enumerate(pain_points, 1):
            _emit(f"Pain point {i}: {pp}", "pain_point_item")

        # Step 4 – solutions
        _emit("Generating solutions...", "solutions")
        solutions = self.generate_solutions(company_name, pain_points)
        for i, sol in enumerate(solutions, 1):
            _emit(f"Solution {i} ready.", "solution_item")

        # Step 5 – draft email
        _emit("Drafting outreach email...", "email")
        email_body = self.generate_outreach_email(company_name, pain_points, solutions)
        _emit("Email draft complete.", "email_done")

        return {
            "company_name": company_name,
            "website": website_url,
            "industry": industry,
            "pain_points": pain_points,
            "solutions": solutions,
            "email_body": email_body,
        }

    def display_result(self, result: dict) -> None:
        """Pretty-print the research result to the terminal."""
        sep = "=" * 60
        print(f"\n{sep}")
        success(f" COMPANY: {result['company_name']}", False)
        if result.get("website"):
            info(f" Website: {result['website']}", False)
        print(sep)

        info("\n INDUSTRY", False)
        print(f"  {result['industry']}\n")

        info(" PAIN POINTS & SOLUTIONS", False)
        for i, (pp, sol) in enumerate(
            zip(result["pain_points"], result["solutions"]), 1
        ):
            print(f"\n  [{i}] Pain Point:")
            print(f"      {pp}")
            print(f"      Solution:")
            print(f"      {sol}")

        info("\n OUTREACH EMAIL DRAFT", False)
        print("-" * 60)
        print(result["email_body"])
        print("-" * 60)

    # ------------------------------------------------------------------
    # Interactive menu
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Interactive entry point called from main.py."""
        info("\n====== PAIN POINT AGENT ======", False)
        info(" Researches companies, surfaces workflow pain points,", False)
        info(" generates solutions, and drafts personalised emails.", False)
        info("==============================\n", False)

        while True:
            info("OPTIONS", False)
            print("  1. Research a single company")
            print("  2. Batch research from a file")
            print("  3. View saved results")
            print("  4. Back\n")

            choice = question("Select an option: ").strip()

            if choice == "1":
                self._run_single()

            elif choice == "2":
                self._run_batch()

            elif choice == "3":
                self._view_saved()

            elif choice == "4":
                break

            else:
                warning("Invalid option. Try again.")

    def _run_single(self) -> None:
        company_name = question(" => Company name: ").strip()
        if not company_name:
            warning("Company name cannot be empty.")
            return

        website_url = question(" => Website URL (leave blank to skip): ").strip()

        result = self.research_company(company_name, website_url)
        self.display_result(result)

        # Optional: send email
        if result.get("email_body"):
            send = question("\n Send this email? (Yes/No): ").strip().lower()
            if send == "yes":
                self._send_email(result)

        # Save to cache
        cache = _load_cache()
        cache.append(result)
        _save_cache(cache)
        success(" => Result saved to cache.")

    def _run_batch(self) -> None:
        """
        Read a plain-text file where each line is:
            Company Name, https://website.com
        or just:
            Company Name
        """
        file_path = question(
            " => Path to file (one company per line, optionally: Name, URL): "
        ).strip()

        if not os.path.exists(file_path):
            error(f" => File not found: {file_path}")
            return

        with open(file_path, "r", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            warning(" => File is empty.")
            return

        info(f" => Found {len(lines)} companies. Starting batch research...\n")

        cache = _load_cache()
        email_creds = get_email_credentials()
        auto_send = False

        if email_creds.get("username") and email_creds.get("password"):
            auto_send_input = question(
                " => Auto-send outreach emails to each company? (Yes/No): "
            ).strip().lower()
            auto_send = auto_send_input == "yes"

        for line in lines:
            parts = [p.strip() for p in line.split(",", 1)]
            company_name = parts[0]
            website_url = parts[1] if len(parts) > 1 else ""

            try:
                result = self.research_company(company_name, website_url)
                self.display_result(result)
                cache.append(result)
                _save_cache(cache)

                if auto_send and result.get("email_body"):
                    self._send_email(result)

            except Exception as e:
                error(f" => Failed for '{company_name}': {e}")
                continue

        success(f"\n => Batch complete. {len(lines)} companies researched.")

    def _view_saved(self) -> None:
        cache = _load_cache()
        if not cache:
            warning(" => No saved results found.")
            return

        info(f" => {len(cache)} saved result(s):\n", False)
        for i, r in enumerate(cache, 1):
            print(f"  {i}. {r['company_name']} — {r.get('website', 'no URL')}")

        choice = question("\n Select a result to view (or Enter to go back): ").strip()
        if not choice:
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cache):
                self.display_result(cache[idx])
            else:
                warning("Invalid selection.")
        except ValueError:
            warning("Invalid input.")

    def _send_email(self, result: dict) -> None:
        """Send the drafted email using configured SMTP credentials."""
        try:
            import yagmail
        except ImportError:
            error(" => yagmail is not installed. Cannot send email.")
            return

        email_creds = get_email_credentials()
        if not email_creds.get("username") or not email_creds.get("password"):
            warning(
                " => Email credentials not configured in config.json. Skipping send."
            )
            return

        recipient = question(
            f" => Recipient email for {result['company_name']}: "
        ).strip()
        if not recipient or "@" not in recipient:
            warning(" => Invalid email address. Skipping.")
            return

        subject = f"Quick question for {result['company_name']}"

        try:
            yag = yagmail.SMTP(
                user=email_creds["username"],
                password=email_creds["password"],
                host=email_creds.get("smtp_server", "smtp.gmail.com"),
                port=email_creds.get("smtp_port", 587),
            )
            yag.send(to=recipient, subject=subject, contents=result["email_body"])
            success(f" => Email sent to {recipient}")
        except Exception as e:
            error(f" => Failed to send email: {e}")
