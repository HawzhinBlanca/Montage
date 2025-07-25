#!/usr/bin/env python3
"""
P0-06: Billing Dashboard Check Script

This script helps verify billing usage across all API providers to detect
any unauthorized charges from the exposed API keys that were rotated.

For security compliance, this script provides instructions and partial automation
rather than directly accessing billing APIs (which would require additional credentials).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BillingChecker:
    """
    P0-06: Automated billing verification for exposed API key incident

    Provides guided process to check for unauthorized usage across:
    - OpenAI GPT/Whisper APIs
    - Anthropic Claude APIs
    - Google/Gemini APIs
    - Deepgram Speech APIs
    """

    def __init__(self):
        self.providers = {
            "openai": {
                "name": "OpenAI",
                "dashboard_url": "https://platform.openai.com/usage",
                "billing_url": "https://platform.openai.com/account/billing/overview",
                "api_key_env": "OPENAI_API_KEY",
                "risk_level": "HIGH"
            },
            "anthropic": {
                "name": "Anthropic",
                "dashboard_url": "https://console.anthropic.com/dashboard",
                "billing_url": "https://console.anthropic.com/account/billing",
                "api_key_env": "ANTHROPIC_API_KEY",
                "risk_level": "HIGH"
            },
            "google": {
                "name": "Google Cloud/Gemini",
                "dashboard_url": "https://console.cloud.google.com/apis/dashboard",
                "billing_url": "https://console.cloud.google.com/billing",
                "api_key_env": "GEMINI_API_KEY",
                "risk_level": "MEDIUM"
            },
            "deepgram": {
                "name": "Deepgram",
                "dashboard_url": "https://console.deepgram.com/usage",
                "billing_url": "https://console.deepgram.com/billing",
                "api_key_env": "DEEPGRAM_API_KEY",
                "risk_level": "MEDIUM"
            }
        }

        self.incident_date = datetime(2025, 1, 21)  # Approximate date keys were exposed
        self.check_period_days = 60  # Check last 60 days for suspicious activity

    def print_security_banner(self):
        """Print security incident banner"""
        print("=" * 80)
        print("🚨 SECURITY INCIDENT BILLING VERIFICATION")
        print("=" * 80)
        print(f"Incident Date: {self.incident_date.strftime('%Y-%m-%d')}")
        print(f"Check Period: Last {self.check_period_days} days")
        print("Risk Assessment: API keys were exposed in git history")
        print("=" * 80)
        print()

    def check_env_keys_status(self):
        """Check current status of API keys in environment"""
        print("📋 CURRENT API KEY STATUS CHECK")
        print("-" * 50)

        for _provider_id, config in self.providers.items():
            env_var = config["api_key_env"]
            current_key = os.environ.get(env_var, "NOT_SET")

            if current_key == "NOT_SET":
                status = "❌ NOT SET"
            elif "PLACEHOLDER" in current_key:
                status = "✅ PLACEHOLDER (ROTATED)"
            elif current_key.startswith(("sk-", "hf_", "your-")):
                status = "⚠️ LOOKS LIKE REAL KEY - VERIFY ROTATION"
            else:
                status = "❓ UNKNOWN FORMAT"

            print(f"{config['name']:20} ({env_var}): {status}")

        print()

    def generate_manual_checklist(self):
        """Generate manual verification checklist"""
        print("📝 MANUAL BILLING VERIFICATION CHECKLIST")
        print("-" * 50)
        print()

        for _provider_id, config in self.providers.items():
            print(f"🔍 {config['name']} ({config['risk_level']} RISK)")
            print(f"   Dashboard: {config['dashboard_url']}")
            print(f"   Billing:   {config['billing_url']}")
            print()
            print("   Manual Steps:")
            print("   1. Log into the dashboard using your account credentials")
            print("   2. Navigate to Usage/Billing section")
            print(f"   3. Export usage data for {self.incident_date.strftime('%Y-%m-%d')} to present")
            print("   4. Look for:")
            print("      - Unusual spikes in API calls")
            print("      - Requests from unknown IP addresses")
            print("      - Usage patterns inconsistent with your application")
            print("      - Any charges you don't recognize")
            print("   5. Save screenshots and CSV exports to documentation/billing_evidence/")
            print()

    def create_evidence_directory(self):
        """Create directory structure for billing evidence"""
        evidence_dir = Path("documentation/billing_evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each provider
        for provider_id, config in self.providers.items():
            provider_dir = evidence_dir / provider_id
            provider_dir.mkdir(exist_ok=True)

            # Create README for each provider
            readme_path = provider_dir / "README.md"
            readme_content = f"""# {config['name']} Billing Evidence

## Incident Details
- **Date**: {self.incident_date.strftime('%Y-%m-%d')}
- **Issue**: API key exposed in git history
- **Risk Level**: {config['risk_level']}

## Required Evidence
1. Usage dashboard screenshots (last 60 days)
2. Billing overview screenshots
3. CSV export of API usage data
4. Any suspicious activity reports

## Dashboard URLs
- Usage: {config['dashboard_url']}
- Billing: {config['billing_url']}

## Files to Upload
- `usage_screenshot_YYYY-MM-DD.png`
- `billing_screenshot_YYYY-MM-DD.png`
- `usage_export_YYYY-MM-DD.csv`
- `suspicious_activity.txt` (if any found)
"""

            with open(readme_path, 'w') as f:
                f.write(readme_content)

        logger.info(f"✅ Evidence directory structure created at: {evidence_dir}")

    def generate_verification_report_template(self):
        """Generate template for verification report"""
        report_path = Path("documentation/billing_verification_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_content = f"""# API Key Exposure Billing Verification Report

**Report Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Incident Date**: {self.incident_date.strftime('%Y-%m-%d')}
**Verification Period**: {self.check_period_days} days

## Executive Summary
<!-- Complete after verification -->
- [ ] No unauthorized charges detected
- [ ] Suspicious activity found (detail in findings section)
- [ ] Unable to verify (explain in notes section)

## Provider Verification Status

"""

        for provider_id, config in self.providers.items():
            report_content += f"""### {config['name']} ({config['risk_level']} Risk)

**Dashboard Checked**: [ ] Yes [ ] No
**Billing Reviewed**: [ ] Yes [ ] No
**Evidence Collected**: [ ] Yes [ ] No

**Usage Summary**:
- Total API calls in period: <!-- Fill from dashboard -->
- Total charges in period: $<!-- Fill from billing -->
- Unusual activity detected: [ ] Yes [ ] No

**Notes**:
<!-- Add any observations, unusual patterns, or concerns -->

**Evidence Files**:
- Usage screenshot: `billing_evidence/{provider_id}/usage_screenshot_YYYY-MM-DD.png`
- Billing screenshot: `billing_evidence/{provider_id}/billing_screenshot_YYYY-MM-DD.png`
- Usage export: `billing_evidence/{provider_id}/usage_export_YYYY-MM-DD.csv`

---

"""

        report_content += """## Findings and Recommendations

### Unauthorized Usage Detected
<!-- Complete if suspicious activity found -->
- **Provider**:
- **Time Period**:
- **Details**:
- **Estimated Cost Impact**: $
- **Immediate Actions Taken**:

### Overall Assessment
<!-- Complete after all providers checked -->
- **Total Financial Impact**: $
- **Security Risk Assessment**:
- **Additional Actions Required**:

### Next Steps
- [ ] Report suspicious activity to providers
- [ ] Review and strengthen key rotation procedures
- [ ] Implement additional monitoring/alerts
- [ ] Document lessons learned

## Verification Completed By
**Name**:
**Date**:
**Signature**:

---
*This report documents the billing verification process following the API key exposure incident.*
"""

        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"✅ Verification report template created at: {report_path}")

    def run_verification_process(self):
        """Run complete billing verification process"""
        print()
        self.print_security_banner()

        # Check current environment status
        self.check_env_keys_status()

        # Create evidence collection structure
        self.create_evidence_directory()

        # Generate report template
        self.generate_verification_report_template()

        # Provide manual verification steps
        self.generate_manual_checklist()

        print("🎯 NEXT STEPS")
        print("-" * 50)
        print("1. Complete manual verification for each provider")
        print("2. Collect screenshots and export data")
        print("3. Complete the verification report template")
        print("4. Archive all evidence in billing_evidence/ directory")
        print("5. If suspicious activity found, immediately:")
        print("   - Contact provider support")
        print("   - Document all details")
        print("   - Consider additional security measures")
        print()
        print("📁 Evidence collection directory: documentation/billing_evidence/")
        print("📋 Report template: documentation/billing_verification_report.md")
        print()

if __name__ == "__main__":
    checker = BillingChecker()
    checker.run_verification_process()
