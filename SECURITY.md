# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in BrainPy, please report it to us responsibly:

### Preferred Method: Security Advisories

Use GitHub's Security Advisory feature:
1. Navigate to the [Security tab](https://github.com/brainpy/BrainPy/security/advisories)
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

### Alternative Method: Email

Send an email to: **chao.brain@qq.com**

Please include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: We will acknowledge your report within **3 business days**
- **Initial Response**: You'll receive a detailed response within **7 days** indicating the next steps
- **Updates**: We will keep you informed of our progress throughout the resolution process
- **Disclosure**: We aim to address critical vulnerabilities within **30 days**

These timelines may extend during holiday periods or when triage volunteers are unavailable.

### What to Expect

After you submit a report:
1. We will confirm the vulnerability and determine its impact
2. We will develop and test a fix
3. We will release a security advisory and patched versions
4. We will publicly acknowledge your responsible disclosure (unless you prefer to remain anonymous)

## Security Update Policy

- Security updates will be released as patch versions (e.g., 2.4.1)
- Critical vulnerabilities may warrant immediate releases
- All security updates will be announced via:
  - GitHub Security Advisories
  - Release notes
  - Project documentation

## Reporting Bugs in Third-Party Dependencies

Security bugs in third-party modules (jax, numpy, etc.) should be reported directly to their respective maintainers. We will update dependencies promptly when security patches become available.

## Bug Bounty Program

We currently do not offer a paid bug bounty program, but we deeply appreciate and acknowledge all security researchers who help keep BrainPy secure.

## Security Best Practices for Users

When using BrainPy:
- Always use the latest stable version
- Keep dependencies updated
- Review and understand code before executing untrusted models
- Be cautious when loading pre-trained models from untrusted sources
- Follow security best practices when deploying BrainPy in production environments

Thank you for helping keep BrainPy and its users safe!
