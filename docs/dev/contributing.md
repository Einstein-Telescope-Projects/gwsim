# Contributing to gwsim

🎉 Thank you for your interest in contributing to gwsim! 🌌📊 Your ideas, fixes, and improvements are welcome and appreciated as we work to enhance this package for generating simulated gravitational-wave (GW) data.

Whether you’re fixing a typo, reporting a bug, suggesting a feature, or submitting a merge request—this guide will help you get started.

## 📌 How to Contribute

1. Open an Issue

- Have a question, bug report, or feature suggestion? [Open an issue](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues/new) and describe your idea clearly, including its relevance to generating simulated GW data.
- Check for existing issues before opening a new one.

2. Fork and Clone the Repository

```shell
git clone <GIT URL of your forked repository>
cd gwsim
```

3. Set Up Your Environment

We recommend using a virtual environment:

```shell
python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

4. Set Up Pre-commit Hooks

We use pre-commit to ensure code quality and consistency. After installing dependencies, run:

```shell
pre-commit install
```

This ensures checks like code formatting, linting, and basic hygiene run automatically when you commit.

5. Create a New Branch

Give it a meaningful name like fix-gw-signal-generation or feature-add-noise-model.

6. Make Changes

- Write clear, concise, and well-documented code, ensuring it aligns with the goal of generating simulated GW data.
- Follow PEP 8 style conventions.
- Add or update unit tests, especially for GW signal generation and noise simulation, when applicable.

7. Run Tests

Ensure that all tests pass before opening a merge request:

```shell
pytest
```

8. Open a Merge Request

Clearly describe the motivation and scope of your change, especially how it impacts GW data simulation. Link it to the relevant issue if applicable.The pull request titles should match the Conventional Commits spec.

## 💡 Tips

- Be kind and constructive in your communication.
- Keep MRs focused and atomic—smaller changes are easier to review.
- Document new features and update existing docs, especially for new GW simulation parameters or methods.
- Tag your MR with relevant labels if you can (e.g., `type::bug`, `type::enhancement`, `type::documentation`).

## 📜 Licensing

By contributing, you agree that your contributions will be licensed under the project’s MIT License.

---

Thanks again for being part of the gwsim community and helping advance gravitational-wave research!

---
