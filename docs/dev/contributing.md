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

We recommend using `uv` to manage virtual environments for installing gwsim.

If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

```shell
uv venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

4. Set Up Pre-commit Hooks and Commitlint

We use **pre-commit** to ensure code quality and consistency, and **commitlint** to enforce commit message conventions.

#### Install pre-commit hooks

After installing Python dependencies, install the Git hooks:

```shell
pre-commit install
pre-commit install --hook-type commit-msg
```

This ensures automatic checks for code formatting, linting, and hygiene on every commit.

#### Install commitlint (via npm)

Commitlint enforces that commit messages follow the Conventional Commits standard. Install it globally or locally:

```shell
# Install locally (in the project)
npm install
```

The project includes a `commitlint.config.js` configuration file that defines the commit message rules. Once installed, commitlint will automatically validate your commit messages when pre-commit runs.

**Important:** Commit messages are validated in CI/CD pipelines, and the changelog is auto-generated from commits. See section "Commit Message Guidelines" below for details.

5. Create a New Branch

Give it a meaningful name like fix-gw-signal-generation or feature-add-noise-model.

6. Make Changes

- Write clear, concise, and well-documented code, ensuring it aligns with the goal of generating simulated GW data.
- **Follow PEP 8 style conventions strictly**—linting rules are enforced via pre-commit and in CI/CD.
- Add or update unit tests, especially for GW signal generation and noise simulation, when applicable.
- **Keep changes atomic and focused**: one type of change per commit (e.g., do not mix refactoring with feature addition).

7. Run Tests

Ensure that all tests pass before opening a merge request:

```shell
pytest
```

8. Open a Merge Request

Clearly describe the motivation and scope of your change, especially how it impacts GW data simulation. Link it to the relevant issue if applicable. Ensure all commit messages follow the guidelines in "Commit Message Guidelines" below.

## 📝 Commit Message Guidelines

**Why this matters:** Our changelog is automatically generated from commit messages using git-cliff. Commit messages must follow the Conventional Commits format and adhere to strict rules.

### Rules

1. **One type of change per commit**

   - Do not mix different types of changes (e.g., bug fixes, features, refactoring) in a single commit.
   - Example: if you refactor code AND add a feature, make two separate commits.

2. **Descriptive and meaningful messages**

   - Describe _what_ changed and _why_, not just _what_ was edited.
   - Avoid vague messages like "fix bug" or "update code"; instead use "fix: prevent signal saturation in noise simulation" or "feat: add support for multi-detector frame merging".

3. **Follow Conventional Commits format**

   - All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) standard.
   - Format: `<type>(<scope>): <subject>`
   - Allowed types:
     - build: Changes that affect the build system or external dependencies
     - ci: Changes to our CI configuration files and scripts
     - docs: Documentation only changes
     - feat: A new feature
     - fix: A bug fix
     - perf: A code change that improves performance
     - refactor: A code change that neither fixes a bug nor adds a feature
     - style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
     - test: Adding missing tests or correcting existing tests
   - Example:

     ```
     feat(signal): add BBH waveform generation for aligned-spin systems

     This commit introduces support for aligned-spin binary black hole
     waveforms using PyCBC, enabling more realistic simulations.
     ```

   - Commitlint will validate your message format automatically.

### Examples

✅ **Good commits:**

```
feat(noise): Implement colored noise with PSD shaping
fix(cli): Resolve frame file path resolution on Windows
docs: Clarify metadata JSON schema in README
test(validate): Add edge case tests for boundary conditions
refactor(simulator): Simplify noise factory registration
```

❌ **Bad commits:**

```
fixed stuff
wip: many changes
update code
more fixes (no type/scope)
```

## 💡 Tips

- Be kind and constructive in your communication.
- Keep MRs focused and atomic—smaller changes are easier to review.
- Document new features and update existing docs, especially for new GW simulation parameters or methods.
- Tag your MR with relevant labels if you can (e.g., `type::bug`, `type::enhancement`, `type::documentation`).
- If commitlint rejects your message, the error will tell you why; adjust and re-commit.

## 📜 Licensing

By contributing, you agree that your contributions will be licensed under the project’s MIT License.

---

Thanks again for being part of the gwsim community and helping advance gravitational-wave research!

---
