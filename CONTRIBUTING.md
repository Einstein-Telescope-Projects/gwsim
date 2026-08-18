# Contributing to gwmock

🎉 Thank you for your interest in contributing to gwmock! 🌌📊 Your ideas,
fixes, and improvements are welcome and appreciated as we work to enhance this
package for generating simulated gravitational-wave (GW) data.

Whether you’re fixing a typo, reporting a bug, suggesting a feature, or
submitting a merge request—this guide will help you get started.

## How to Contribute

<!-- prettier-ignore-start -->

1. Open an Issue

    - Have a question, bug report, or feature suggestion?
    [Open an issue](https://github.com/Leuven-Gravity-Institute/gwmock/issues/new) and describe your idea clearly,
    including its relevance to generating simulated GW data.
    - Check for existing issues before opening a new one.

2. Fork and Clone the Repository

    ```shell
    git clone <GIT URL of your forked repository>
    cd gwmock
    ```

3. Set Up Your Environment

    We recommend using `uv` to manage virtual environments for installing gwmock.

    If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

    - Install via pip: `pip install --upgrade pip && pip install uv`
    - Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
    - Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

    ```shell
    uv venv
    source .venv/bin/activate # on Windows: .venv\Scripts\activate
    uv sync --group dev
    ```

4. Set Up Prek Hooks

    We use **prek** to ensure code quality and consistency.
    After installing Python dependencies, install the Git hooks:

    ```shell
    uv run prek install
    ```

    This ensures automatic checks for code formatting, linting, and hygiene on every commit.

5. Create a New Branch

    Give it a meaningful name like fix-gw-signal-generation or feature-add-noise-model.

6. Make Changes

    - Write clear, concise, and well-documented code, ensuring it aligns with the goal of generating simulated GW data.
    - **Follow PEP 8 style conventions strictly**—linting rules are enforced via prek and in CI/CD.
    - Add or update unit tests, especially for GW signal generation and noise simulation, when applicable.
    - Physics changes should target the relevant subpackage repository (`gwmock-signal`, `gwmock-noise`, or `gwmock-pop`); keep gwmock changes focused on adapters, orchestration, and docs.
    - **Keep changes atomic and focused**: one type of change per commit (e.g., do not mix refactoring with feature addition).

7. Run Tests

    Ensure that all tests pass before opening a merge request:

    ```shell
    uv run pytest
    ```

    Mutation testing checks that the tests actually fail when the code is wrong. It mutates a copy of
    the source under `mutants/` and is much slower than the unit suite, so run it on the code you
    touched rather than the whole package:

    ```shell
    uv run mutmut run                     # whole package
    uv run mutmut results                 # list surviving mutants
    uv run mutmut show <mutant-name>      # diff of one survivor
    uv run mutmut browse                  # interactive results browser

    # `run` also takes mutant-name globs, so a single module can be targeted; names are the
    # dotted module path plus the mutated function, as printed by `mutmut results`.
    uv run mutmut run 'src.gwmock.cli.main.*'
    ```

    A surviving mutant is a change to the source that no test noticed. Either add the test that
    catches it, or convince yourself the mutant is equivalent to the original.

8. Open a Pull Request

    Clearly describe the motivation and scope of your change, especially how it impacts GW data simulation.
    Link it to the relevant issue if applicable. Ensure the title of the pull request follow the guidelines in
    "Commit Message Guidelines" below.

<!-- prettier-ignore-end -->

## Commit Message Guidelines

**Why this matters:** Our changelog is automatically generated from commit
messages using git-cliff. Commit messages must follow the Conventional Commits
format and adhere to strict rules. Since we use squash commits, the pull request
title will automatically become the commit message on the main branch.

### Rules

<!-- prettier-ignore-start -->

1. **One type of change per pull request**

    - Do not mix different types of changes (e.g., bug fixes, features, refactoring) in a single pull request.
    - Example: if you refactor code AND add a feature, make two separate pull requests.

2. **Descriptive and meaningful messages**

    - Describe _what_ changed and _why_, not just _what_ was edited.
    - Avoid vague messages like "fix bug" or "update code"; instead use "fix: prevent signal saturation in noise
    simulation" or "feat: add support for multi-detector frame merging".

3. **Follow Conventional Commits format**

    - All pull request titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) standard.
    - Format: `<type>(<scope>): <subject>`
    - Allowed types:
        - build: Changes that affect the build system or external dependencies
        - ci: Changes to our CI configuration files and scripts
        - docs: Documentation only changes
        - feat: A new feature
        - fix: A bug fix
        - perf: A code change that improves performance
        - refactor: A code change that neither fixes a bug nor adds a feature
        - style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)
        - test: Adding missing tests or correcting existing tests
    - Example:

        ```text
        feat(signal): add BBH waveform generation for aligned-spin systems

        This commit introduces support for aligned-spin binary black hole
        waveforms using PyCBC, enabling more realistic simulations.
        ```

    - Semantic pull request will validate your message format automatically.

<!-- prettier-ignore-end -->

### Examples

✅ **Good pull request titles:**

```text
feat(noise): Implement colored noise with PSD shaping
fix(cli): Resolve frame file path resolution on Windows
docs(metadata): Clarify metadata JSON schema in README
test(validate): Add edge case tests for boundary conditions
refactor(simulator): Simplify noise factory registration
```

❌ **Bad pull request titles:**

```text
fixed stuff
wip: many changes
update code
more fixes (no type/scope)
```

## 💡 Tips

- Be kind and constructive in your communication.
- Keep PRs focused and atomic—smaller changes are easier to review.
- Document new features and update existing docs, especially for new GW
  simulation parameters or methods.
- Tag your PR with relevant labels if you can (e.g., `bug`, `enhancement`,
  `documentation`).

## Licensing

By contributing, you agree that your contributions will be licensed under the
same terms as the project: **GPL-3.0-or-later** (see the repository `LICENSE`
file).

---

Thanks again for being part of the gwmock community and helping advance
gravitational-wave research!

---
