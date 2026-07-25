# Contributing to MIRIX

We’re thrilled to have you join us in contributing to MIRIX! As passionate champions of open-source collaboration, we’re committed to building not only a powerful tool but also a vibrant community where developers of all skill levels can make impactful contributions. Before you get started, we encourage you to review the [MIRIX Code Standards](https://github.com/Mirix-AI/MIRIX/blob/main/CODE_STANDARDS.md). This will help ensure consistency and set the stage for success for everyone involved.

Our contribution pathways are designed to make it easy for you to get involved and start making an impact immediately:

# Four Ways to Get Involved

### Help With Existing Issues

Our developers frequently tag issues with "help wanted" and "good first issue," highlighting pre-vetted tasks with a clear scope. Plus, there's always someone available to assist you if you encounter any challenges along the way.

### Create Your Own Tickets

Noticed something that needs fixing? Got an idea for an improvement? You don’t need permission to identify problems—the people closest to the pain are often the best equipped to envision solutions.

For **feature requests**, share the story of what you’re trying to achieve. What are you working on? What’s standing in your way? What would make your life easier? Submit your ideas through our [GitHub issue tracker](https://github.com/Mirix-AI/MIRIX/issues) using the "Feature Request" label.

For **bug reports**, it’s important to provide enough details for us to reproduce the issue. Use our [GitHub issue tracker](https://github.com/Mirix-AI/MIRIX/issues) and include the following:

  - A clear and concise title summarizing the problem
  - What you were attempting to do when the issue occured
  - What you expected to happen versus what actually happened
  - A code sample, test case, or detailed steps to help us replicate the issue

### Share Your Use Cases

Not all contributions are about writing code—sometimes the most impactful ones come from sharing how you're using our project. If you're leveraging MIRIX in an interesting way, consider adding it to the [samples](https://github.com/Mirix-AI/MIRIX/tree/main/samples) folder. This not only helps others explore new possibilities but also counts as a meaningful contribution. We often highlight compelling examples in our blog posts and videos, so your work could be showcased to the broader community!

### Help Others in Discord

Join our [Discord](https://discord.gg/5HWyxJrh) community and contribute by lending a hand at the helpdesk. Answering questions and helping troubleshoot issues is an invaluable way to support the community. The knowledge you share today could save someone hours of frustration tomorrow.

## What happens next?

### Notes for Large Changes
> Please keep changes as concise as possible. For major architectural changes (>500 LOC), please create a GitHub issue (RFC) to discuss the technical design and justification. Otherwise, we may tag it as "rfc-required" and potentially not review the PR.

Once you've identified an issue tagged with "good first issue" or "help wanted," or have an example you'd like to share, here's how to turn it into a contribution:

1. Share your proposed approach in the issue discussion or on [Discord](https://discord.gg/5HWyxJrh) before diving into the code. This ensures your solution aligns with MIRIX's architecture from the outset and helps avoid potential rework.

2. Fork the repository, create a branch for your changes, and submit a pull request (PR). Detailed technical instructions are provided below — be open to feedback during the review process.

## Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/Mirix-AI/MIRIX
   cd MIRIX
   ```
3. Set up your development environment:

   Follow the "Development Installation" section in this [Get Started](https://docs.mirix.io/getting-started/installation/) document. 

   If you prefer to poetry instead of pip, follow these additional steps. ([uv](https://docs.astral.sh/uv/getting-started/installation/) follows similar steps)
   - Ensure you have Python 3.11+ installed.
   - Install poetry: https://python-poetry.org/docs/
   - [Recommended] Configure poetry to create virtual env under your current project
     ```
     poetry config virtualenvs.in-project true
     ```
   - Install project dependencies:
     ```
     make install
     ```

## Making Changes

1. Create a new branch for your changes:
   ```
   git checkout -b your-branch-name
   ```
2. Make your changes in the codebase.
3. Write or update tests as necessary.
4. Run the tests to ensure they pass:
   ```
   make test
   ```
5. Format your code:
   ```
   make format
   ```
6. Run linting checks:
   ```
   make lint
   ```

## Submitting Changes

1. Commit your changes:
   ```
   git commit -m "Your commit message"
   ```
2. Push to your fork:
   ```
   git push origin your-branch-name
   ```
3. Submit a pull request through the GitHub website to https://github.com/Mirix-AI/MIRIX.

## Pull Request Guidelines

- Provide a clear title and description of your changes.
- Include any relevant issue numbers in the PR description.
- Ensure all tests pass and there are no linting errors.
- Update documentation if you're changing functionality.

## Code Style and Quality

We use several tools to maintain code quality:

- Ruff for linting and formatting
- Pyright for static type checking
- Pytest for testing

Before submitting a pull request, please run:

```
make check
```

This command will format your code, run linting checks, and execute tests.

## Third-Party Integrations

When contributing integrations for third-party services (LLM providers, embedding services, databases, etc.), please follow these patterns:

### Optional Dependencies

All third-party integrations must be optional dependencies to keep the core library lightweight. Follow this pattern:

1. **Add to `pyproject.toml`**: Define your dependency as an optional extra AND include it in the dev extra:
   ```toml
   [project.optional-dependencies]
   your-service = ["your-package>=1.0.0"]
   dev = [
       # ... existing dev dependencies
       "your-package>=1.0.0",  # Include all optional extras here
       # ... other dependencies
   ]
   ```

2. **Use TYPE_CHECKING pattern**: In your integration module, import dependencies conditionally:
   ```python
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       import your_package
       from your_package import SomeType
   else:
       try:
           import your_package
           from your_package import SomeType
       except ImportError:
           raise ImportError(
               'your-package is required for YourServiceClient. '
               'Install it with: pip install MIRIX[your-service]'
           ) from None
   ```

3. **Benefits of this pattern**:
   - Fast startup times (no import overhead during type checking)
   - Clear error messages with installation instructions
   - Proper type hints for development
   - Consistent user experience

4. **Do NOT**:
   - Add optional imports to `__init__.py` files
   - Use direct imports without error handling
   - Include optional dependencies in the main `dependencies` list

### Integration Structure

- Place LLM clients in `mirix/llm_client/`
- Place embedding clients in `mirix/embedder/`
- Place database drivers in `mirix/driver/`
- Place database script python files in `database/`
- Place samples in `samples/`
- Follow existing naming conventions (e.g., `your_service_client.py`)

### Testing

- Add comprehensive tests in the appropriate `tests/` subdirectory
- Mark integration tests with `_int` suffix if they require external services
- Include both unit tests and integration tests where applicable

# Questions?

Stuck on a contribution or have a half-formed idea? Come say hello in our [Discord server](https://discord.gg/5HWyxJrh). Whether you're ready to contribute or just want to learn more, we're happy to have you! It's faster than GitHub issues and you'll find both maintainers and fellow contributors ready to help.

Thank you for your contribution to MIRIX!
