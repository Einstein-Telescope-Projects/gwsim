"""Template validation utilities for gwsim CLI."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("gwsim")


class TemplateValidator:
    """Validate template strings for simulators."""

    @staticmethod
    def validate_template(template: str, simulator_name: str) -> tuple[bool, list[str]]:
        """Validate template and return (is_valid, errors)."""
        errors = []

        try:
            # Extract all placeholder fields from template
            # template_fields = TemplateValidator._extract_template_fields(template)

            # Try to format with dummy data to catch syntax errors
            dummy_state = TemplateValidator._create_dummy_state()
            template.format(**dummy_state)

            logger.debug("Template validation passed for %s: %s", simulator_name, template)

        except KeyError as e:
            errors.append(f"Missing template field: {e}")
        except ValueError as e:
            errors.append(f"Template formatting error: {e}")
        except (AttributeError, TypeError) as e:
            errors.append(f"Template validation error: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def extract_template_fields(template: str) -> set[str]:
        """Extract field names from template string."""
        # Find all {field_name} patterns, excluding format specs
        fields = re.findall(r"\{([^}:]+)", template)
        return set(fields)

    @staticmethod
    def _create_dummy_state() -> dict:
        """Create dummy state data for validation."""
        return {
            "counter": 1,
            "start_time": 1696291200,
            "duration": 4096,
            "detector": "H1",
            "batch_id": "test",
            "sample_rate": 4096,
            "end_time": 1696295296,
        }
