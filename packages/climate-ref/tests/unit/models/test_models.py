"""Tests for model repr methods."""

from climate_ref.models import Diagnostic, Provider


class TestModelRepr:
    """Test __repr__ methods of models."""

    def test_provider_repr(self, db):
        """Test Provider.__repr__."""
        provider = Provider(slug="test-provider", name="Test Provider", version="1.0.0")
        db.session.add(provider)
        db.session.flush()

        result = repr(provider)
        assert "Provider" in result
        assert "test-provider" in result
        assert "1.0.0" in result

    def test_diagnostic_repr(self, db):
        """Test Diagnostic.__repr__."""
        provider = Provider(slug="test-provider", name="Test Provider", version="1.0.0")
        db.session.add(provider)
        db.session.flush()

        diagnostic = Diagnostic(slug="test-diagnostic", name="Test Diagnostic", provider_id=provider.id)
        db.session.add(diagnostic)
        db.session.flush()

        result = repr(diagnostic)
        assert "Metric" in result
        assert "test-diagnostic" in result
