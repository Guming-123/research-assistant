#!/bin/bash
# Test runner for Multi-Agent Literature Review System

set -e

echo "====================================="
echo "Multi-Agent Literature Review System"
echo "Test Runner v1.2"
echo "====================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install -q pytest pytest-asyncio pytest-cov
fi

# Run tests with coverage
echo "Running tests with coverage report..."
echo ""

pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --tb=short \
    -W ignore

echo ""
echo "====================================="
echo "Test Results Summary"
echo "====================================="
echo ""
echo "Coverage report generated in: htmlcov/index.html"
echo ""
echo "To view detailed coverage:"
echo "  1. Open htmlcov/index.html in a browser"
echo "  2. Or run: pytest tests/ --cov=src --cov-report=html"
echo ""

# Exit with appropriate code
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Please check the output above."
    exit 1
fi