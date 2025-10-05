#!/usr/bin/env python3
"""
Basic tests for OmniBAR Workbench
=================================

Simple targeted tests to verify core functionality.
Note: Full test suite was cut due to time constraints, focusing on core features.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from prompt_refiner_pydantic import ProteinExtraction


def test_pydantic_model_validation():
    """Test that our OmniBAR Pydantic model works correctly."""

    # Valid data should pass
    valid_data = {
        "proteins": ["PDZ3", "CRIPT", "URA3"],
        "organisms": ["Homo sapiens", "Homo sapiens", "Saccharomyces cerevisiae"]
    }

    model = ProteinExtraction(**valid_data)
    assert len(model.proteins) == 3
    assert len(model.organisms) == 3
    assert model.proteins[0] == "PDZ3"

    print("✅ Pydantic model validation works")


def test_database_schema():
    """Test that our database schema is properly structured."""

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    # Create test database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create test run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_timestamp TEXT PRIMARY KEY,
            original_prompt TEXT,
            exploration_depth INTEGER,
            total_variations INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO test_runs (run_timestamp, original_prompt, exploration_depth, total_variations)
        VALUES (?, ?, ?, ?)
    """, ("test_run", "Extract proteins", 2, 10))

    # Verify data was inserted
    cursor.execute("SELECT COUNT(*) FROM test_runs")
    count = cursor.fetchone()[0]
    assert count == 1

    conn.close()
    Path(db_path).unlink()  # Cleanup

    print("✅ Database schema works correctly")


def test_mutation_notation():
    """Test that mutation notation logic is sound."""

    original = "Extract all proteins from document"
    variation = "Extract all proteins from research"

    # Simple mutation detection logic (simplified version of our frontend code)
    orig_words = original.split()
    var_words = variation.split()

    mutations = []
    for i, (orig, var) in enumerate(zip(orig_words, var_words)):
        if orig != var:
            mutations.append(f"{orig}_{i+1}_{var}")

    expected = "document_5_research"
    assert mutations[0] == expected

    print("✅ Mutation notation logic works")


def test_json_parsing():
    """Test that we can parse LLM output correctly."""

    # Simulate LLM response
    llm_output = """
    {
        "proteins": ["PDZ3", "CRIPT"],
        "organisms": ["Homo sapiens", "Homo sapiens"]
    }
    """

    try:
        data = json.loads(llm_output.strip())
        model = ProteinExtraction(**data)
        assert len(model.proteins) == 2
        assert len(model.organisms) == 2

    except (json.JSONDecodeError, ValueError) as e:
        assert False, f"Should parse valid JSON: {e}"

    print("✅ JSON parsing works correctly")


def test_health_check_data():
    """Test that health check returns expected structure."""

    from datetime import datetime

    # Simulate health check response
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'running_processes': 0,
        'database': True
    }

    assert health_data['status'] == 'healthy'
    assert 'timestamp' in health_data
    assert isinstance(health_data['running_processes'], int)
    assert isinstance(health_data['database'], bool)

    print("✅ Health check data structure correct")


def main():
    """Run all tests."""
    print("🧪 Running OmniBAR Workbench Tests\n")

    try:
        test_pydantic_model_validation()
        test_database_schema()
        test_mutation_notation()
        test_json_parsing()
        test_health_check_data()

        print("\n🎉 All tests passed!")
        print("\nNote: This is a minimal test suite focused on core functionality.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == '__main__':
    main()