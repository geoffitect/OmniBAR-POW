#!/usr/bin/env python3
"""
Test suite for Prompt Refiner functionality
Tests the core components without requiring API calls
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.prompt_refiner_suite import (
    ThesaurusReplacer,
    PromptVariationGenerator,
    PromptRefinerDatabase
)
import tempfile
import os


def test_thesaurus_replacer():
    """Test synonym generation"""
    print("\n🧪 Testing ThesaurusReplacer...")

    replacer = ThesaurusReplacer()

    # Test basic synonym retrieval
    synonyms = replacer.get_synonyms("extract")
    print(f"  ✓ Found {len(synonyms)} synonyms for 'extract': {synonyms}")
    assert len(synonyms) > 0, "Should find at least one synonym"

    # Test fallback for common word
    synonyms = replacer.get_synonyms("document")
    print(f"  ✓ Found {len(synonyms)} synonyms for 'document': {synonyms}")
    assert len(synonyms) > 0, "Should find fallback synonyms"

    # Test word with no synonyms
    synonyms = replacer.get_synonyms("xyzabc")
    print(f"  ✓ Found {len(synonyms)} synonyms for 'xyzabc': {synonyms}")
    assert len(synonyms) == 0, "Should return empty list for nonsense words"

    print("  ✅ ThesaurusReplacer tests passed")


def test_prompt_variation_generator():
    """Test prompt variation generation"""
    print("\n🧪 Testing PromptVariationGenerator...")

    # Test with depth 1
    generator = PromptVariationGenerator(exploration_depth=1)
    prompt = "Extract all data from document"
    variations = generator.generate_variations(prompt)

    print(f"  ✓ Generated {len(variations)} variations for depth 1")
    assert len(variations) > 1, "Should generate at least original + variations"

    # Check original is included
    original = next((v for v in variations if v['variation_id'] == 'original'), None)
    assert original is not None, "Should include original prompt"
    assert original['prompt'] == prompt, "Original should match input"
    print(f"  ✓ Original prompt preserved: '{original['prompt']}'")

    # Check variations have proper structure
    for var in variations[:3]:
        assert 'prompt' in var, "Each variation should have prompt"
        assert 'changes' in var, "Each variation should have changes list"
        assert 'variation_id' in var, "Each variation should have ID"
        assert 'depth' in var, "Each variation should have depth"
        print(f"  ✓ Variation {var['variation_id']}: {var['prompt'][:50]}...")

    # Test with depth 2
    generator = PromptVariationGenerator(exploration_depth=2)
    variations_d2 = generator.generate_variations(prompt)
    print(f"  ✓ Generated {len(variations_d2)} variations for depth 2")
    assert len(variations_d2) > len(variations), "Depth 2 should generate more variations"

    # Check depth distribution
    depth_counts = {}
    for var in variations_d2:
        d = var['depth']
        depth_counts[d] = depth_counts.get(d, 0) + 1

    print(f"  ✓ Depth distribution: {depth_counts}")

    print("  ✅ PromptVariationGenerator tests passed")


def test_database():
    """Test database storage and retrieval"""
    print("\n🧪 Testing PromptRefinerDatabase...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db = f.name

    try:
        db = PromptRefinerDatabase(db_path=temp_db)

        # Test storing run metadata
        run_timestamp = "20250102_120000"
        db.store_test_run(
            run_timestamp=run_timestamp,
            original_prompt="Test prompt",
            exploration_depth=2,
            total_variations=10,
            test_document="Test document"
        )
        print("  ✓ Stored test run metadata")

        # Test storing variation results
        db.store_variation_result(
            run_timestamp=run_timestamp,
            original_prompt="Test prompt",
            variation_id="original",
            variation_prompt="Test prompt",
            depth=0,
            changes=[],
            scores={'clarity': 0.85, 'effectiveness': 0.82, 'precision': 0.88, 'overall': 0.85},
            test_document="Test document"
        )

        db.store_variation_result(
            run_timestamp=run_timestamp,
            original_prompt="Test prompt",
            variation_id="d1_0test",
            variation_prompt="Test variation prompt",
            depth=1,
            changes=[{'position': 0, 'from': 'Test', 'to': 'Sample'}],
            scores={'clarity': 0.90, 'effectiveness': 0.88, 'precision': 0.92, 'overall': 0.90},
            test_document="Test document"
        )
        print("  ✓ Stored 2 variation results")

        # Test retrieval
        best_variations = db.get_best_variations(run_timestamp, top_n=2)
        assert len(best_variations) == 2, "Should retrieve 2 variations"
        print(f"  ✓ Retrieved {len(best_variations)} best variations")

        # Check ordering (best first)
        assert best_variations[0]['scores']['overall'] >= best_variations[1]['scores']['overall'], \
            "Should be ordered by overall score"
        print(f"  ✓ Correct ordering: {best_variations[0]['scores']['overall']:.2f} >= "
              f"{best_variations[1]['scores']['overall']:.2f}")

        # Verify data integrity
        best = best_variations[0]
        assert best['variation_id'] == 'd1_0test', "Best should be the higher-scoring variation"
        assert best['depth'] == 1, "Depth should be preserved"
        assert len(best['changes']) == 1, "Changes should be preserved"
        print(f"  ✓ Data integrity verified for variation: {best['variation_id']}")

        db.close()
        print("  ✅ Database tests passed")

    finally:
        # Cleanup
        if os.path.exists(temp_db):
            os.unlink(temp_db)


def test_integration():
    """Integration test: Generate variations and simulate storage"""
    print("\n🧪 Testing Integration...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db = f.name

    try:
        # Generate variations
        generator = PromptVariationGenerator(exploration_depth=1)
        prompt = "Extract key findings from document"
        variations = generator.generate_variations(prompt)
        print(f"  ✓ Generated {len(variations)} variations")

        # Store in database
        db = PromptRefinerDatabase(db_path=temp_db)
        run_timestamp = "20250102_120000"

        db.store_test_run(
            run_timestamp=run_timestamp,
            original_prompt=prompt,
            exploration_depth=1,
            total_variations=len(variations),
            test_document="Integration test document"
        )

        # Simulate storing results for first 3 variations
        for i, var in enumerate(variations[:3]):
            # Simulate scores (original gets 0.80, variations get slightly higher)
            base_score = 0.80 if var['variation_id'] == 'original' else 0.80 + i * 0.03
            scores = {
                'clarity': base_score + 0.02,
                'effectiveness': base_score,
                'precision': base_score + 0.03,
                'overall': base_score + 0.02
            }

            db.store_variation_result(
                run_timestamp=run_timestamp,
                original_prompt=prompt,
                variation_id=var['variation_id'],
                variation_prompt=var['prompt'],
                depth=var['depth'],
                changes=var['changes'],
                scores=scores,
                test_document="Integration test document"
            )

        print(f"  ✓ Stored {3} variation results")

        # Retrieve and verify
        best_variations = db.get_best_variations(run_timestamp, top_n=3)
        assert len(best_variations) == 3, "Should retrieve 3 variations"

        print("  ✓ Top 3 variations:")
        for i, var in enumerate(best_variations):
            print(f"    {i+1}. {var['variation_id']}: Overall={var['scores']['overall']:.2f}")

        db.close()
        print("  ✅ Integration tests passed")

    finally:
        # Cleanup
        if os.path.exists(temp_db):
            os.unlink(temp_db)


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 PROMPT REFINER TEST SUITE")
    print("=" * 70)

    try:
        test_thesaurus_replacer()
        test_prompt_variation_generator()
        test_database()
        test_integration()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
