#!/usr/bin/env python3
"""
Prompt Refiner Backend - Pydantic Validation
===================================================

OBJECTIVE EVALUATION APPROACH:
Instead of using LLM-as-judge scoring, this uses deterministic
Pydantic validation to measure prompt performance. Much faster, cheaper, and more
reliable than LLM judges.

This example demonstrates:
- Automated prompt variation generation with exploration depth control
- Pydantic-based extraction validation (field presence, type correctness, format, pairs)
- SQL Database storage for live visualization and comparison
- Cost-free testing with Ollama (llama3.2:1b recommended)

Evaluation Metrics (100% algorithmic, 0% subjective):
1. Field Completeness: Did we extract all required fields? (0.0-1.0)
2. Type Correctness: Are the field types correct? (0.0-1.0)
3. Format Correctness: Do the outputs match the request schema (Pydantic/JSON)?
5. Paired Values Equal Length: Do requested paired lists have a consistent number of entries?
5. Overall Score: Composite metric across all dimensions

CLI Usage:
    # Cost-free with Ollama (recommended)
    python prompt_refiner_pydantic.py --ollama --ollama-model llama3.2:1b --depth 1

    # With OpenAI GPT-4
    # Add API key to ../.env (i.e. OPENAI_API_KEY="sk-...")
    python prompt_refiner_pydantic.py --depth 1

WebUI Usage:
    pip install -r requirements-workbench.txt
    python examples/start_workbench.py
"""

import asyncio
from pathlib import Path
import json
import sqlite3
from typing import List, Dict, Any
from itertools import combinations, product
from datetime import datetime
import argparse
import re

from pydantic import BaseModel, Field, ValidationError
from omnibar import OmniBarmarker, Benchmark
from omnibar.objectives import CombinedBenchmarkObjective
from omnibar.objectives.base import BaseBenchmarkObjective
from omnibar.core.types import FloatEvalResult
from langchain_openai import ChatOpenAI

import pymupdf

# Load environment variables
def load_environment_variables():
    """Load environment variables from various possible locations."""
    try:
        from dotenv import load_dotenv
        import os

        custom_env = os.getenv("OMNIBAR_ENV_PATH")
        if custom_env:
            custom_path = Path(custom_env)
            if custom_path.exists():
                load_dotenv(custom_path)
                print(f"✅ Loaded environment variables from {custom_path}")
                return

        potential_env_paths = [
            Path(".env"),
            Path("../.env"),
            Path("../../.env"),
        ]

        for env_path in potential_env_paths:
            if env_path.exists():
                load_dotenv(env_path.resolve())
                print(f"✅ Loaded environment variables from {env_path.resolve()}")
                return

        print("⚠️  No .env file found, using system environment variables")

    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")

load_environment_variables()


# =============================================================================
# Pydantic Model for Extraction Validation
# =============================================================================

class ProteinExtraction(BaseModel):
    """Expected structure for protein data extraction from research papers."""
    proteins: List[str] = Field(description="List of all protein names mentioned in the document")
    organisms: List[str] = Field(description="Pair list of source organisms for each protein (same order as `proteins`)")


# =============================================================================
# Algorithmic Validation Objectives
# =============================================================================

class PydanticFieldCompletenessObjective(BaseBenchmarkObjective):
    """
    Evaluates field completeness: Are all required fields present?

    Score = (fields_present / total_required_fields)
    100% algorithmic, 0% subjective.
    """

    expected_model: type[BaseModel] = Field(default=ProteinExtraction)

    def __init__(self, name: str, output_key: str, expected_model: type[BaseModel]):
        super().__init__(
            name=name,
            description=f"Measures field completeness for {expected_model.__name__}",
            output_key=output_key,
            goal="All fields present",  # Not used, but required by base class
            valid_eval_result_type=FloatEvalResult,
            expected_model=expected_model
        )

    async def _eval_fn_async(self, goal: Any, filtered_output: Dict[str, Any], **kwargs) -> FloatEvalResult:
        """Evaluate field completeness algorithmically."""
        output = filtered_output.get(self.output_key, "")

        try:
            # Try to parse as JSON
            if isinstance(output, str):
                data = json.loads(output)
            else:
                data = output

            # Get required fields from Pydantic model
            required_fields = set(self.expected_model.model_fields.keys())
            present_fields = set(data.keys())

            # Calculate completeness score
            completeness = len(present_fields & required_fields) / len(required_fields)

            return FloatEvalResult(result=completeness)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Failed to parse - zero completeness
            return FloatEvalResult(result=0.0)


class PydanticTypeCorrectnessObjective(BaseBenchmarkObjective):
    """
    Evaluates type correctness: Do fields have correct types?

    Score = (correctly_typed_fields / total_fields_present)
    Uses Pydantic validation - 100% deterministic.
    """

    expected_model: type[BaseModel] = Field(default=ProteinExtraction)

    def __init__(self, name: str, output_key: str, expected_model: type[BaseModel]):
        super().__init__(
            name=name,
            description=f"Measures type correctness for {expected_model.__name__}",
            output_key=output_key,
            goal="All types correct",  # Not used, but required by base class
            valid_eval_result_type=FloatEvalResult,
            expected_model=expected_model
        )

    async def _eval_fn_async(self, goal: Any, filtered_output: Dict[str, Any], **kwargs) -> FloatEvalResult:
        """Evaluate type correctness using Pydantic validation."""
        output = filtered_output.get(self.output_key, "")

        try:
            # Try to parse as JSON
            if isinstance(output, str):
                data = json.loads(output)
            else:
                data = output

            # Try to validate with Pydantic model
            try:
                _ = self.expected_model(**data)
                # All fields validated successfully
                return FloatEvalResult(result=1.0)
            except ValidationError as e:
                # Some fields failed validation
                total_fields = len(data.keys())
                if total_fields == 0:
                    return FloatEvalResult(result=0.0)

                # Count how many fields passed vs failed
                error_fields = set()
                for error in e.errors():
                    if error['loc']:
                        error_fields.add(error['loc'][0])

                correct_fields = total_fields - len(error_fields)
                return FloatEvalResult(result=correct_fields / total_fields)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Failed to parse - zero type correctness
            return FloatEvalResult(result=0.0)


class PydanticFormatValidityObjective(BaseBenchmarkObjective):
    """
    Evaluates format validity: Do list fields contain appropriate entries?

    Checks:
    - Lists have > 0 items
    - String fields are non-empty
    - List items are non-empty strings

    Score = (valid_fields / total_fields)
    100% algorithmic rule-based validation.
    """

    expected_model: type[BaseModel] = Field(default=ProteinExtraction)

    def __init__(self, name: str, output_key: str, expected_model: type[BaseModel]):
        super().__init__(
            name=name,
            description=f"Measures format validity for {expected_model.__name__}",
            output_key=output_key,
            goal="All formats valid",  # Not used, but required by base class
            valid_eval_result_type=FloatEvalResult,
            expected_model=expected_model
        )

    async def _eval_fn_async(self, goal: Any, filtered_output: Dict[str, Any], **kwargs) -> FloatEvalResult:
        """Evaluate format validity algorithmically."""
        output = filtered_output.get(self.output_key, "")

        try:
            # Try to parse as JSON
            if isinstance(output, str):
                data = json.loads(output)
            else:
                data = output

            total_fields = len(data.keys())
            if total_fields == 0:
                return FloatEvalResult(result=0.0)

            valid_count = 0

            for field_name, field_value in data.items():
                # Check format based on type
                if isinstance(field_value, list):
                    # Lists should have at least 1 item with non-empty strings
                    if len(field_value) > 0 and all(isinstance(item, str) and len(item.strip()) > 0 for item in field_value):
                        valid_count += 1
                elif isinstance(field_value, str):
                    # Strings should be non-empty and reasonable length
                    if len(field_value.strip()) > 0 and len(field_value) < 500:
                        valid_count += 1
                elif isinstance(field_value, int):
                    # Integers should be positive and reasonable
                    if field_value > 0 and field_value < 1000:
                        valid_count += 1
                else:
                    # Other types - accept as valid
                    valid_count += 1

            return FloatEvalResult(result=valid_count / total_fields)

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Failed to parse - zero format validity
            return FloatEvalResult(result=0.0)


class ProteinOrganismPairingObjective(BaseBenchmarkObjective):
    """
    Validates that proteins and organisms lists have equal length.

    This ensures each protein has a corresponding organism.
    Score = 1.0 if lengths match, 0.0 otherwise.
    """

    def __init__(self, name: str, output_key: str):
        super().__init__(
            name=name,
            description="Validates proteins and organisms have equal length",
            output_key=output_key,
            goal="Equal length lists",
            valid_eval_result_type=FloatEvalResult
        )

    async def _eval_fn_async(self, goal: Any, filtered_output: Dict[str, Any], **kwargs) -> FloatEvalResult:
        """Check if proteins and organisms lists have equal length."""
        output = filtered_output.get(self.output_key, "")

        try:
            # Try to parse as JSON
            if isinstance(output, str):
                data = json.loads(output)
            else:
                data = output

            # Get both lists
            proteins = data.get("proteins", [])
            organisms = data.get("organisms", [])

            # Check if both are lists
            if not isinstance(proteins, list) or not isinstance(organisms, list):
                return FloatEvalResult(result=0.0)

            # Check if lengths match
            if len(proteins) == len(organisms) and len(proteins) > 0:
                return FloatEvalResult(result=1.0)
            else:
                return FloatEvalResult(result=0.0)

        except (json.JSONDecodeError, AttributeError, TypeError, KeyError):
            return FloatEvalResult(result=0.0)


# =============================================================================
# Document Extraction Agent with Pydantic Output
# =============================================================================

# Global LLM client cache to avoid reloading models
_ollama_llm_cache = {}
_openai_llm_cache = {}

def get_cached_llm(use_ollama: bool, model: str):
    """Get or create a cached LLM client to avoid model reloading."""
    if use_ollama:
        cache_key = f"ollama_{model}"
        if cache_key not in _ollama_llm_cache:
            _ollama_llm_cache[cache_key] = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model=model,
                temperature=0
            )
        return _ollama_llm_cache[cache_key]
    else:
        cache_key = f"openai_{model}"
        if cache_key not in _openai_llm_cache:
            _openai_llm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=0
            )
        return _openai_llm_cache[cache_key]


def count_tokens(text: str) -> int:
    """Simple token counting approximation: split by whitespace and punctuation"""
    import re
    # Split on whitespace and punctuation, filter empty strings
    tokens = [t for t in re.split(r'[\s\W]+', text) if t]
    return len(tokens)


class DocumentExtractionAgent:
    """
    Agent that extracts structured data from PDF documents.
    Returns JSON output for Pydantic validation.
    """

    def __init__(self, pdf_path: str, use_ollama: bool = False, ollama_model: str = "llama3.2:1b"):
        self.pdf_path = pdf_path
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model

        # Get cached LLM
        if use_ollama:
            self.llm = get_cached_llm(use_ollama=True, model=ollama_model)
        else:
            self.llm = get_cached_llm(use_ollama=False, model="gpt-4")

    async def ainvoke(self, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Extract structured data from PDF using the given prompt.

        The prompt should guide WHAT to extract, and we validate the structure.
        """

        # Build extraction prompt - let the user prompt guide what to extract
        # Extract text from PDF once
        doc = pymupdf.open(self.pdf_path)
        pdf_text = "\n\n".join([page.get_text() for page in doc])
        doc.close()

        full_prompt = f"""
{user_prompt}
<pdfdata> {pdf_text} </pdfdata>
"""
        # Call LLM
        # print(f"FULL INPUT FOR DEBUG: {full_prompt}")
        response = await self.llm.ainvoke(full_prompt)

        # Extract JSON from response
        content = response.content
        print(content)

        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Get token counts from response context if available
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, 'response_metadata') and response.response_metadata:
            token_usage = response.response_metadata.get('token_usage', {})
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)

        # Fallback to simple counting if no metadata available
        if input_tokens == 0:
            input_tokens = count_tokens(full_prompt)
        if output_tokens == 0:
            output_tokens = count_tokens(content)

        # Count extracted data tokens (for efficiency calculation)
        extracted_tokens = 0
        try:
            if isinstance(content, str):
                data = json.loads(content)
                # Count tokens in all extracted values
                for value in data.values():
                    if isinstance(value, list):
                        extracted_tokens += sum(count_tokens(str(item)) for item in value)
                    else:
                        extracted_tokens += count_tokens(str(value))
        except (json.JSONDecodeError, TypeError, AttributeError):
            extracted_tokens = 0

        # Return as dict for OmniBAR
        return {
            "extraction": content,
            "prompt_used": user_prompt,
            "raw_response": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "extracted_tokens": extracted_tokens
        }

    def invoke(self, user_prompt: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for ainvoke."""
        return asyncio.run(self.ainvoke(user_prompt, **kwargs))


def create_extraction_agent(pdf_path: str, use_ollama: bool = False, ollama_model: str = "llama3.2:1b"):
    """Factory function to create document extraction agent."""
    return DocumentExtractionAgent(pdf_path, use_ollama, ollama_model)


# =============================================================================
# Thesaurus Replacer (reused from original)
# =============================================================================

class ThesaurusReplacer:
    """Generate synonym replacements for prompt variation."""

    def __init__(self):
        # Try to load NLTK
        try:
            import nltk     # General installation check
            from nltk.corpus import wordnet
            self.wordnet = wordnet
            self.has_nltk = True
        except ImportError:
            self.has_nltk = False

        # Fallback dictionary for common prompt words
        self.fallback_synonyms = {
            "extract": ["retrieve", "obtain", "pull", "get"],
            "main": ["primary", "key", "principal", "central"],
            "programming": ["coding", "development", "software"],
            "objectives": ["goals", "aims", "targets", "purposes"],
            "document": ["file", "text", "content", "record"],
            "following": ["subsequent", "next", "below"],
            "brief": ["short", "concise", "summary"],
            "key": ["essential", "critical", "important", "main"],
            "data": ["information", "details", "facts"],
        }

    def get_synonyms(self, word: str, pos_tag: str = None) -> List[str]:
        """Get synonyms for a word using WordNet or fallback."""
        # Try NLTK WordNet first
        if self.has_nltk:
            synonyms = self._get_wordnet_synonyms(word, pos_tag)
            if synonyms:
                return synonyms

        # Fall back to dictionary
        return self._get_fallback_synonyms(word)

    def _get_wordnet_synonyms(self, word: str, pos_tag: str = None) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()

        for synset in self.wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym.lower())

        result = list(synonyms)[:3]

        # If WordNet found no good synonyms, fall back to dictionary
        if not result:
            result = self._get_fallback_synonyms(word)

        return result

    def _get_fallback_synonyms(self, word: str) -> List[str]:
        """Get synonyms from fallback dictionary."""
        return self.fallback_synonyms.get(word.lower(), [])


# =============================================================================
# Prompt Variation Generator (reused from original)
# =============================================================================

class PromptVariationGenerator:
    """Generate prompt variations at specified exploration depth."""

    def __init__(self, exploration_depth: int = 2):
        self.exploration_depth = exploration_depth
        self.replacer = ThesaurusReplacer()

        # Words to skip (articles, prepositions, etc.)
        self.skip_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can'
        }

    def generate_variations(self, original_prompt: str) -> List[Dict[str, Any]]:
        """Generate all prompt variations up to exploration depth."""
        words = original_prompt.split()

        # Find replaceable words
        replaceable_positions = []
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?;:')
            if clean_word.lower() not in self.skip_words and len(clean_word) > 2:
                synonyms = self.replacer.get_synonyms(clean_word)
                if synonyms:
                    replaceable_positions.append((i, clean_word, synonyms))

        print(f"📝 Found {len(replaceable_positions)} replaceable words")

        # Generate variations at each depth
        all_variations = []

        # Depth 0: Original
        all_variations.append({
            'variation_id': 'original',
            'variation_prompt': original_prompt,
            'depth': 0,
            'changes': []
        })

        # Depths 1 through exploration_depth
        for depth in range(1, self.exploration_depth + 1):
            depth_variations = self._generate_depth_variations(
                words, replaceable_positions, depth
            )
            all_variations.extend(depth_variations)
            print(f"  Depth {depth}: {len(depth_variations)} variations")

        print(f"✅ Generated {len(all_variations)} total variations (including original)")
        return all_variations

    def _generate_depth_variations(
        self, words: List[str], replaceable_positions: List[tuple], depth: int
    ) -> List[Dict[str, Any]]:
        """Generate variations for a specific depth."""
        variations = []

        # Generate all combinations of positions at this depth
        for position_combo in combinations(range(len(replaceable_positions)), depth):
            # For each combination, get synonym options
            replacement_options = []
            for pos_idx in position_combo:
                pos, word, synonyms = replaceable_positions[pos_idx]
                replacement_options.append([(pos, word, syn) for syn in synonyms])

            # Generate all products of synonym choices
            for replacement_set in product(*replacement_options):
                new_words = words.copy()
                changes = []

                for pos, original, replacement in replacement_set:
                    new_words[pos] = replacement
                    changes.append({
                        'position': pos,
                        'original': original,
                        'replacement': replacement
                    })

                variation_prompt = ' '.join(new_words)

                # Create variation ID
                position_ids = '_'.join([str(c['position']) + c['replacement'][:3] for c in changes])
                variation_id = f"d{depth}_{position_ids}"

                variations.append({
                    'variation_id': variation_id,
                    'variation_prompt': variation_prompt,
                    'depth': depth,
                    'changes': changes
                })

        return variations


# =============================================================================
# Database Storage (reused from original)
# =============================================================================

class PromptRefinerDatabase:
    """SQLite database for storing prompt optimization results."""

    def __init__(self, db_path: str = "prompt_refiner_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Drop old tables if they exist (clean slate for Pydantic approach)
        cursor.execute("DROP TABLE IF EXISTS prompt_variations")
        cursor.execute("DROP TABLE IF EXISTS test_runs")

        # Test runs table
        cursor.execute("""
            CREATE TABLE test_runs (
                run_timestamp TEXT PRIMARY KEY,
                original_prompt TEXT NOT NULL,
                exploration_depth INTEGER NOT NULL,
                total_variations INTEGER NOT NULL,
                test_document TEXT,
                evaluation_type TEXT DEFAULT 'pydantic',
                created_at TEXT NOT NULL
            )
        """)

        # Prompt variations table - Pydantic validation schema
        cursor.execute("""
            CREATE TABLE prompt_variations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                variation_id TEXT NOT NULL,
                variation_prompt TEXT NOT NULL,
                depth INTEGER NOT NULL,
                changes TEXT,
                completeness_score REAL NOT NULL,
                type_score REAL NOT NULL,
                format_score REAL NOT NULL,
                pairing_score REAL NOT NULL,
                overall_score REAL NOT NULL,
                extracted_json TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                efficiency_score REAL,
                test_document TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_timestamp) REFERENCES test_runs(run_timestamp)
            )
        """)

        self.conn.commit()

    def store_run(
        self,
        run_timestamp: str,
        original_prompt: str,
        exploration_depth: int,
        total_variations: int,
        test_document: str
    ):
        """Store a test run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO test_runs (
                run_timestamp, original_prompt, exploration_depth,
                total_variations, test_document, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_timestamp, original_prompt, exploration_depth,
            total_variations, test_document, datetime.now().isoformat()
        ))
        self.conn.commit()

    def store_variation_result(
        self,
        run_timestamp: str,
        variation_id: str,
        variation_prompt: str,
        depth: int,
        changes: List[Dict],
        completeness_score: float,
        type_score: float,
        format_score: float,
        pairing_score: float,
        overall_score: float,
        extracted_json: str,
        input_tokens: int,
        output_tokens: int,
        efficiency_score: float,
        test_document: str
    ):
        """Store a single variation result."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_variations (
                run_timestamp, variation_id, variation_prompt, depth, changes,
                completeness_score, type_score, format_score, pairing_score, overall_score,
                extracted_json, input_tokens, output_tokens, efficiency_score,
                test_document, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_timestamp, variation_id, variation_prompt, depth,
            json.dumps(changes), completeness_score, type_score, format_score,
            pairing_score, overall_score, extracted_json, input_tokens, output_tokens,
            efficiency_score, test_document, datetime.now().isoformat()
        ))
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


# =============================================================================
# Main Prompt Refiner Suite Runner
# =============================================================================

async def run_prompt_refiner_suite(
    original_prompt: str,
    exploration_depth: int,
    pdf_path: str,
    iterations_per_variation: int = 2,
    use_ollama: bool = False,
    ollama_model: str = "phi4-mini:latest"
):
    """Run the complete prompt refiner suite with Pydantic validation."""

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🎯 PROMPT REFINER SUITE - PYDANTIC VALIDATION EDITION")
    print("=" * 70)
    print(f"📝 Original Prompt: {original_prompt}")
    print(f"🔍 Exploration Depth: {exploration_depth}")
    print(f"🔄 Iterations per variation: {iterations_per_variation}")
    print(f"📄 PDF Document: {pdf_path}")
    if use_ollama:
        print(f"🆓 Using Ollama model: {ollama_model} (cost-free!)")
    else:
        print("💰 Using OpenAI GPT-4 (API costs apply)")
    print()

    # Generate variations
    print("🔍 Generating variations with exploration depth:", exploration_depth)
    generator = PromptVariationGenerator(exploration_depth=exploration_depth)
    variations = generator.generate_variations(original_prompt)

    # Create database
    db = PromptRefinerDatabase()
    db.store_run(
        run_timestamp=run_timestamp,
        original_prompt=original_prompt,
        exploration_depth=exploration_depth,
        total_variations=len(variations),
        test_document=pdf_path
    )

    # Create algorithmic objectives
    completeness_obj = PydanticFieldCompletenessObjective(
        name="completeness_score",
        output_key="extraction",
        expected_model=ProteinExtraction
    )

    type_obj = PydanticTypeCorrectnessObjective(
        name="type_correctness",
        output_key="extraction",
        expected_model=ProteinExtraction
    )

    format_obj = PydanticFormatValidityObjective(
        name="format_validity",
        output_key="extraction",
        expected_model=ProteinExtraction
    )

    pairing_obj = ProteinOrganismPairingObjective(
        name="pairing_validation",
        output_key="extraction"
    )

    combined_objective = CombinedBenchmarkObjective(
        name="pydantic_validation",
        objectives=[completeness_obj, type_obj, format_obj, pairing_obj]
    )

    # Test each variation
    print(f"\n🧪 Testing {len(variations)} prompt variations...")
    print("=" * 70)

    for idx, variation in enumerate(variations, 1):
        print(f"\n[{idx}/{len(variations)}] Testing: {variation['variation_id']}")
        print(f"   Prompt: {variation['variation_prompt'][:80]}...")

        # Create benchmark
        benchmark = Benchmark(
            name=f"Extraction Test - {variation['variation_id']}",
            input_kwargs={"user_prompt": variation['variation_prompt']},
            objective=combined_objective,
            iterations=iterations_per_variation,
            verbose=False
        )

        # Get the raw extraction output by calling the agent directly
        # This allows us to capture token counts and raw JSON
        agent = create_extraction_agent(pdf_path, use_ollama, ollama_model)

        try:
            # Call the agent directly to get raw output with token data
            direct_result = await agent.ainvoke(variation['variation_prompt'])
            extracted_json = direct_result.get('extraction', '')
            input_tokens = direct_result.get('input_tokens', 0)
            output_tokens = direct_result.get('output_tokens', 0)
            extracted_tokens = direct_result.get('extracted_tokens', 0)
            efficiency_score = extracted_tokens / input_tokens if input_tokens > 0 else 0.0

            print(f"   ✅ Extracted: {len(extracted_json)} chars, {input_tokens}→{output_tokens} tokens, eff={efficiency_score:.3f}")
        except Exception as e:
            print(f"   ⚠️ Direct extraction failed: {e}")
            extracted_json = ""
            input_tokens = 0
            output_tokens = 0
            extracted_tokens = 0
            efficiency_score = 0.0

        # Run benchmark for validation scores
        benchmarker = OmniBarmarker(
            executor_fn=lambda: agent,
            executor_kwargs={},
            initial_input=[benchmark],
            enable_logging=True  # Need logging enabled to extract scores!
        )

        _ = await benchmarker.benchmark_async(max_concurrent=1)

        # Extract scores from validation results
        logs = benchmarker.logger.get_all_logs()

        completeness_score = 0.0
        type_score = 0.0
        format_score = 0.0
        pairing_score = 0.0

        for log in logs:
            obj_name = log.metadata.get('objective_name', '').lower()
            for entry in log.entries:
                if hasattr(entry.eval_result, 'result') and entry.eval_result.result is not None:
                    score = float(entry.eval_result.result)
                    if 'completeness' in obj_name:
                        completeness_score = max(completeness_score, score)
                    elif 'type' in obj_name:
                        type_score = max(type_score, score)
                    elif 'format' in obj_name:
                        format_score = max(format_score, score)
                    elif 'pairing' in obj_name:
                        pairing_score = max(pairing_score, score)

        overall_score = (completeness_score + type_score + format_score + pairing_score) / 4.0

        print(f"   ✅ Completeness: {completeness_score:.2f} | Type: {type_score:.2f} | Format: {format_score:.2f} | Pairing: {pairing_score:.2f} | Overall: {overall_score:.2f}")
        print(f"   📊 Tokens: {input_tokens} in → {output_tokens} out → {extracted_tokens} extracted | Efficiency: {efficiency_score:.3f}")

        # Store results
        db.store_variation_result(
            run_timestamp=run_timestamp,
            variation_id=variation['variation_id'],
            variation_prompt=variation['variation_prompt'],
            depth=variation['depth'],
            changes=variation['changes'],
            completeness_score=completeness_score,
            type_score=type_score,
            format_score=format_score,
            pairing_score=pairing_score,
            overall_score=overall_score,
            extracted_json=extracted_json,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            efficiency_score=efficiency_score,
            test_document=pdf_path
        )

    db.close()

    print("\n" + "=" * 70)
    print("✅ PROMPT OPTIMIZATION COMPLETE!")
    print("💾 Results saved to: prompt_refiner_results.db")
    print(f"   Run timestamp: {run_timestamp}")
    print("\n📊 View results:")
    print(f"   python visualize_prompt_results.py --run {run_timestamp}")
    print(f"   python visualize_prompt_landscape.py --run {run_timestamp}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():

    # Example default extraction prompt:
    prompt = """Extract all named proteins and their source organisms from this research paper (<pdfdata>).
Return your answer as a JSON object with these fields:
- "proteins": list of all protein names mentioned in the document (format = Gene name)
- "organisms": list of source organisms for each protein in the same order (format = Genus_species)
Return ONLY the JSON object."""

    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Prompt Refiner Suite - Pydantic Validation Edition"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=prompt,
        help="Original prompt to optimize"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Exploration depth (1-3)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="../test.pdf",
        help="Path to PDF document"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Iterations per variation"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama instead of OpenAI (cost-free!)"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="phi4-mini:latest",
        help="Ollama model to use (default: llama3.2:1b)"
    )

    args = parser.parse_args()

    # Run the suite
    asyncio.run(run_prompt_refiner_suite(
        original_prompt=args.prompt,
        exploration_depth=args.depth,
        pdf_path=args.pdf,
        iterations_per_variation=args.iterations,
        use_ollama=args.ollama,
        ollama_model=args.ollama_model
    ))


if __name__ == "__main__":
    main()
