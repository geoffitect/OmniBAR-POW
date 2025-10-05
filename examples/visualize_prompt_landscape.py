#!/usr/bin/env python3
"""
3D Prompt Landscape Visualizer
==============================

Creates an interactive HTML visualization of prompt optimization results as a 3D landscape.

Visual Dimensions:
- X-axis: Clarity score
- Y-axis: Effectiveness score
- Z-axis: Precision score
- Size: Overall score
- Color: Exploration depth
- Effect: Improvement over original (pulse = better)

Usage:
    python visualize_prompt_landscape.py --run <timestamp>
    python visualize_prompt_landscape.py --latest
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import math
import random
from collections import Counter


class PromptLandscapeVisualizer:
    """
    Creates an interactive 3D visualization of prompt optimization landscapes.
    """

    def __init__(self, db_path: str = "prompt_refiner_results.db"):
        """Initialize with database path"""
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    @staticmethod
    def compute_cosine_similarity(text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using simple word vectors.

        Returns value between 0.0 (completely different) and 1.0 (identical).
        """
        # Tokenize and create word frequency vectors
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        # Get unique words
        all_words = set(words1 + words2)

        # Create frequency vectors
        vec1 = [words1.count(word) for word in all_words]
        vec2 = [words2.count(word) for word in all_words]

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def apply_jitter(values: List[float], jitter_amount: float = 0.02) -> List[float]:
        """
        Apply small random jitter to values to separate overlapping points.

        Args:
            values: List of values to jitter
            jitter_amount: Maximum jitter as fraction of range (default 2%)

        Returns:
            List of jittered values
        """
        return [v + random.uniform(-jitter_amount, jitter_amount) for v in values]

    def get_latest_run(self) -> str:
        """Get the most recent run timestamp"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT run_timestamp
            FROM test_runs
            ORDER BY created_at DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        return result[0] if result else None

    def get_run_data(self, run_timestamp: str) -> Dict[str, Any]:
        """Get all data for a specific run"""
        cursor = self.conn.cursor()

        # Get run metadata
        cursor.execute("""
            SELECT original_prompt, exploration_depth, total_variations, created_at
            FROM test_runs
            WHERE run_timestamp = ?
        """, (run_timestamp,))

        run_info = cursor.fetchone()
        if not run_info:
            raise ValueError(f"Run not found: {run_timestamp}")

        # Get all variations (with backwards compatibility)
        try:
            cursor.execute("""
                SELECT variation_id, variation_prompt, depth, changes,
                       completeness_score, type_score, format_score, pairing_score, overall_score,
                       extracted_json, input_tokens, output_tokens, efficiency_score
                FROM prompt_variations
                WHERE run_timestamp = ?
                ORDER BY overall_score DESC
            """, (run_timestamp,))
        except sqlite3.OperationalError:
            # Fallback for older database schema without new fields
            cursor.execute("""
                SELECT variation_id, variation_prompt, depth, changes,
                       completeness_score, type_score, format_score, overall_score
                FROM prompt_variations
                WHERE run_timestamp = ?
                ORDER BY overall_score DESC
            """, (run_timestamp,))

        variations = []
        original_score = None
        original_prompt = run_info[0]

        for row in cursor.fetchall():
            # Handle both old and new schema
            if len(row) >= 13:  # New schema with all fields
                var = {
                    'id': row[0],
                    'prompt': row[1],
                    'depth': row[2],
                    'changes': json.loads(row[3]) if row[3] else [],
                    'clarity': row[4],      # completeness_score
                    'effectiveness': row[5], # type_score
                    'precision': row[6],    # format_score
                    'pairing': row[7],      # pairing_score
                    'overall': row[8],      # overall_score
                    'extracted_json': row[9] if len(row) > 9 and row[9] else '',
                    'input_tokens': row[10] if len(row) > 10 and row[10] else 0,
                    'output_tokens': row[11] if len(row) > 11 and row[11] else 0,
                    'efficiency': row[12] if len(row) > 12 and row[12] else 0.0
                }
            else:  # Old schema without pairing, extracted_json, tokens
                var = {
                    'id': row[0],
                    'prompt': row[1],
                    'depth': row[2],
                    'changes': json.loads(row[3]) if row[3] else [],
                    'clarity': row[4],      # completeness_score
                    'effectiveness': row[5], # type_score
                    'precision': row[6],    # format_score
                    'pairing': 0.0,         # Default for old schema
                    'overall': row[7],      # overall_score
                    'extracted_json': '',   # Default for old schema
                    'input_tokens': 0,      # Default for old schema
                    'output_tokens': 0,     # Default for old schema
                    'efficiency': 0.0       # Default for old schema
                }

            # Calculate cosine similarity to original prompt
            var['similarity'] = self.compute_cosine_similarity(var['prompt'], original_prompt)

            # Parse extracted data for histograms
            var['proteins'] = []
            var['organisms'] = []
            if var['extracted_json']:
                try:
                    data = json.loads(var['extracted_json'])
                    var['proteins'] = data.get('proteins', [])
                    var['organisms'] = data.get('organisms', [])
                except json.JSONDecodeError:
                    pass

            variations.append(var)

            if var['id'] == 'original':
                original_score = var['overall']

        return {
            'run_timestamp': run_timestamp,
            'original_prompt': run_info[0],
            'exploration_depth': run_info[1],
            'total_variations': run_info[2],
            'created_at': run_info[3],
            'original_score': original_score,
            'variations': variations
        }

    def generate_score_segmented_data(self, variations: List[Dict]) -> Dict[str, Any]:
        """Generate histogram data segmented by OmniBAR score ranges."""

        # Define score ranges
        score_ranges = {
            'perfect': {'min': 1.0, 'max': 1.0, 'label': '100% Score', 'variations': []},
            'excellent': {'min': 0.8, 'max': 0.99, 'label': '80-99% Score', 'variations': []},
            'good': {'min': 0.6, 'max': 0.79, 'label': '60-79% Score', 'variations': []},
            'poor': {'min': 0.0, 'max': 0.59, 'label': '<60% Score', 'variations': []}
        }

        # Segment variations by score
        for var in variations:
            score = var.get('overall', 0)
            for range_key, range_data in score_ranges.items():
                if range_data['min'] <= score <= range_data['max']:
                    range_data['variations'].append(var)
                    break

        # Generate histograms for each score range
        segmented_data = {}

        for range_key, range_data in score_ranges.items():
            vars_in_range = range_data['variations']

            if not vars_in_range:
                segmented_data[range_key] = {
                    'label': range_data['label'],
                    'count': 0,
                    'protein_histogram': {},
                    'organism_histogram': {},
                    'avg_efficiency': 0,
                    'variations': []
                }
                continue

            # Collect proteins and organisms for this score range
            all_proteins = []
            all_organisms = []
            efficiency_scores = []

            for var in vars_in_range:
                if var.get('proteins') and isinstance(var['proteins'], list):
                    for protein in var['proteins']:
                        if isinstance(protein, str):
                            all_proteins.append(protein)
                        elif isinstance(protein, list):
                            all_proteins.extend([str(p) for p in protein])
                        else:
                            all_proteins.append(str(protein))

                if var.get('organisms') and isinstance(var['organisms'], list):
                    for organism in var['organisms']:
                        if isinstance(organism, str):
                            all_organisms.append(organism)
                        elif isinstance(organism, list):
                            all_organisms.extend([str(o) for o in organism])
                        else:
                            all_organisms.append(str(organism))

                efficiency_scores.append(var.get('efficiency', 0))

            # Generate histograms
            protein_counts = Counter(all_proteins)
            organism_counts = Counter(all_organisms)

            segmented_data[range_key] = {
                'label': range_data['label'],
                'count': len(vars_in_range),
                'protein_histogram': dict(protein_counts.most_common(10)),
                'organism_histogram': dict(organism_counts.most_common(10)),
                'avg_efficiency': sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0,
                'variations': [
                    {
                        'id': v['id'],
                        'prompt': v['prompt'],
                        'overall': v['overall'],
                        'proteins': v.get('proteins', []),
                        'organisms': v.get('organisms', []),
                        'efficiency': v.get('efficiency', 0)
                    } for v in vars_in_range
                ]
            }

        return segmented_data

    def _generate_score_panels_html(self, segmented_data: Dict[str, Any]) -> str:
        """Generate HTML for score-based histogram panels."""
        panels_html = ""

        panel_order = ['perfect', 'excellent', 'good', 'poor']

        for panel_key in panel_order:
            panel_data = segmented_data.get(panel_key, {})
            if panel_data['count'] == 0:
                continue

            panels_html += f"""
            <div class="score-panel {panel_key}">
                <div class="score-panel-header">
                    <div class="score-title">{panel_data['label']}</div>
                    <div class="score-count">{panel_data['count']} variations</div>
                </div>

                <div class="histogram-section">
                    <div class="histogram-title">🧬 Proteins</div>
                    <div id="protein-{panel_key}" class="histogram-plot"></div>
                </div>

                <div class="histogram-section">
                    <div class="histogram-title">🦠 Organisms</div>
                    <div id="organism-{panel_key}" class="histogram-plot"></div>
                </div>

                <div class="interactive-hint">
                    💡 Hover over bars to see prompts and extracted data
                </div>
            </div>
            """

        return panels_html

    def generate_html(self, run_timestamp: str, output_file: str = None) -> str:
        """Generate interactive 3D HTML visualization"""

        data = self.get_run_data(run_timestamp)

        if output_file is None:
            output_file = f"prompt_landscape_{run_timestamp}.html"

        # Prepare data for JavaScript
        variations = data['variations']

        # Generate score-segmented histogram data
        segmented_data = self.generate_score_segmented_data(variations)

        # Calculate improvement percentages
        original_score = data['original_score'] or 0.5
        for var in variations:
            var['improvement'] = ((var['overall'] - original_score) / original_score * 100) if original_score > 0 else 0

        # Color mapping for depth (using distinct colors)
        depth_colors = {
            0: '#808080',  # Gray for original
            1: '#3498db',  # Blue
            2: '#2ecc71',  # Green
            3: '#e74c3c',  # Red
            4: '#f39c12',  # Orange
        }

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Landscape - {run_timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}

        .main-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .score-panel {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            min-height: 600px;
        }}

        .score-panel.perfect {{
            border: 2px solid #2ecc71;
        }}

        .score-panel.excellent {{
            border: 2px solid #3498db;
        }}

        .score-panel.good {{
            border: 2px solid #f39c12;
        }}

        .score-panel.poor {{
            border: 2px solid #e74c3c;
        }}

        .score-panel-header {{
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .score-title {{
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 5px;
            color: #ffffff;
        }}

        .score-count {{
            font-size: 0.9em;
            opacity: 0.8;
            color: #ffffff;
        }}

        .histogram-section {{
            margin-bottom: 25px;
        }}

        .histogram-title {{
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #ffffff;
        }}

        .histogram-plot {{
            height: 200px;
            margin-bottom: 15px;
        }}

        .interactive-hint {{
            font-size: 0.8em;
            opacity: 0.7;
            text-align: center;
            margin-top: 10px;
            color: #ffffff;
        }}

        .histogram-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin-bottom: 5px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            font-size: 0.9em;
        }}

        .histogram-bar {{
            height: 4px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 2px;
            margin-top: 4px;
        }}

        .efficiency-metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            margin-bottom: 10px;
        }}

        .efficiency-value {{
            font-weight: bold;
            color: #2ecc71;
        }}

        header {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .meta-item {{
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
        }}

        .meta-label {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }}

        .meta-value {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .original-prompt {{
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-style: italic;
            border-left: 4px solid #fff;
        }}

        #visualization {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
        }}

        .legend {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }}

        .legend h2 {{
            margin-bottom: 20px;
        }}

        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .legend-item {{
            background: rgba(255, 255, 255, 0.15);
            padding: 15px;
            border-radius: 10px;
        }}

        .legend-item h3 {{
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .color-sample {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            vertical-align: middle;
        }}

        .stats {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
        }}

        .stats h2 {{
            margin-bottom: 20px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}

        footer {{
            text-align: center;
            margin-top: 30px;
            opacity: 0.8;
        }}

        /* Compact mutation notation hover tooltips */
        .plotly .hoverlabel {{
            max-width: 300px;
            white-space: pre-line;
            font-family: 'Courier New', monospace;
            font-size: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌌 Prompt Optimization Landscape</h1>
            <div class="meta">
                <div class="meta-item">
                    <div class="meta-label">Run Timestamp</div>
                    <div class="meta-value">{run_timestamp}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Exploration Depth</div>
                    <div class="meta-value">Depth {data['exploration_depth']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Variations</div>
                    <div class="meta-value">{data['total_variations']}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Created</div>
                    <div class="meta-value">{data['created_at']}</div>
                </div>
            </div>
            <div class="original-prompt">
                <strong>Original Prompt:</strong> {data['original_prompt']}
            </div>
        </header>

        <div class="main-content">
            {self._generate_score_panels_html(segmented_data)}
        </div>

        <div class="legend">
            <h2>📊 Visual Encoding</h2>
            <div class="legend-grid">
                <div class="legend-item">
                    <h3>🎯 Score Segmentation</h3>
                    <p><span class="color-sample" style="background: #2ecc71; border: 2px solid #2ecc71;"></span> <strong>Perfect (100%):</strong> All objectives achieved</p>
                    <p><span class="color-sample" style="background: #3498db; border: 2px solid #3498db;"></span> <strong>Excellent (80-99%):</strong> Near-perfect validation</p>
                    <p><span class="color-sample" style="background: #f39c12; border: 2px solid #f39c12;"></span> <strong>Good (60-79%):</strong> Decent performance</p>
                    <p><span class="color-sample" style="background: #e74c3c; border: 2px solid #e74c3c;"></span> <strong>Poor (<60%):</strong> Low validation scores</p>
                </div>

                <div class="legend-item">
                    <h3>📊 Histogram Bars</h3>
                    <p><strong>🧬 Proteins:</strong> Count of each protein extracted</p>
                    <p><strong>🦠 Organisms:</strong> Count of each organism extracted</p>
                    <p style="margin-top: 10px; opacity: 0.8;">Bar length = frequency of extraction across variations</p>
                </div>

                <div class="legend-item">
                    <h3>🔍 Interactive Hover</h3>
                    <p><strong>Hover over bars</strong> to see:</p>
                    <p>• Original prompts that extracted that entity</p>
                    <p>• OmniBAR validation scores</p>
                    <p>• Complete extraction results</p>
                    <p style="margin-top: 10px; opacity: 0.8;">Identify which prompts produce specific extractions</p>
                </div>

                <div class="legend-item">
                    <h3>⚡ Efficiency Metrics</h3>
                    <p><strong>Token Efficiency:</strong> extracted_tokens / input_tokens</p>
                    <p><strong>Panel Counts:</strong> Number of variations per score range</p>
                    <p style="margin-top: 10px; opacity: 0.8;">Higher efficiency = more extraction per token spent</p>
                </div>
            </div>
        </div>

        <div class="stats">
            <h2>📈 Performance Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Best Overall</div>
                    <div class="stat-value">{max((v['overall'] for v in variations), default=0):.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average Overall</div>
                    <div class="stat-value">{sum(v['overall'] for v in variations) / len(variations) if variations else 0:.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Best Clarity</div>
                    <div class="stat-value">{max((v['clarity'] for v in variations), default=0):.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Best Effectiveness</div>
                    <div class="stat-value">{max((v['effectiveness'] for v in variations), default=0):.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Best Precision</div>
                    <div class="stat-value">{max((v['precision'] for v in variations), default=0):.3f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max Improvement</div>
                    <div class="stat-value">{max((v['improvement'] for v in variations), default=0):+.1f}%</div>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by Prompt Refiner Suite • OmniBAR</p>
        </footer>
    </div>

    <script>
        // Data from Python
        const variations = {json.dumps(variations, indent=8)};
        const depthColors = {json.dumps(depth_colors)};
        const originalScore = {original_score};
        const originalPrompt = {json.dumps(data['original_prompt'])};

        // Apply jitter to separate overlapping points
        function applyJitter(values, amount = 0.015) {{
            return values.map(v => v + (Math.random() - 0.5) * amount);
        }}

        // Prepare 3D scatter plot data with jitter for clustered points
        const x = applyJitter(variations.map(v => v.clarity));
        const y = applyJitter(variations.map(v => v.effectiveness));
        const z = applyJitter(variations.map(v => v.precision));

        // Size based on overall score (scale for visibility)
        const sizes = variations.map(v => 10 + v.overall * 30);

        // Color based on COSINE SIMILARITY to original prompt (NEW!)
        // Red (low similarity) → Yellow → Green (high similarity)
        const colors = variations.map(v => {{
            const sim = v.similarity;  // 0.0 to 1.0
            // Use HSL color space: Hue from 0° (red) to 120° (green)
            const hue = sim * 120;  // 0° = red (different), 120° = green (similar)
            const saturation = 70;
            const lightness = 45;
            return `hsl(${{hue}}, ${{saturation}}%, ${{lightness}}%)`;
        }});

        // Opacity: higher for points in dense clusters (more transparent = less clustered)
        const opacities = variations.map((v, i) => {{
            // Count nearby points (simple clustering detection)
            let nearby = 0;
            const threshold = 0.05;
            for (let j = 0; j < variations.length; j++) {{
                if (i !== j) {{
                    const dist = Math.sqrt(
                        Math.pow(variations[i].clarity - variations[j].clarity, 2) +
                        Math.pow(variations[i].effectiveness - variations[j].effectiveness, 2) +
                        Math.pow(variations[i].precision - variations[j].precision, 2)
                    );
                    if (dist < threshold) nearby++;
                }}
            }}
            // More nearby points = higher opacity (make clusters more visible)
            return Math.max(0.4, Math.min(1.0, 0.5 + nearby * 0.1));
        }});

        // Hover text
        const hoverTexts = variations.map(v =>
            `<b>${{v.id}}</b><br>` +
            `Prompt: ${{v.prompt.substring(0, 60)}}...<br><br>` +
            `Overall: ${{v.overall.toFixed(3)}}<br>` +
            `Completeness: ${{v.clarity.toFixed(3)}}<br>` +
            `Type: ${{v.effectiveness.toFixed(3)}}<br>` +
            `Format: ${{v.precision.toFixed(3)}}<br>` +
            `Similarity: ${{v.similarity.toFixed(3)}}<br>` +
            `Improvement: ${{v.improvement >= 0 ? '+' : ''}}${{v.improvement.toFixed(1)}}%<br>` +
            `Depth: ${{v.depth}}`
        );

        // Create trace
        const trace = {{
            type: 'scatter3d',
            mode: 'markers',
            x: x,
            y: y,
            z: z,
            marker: {{
                size: sizes,
                color: colors,
                opacity: opacities,
                line: {{
                    color: 'white',
                    width: 1
                }}
            }},
            text: hoverTexts,
            hoverinfo: 'text',
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        // Layout
        const layout = {{
            title: {{
                text: '3D Prompt Optimization Landscape',
                font: {{
                    size: 24,
                    color: '#333'
                }}
            }},
            scene: {{
                xaxis: {{
                    title: 'Completeness Score',
                    range: [0, 1],
                    gridcolor: '#e0e0e0'
                }},
                yaxis: {{
                    title: 'Type Correctness',
                    range: [0, 1],
                    gridcolor: '#e0e0e0'
                }},
                zaxis: {{
                    title: 'Format Validity',
                    range: [0, 1],
                    gridcolor: '#e0e0e0'
                }},
                camera: {{
                    eye: {{
                        x: 1.5,
                        y: 1.5,
                        z: 1.3
                    }}
                }},
                bgcolor: '#f8f9fa'
            }},
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            height: 800,
            margin: {{
                l: 0,
                r: 0,
                b: 0,
                t: 50
            }}
        }};

        // Config
        const config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage'],
            displaylogo: false
        }};

        // Generate score-segmented histograms
        const segmentedData = {json.dumps(segmented_data)};
        const panelColors = {{
            'perfect': 'rgba(46, 204, 113, 0.8)',
            'excellent': 'rgba(52, 152, 219, 0.8)',
            'good': 'rgba(243, 156, 18, 0.8)',
            'poor': 'rgba(231, 76, 60, 0.8)'
        }};

        // Function to create compact mutation notation hover text
        function createHoverText(variations, entity, entityList) {{
            return entityList.map(item => {{
                const relevantVars = variations.filter(v =>
                    (entity === 'protein' ? v.proteins : v.organisms).includes(item)
                );

                // Use the global original prompt for comparison
                const originalWords = originalPrompt.split(' ');

                // Create DNA-style alignment content
                const hoverContent = relevantVars.map(v => {{
                    const proteinCount = v.proteins.length;
                    const organismCount = v.organisms.length;

                    // Create compact mutation notation
                    let mutations = '';
                    if (v.prompt === originalPrompt) {{
                        mutations = 'ORIGINAL';
                    }} else {{
                        const varWords = v.prompt.split(' ');
                        const mutationList = [];

                        // Find mutations (word-by-word comparison)
                        const maxLength = Math.max(originalWords.length, varWords.length);
                        for (let i = 0; i < maxLength; i++) {{
                            const origWord = originalWords[i] || '';
                            const varWord = varWords[i] || '';

                            if (origWord !== varWord) {{
                                if (varWord === '') {{
                                    mutationList.push(`${{origWord}}_${{i+1}}_DEL`);  // deletion
                                }} else if (origWord === '') {{
                                    mutationList.push(`INS_${{i+1}}_${{varWord}}`);   // insertion
                                }} else {{
                                    mutationList.push(`${{origWord}}_${{i+1}}_${{varWord}}`);  // substitution
                                }}
                            }}
                        }}

                        // Limit to 2 most significant mutations to keep it short
                        mutations = mutationList.slice(0, 2).join(',') + (mutationList.length > 2 ? '...' : '');
                        if (mutations === '') mutations = 'IDENTICAL';
                    }}

                    return [
                        `${{v.id}} (Score: ${{v.overall.toFixed(3)}})`,
                        `${{mutations}}`,
                        `${{proteinCount}}p ${{organismCount}}o`
                    ].join('\\n');
                }});

                return hoverContent.join('\\n\\n');
            }});
        }}

        // Generate histograms for each score panel
        Object.keys(segmentedData).forEach(panelKey => {{
            const panelData = segmentedData[panelKey];
            if (panelData.count === 0) return;

            const color = panelColors[panelKey];

            // Protein histogram
            const proteinLabels = Object.keys(panelData.protein_histogram);
            const proteinValues = Object.values(panelData.protein_histogram);
            const proteinHoverText = createHoverText(panelData.variations, 'protein', proteinLabels);

            if (proteinLabels.length > 0) {{
                Plotly.newPlot(`protein-${{panelKey}}`, [{{
                    x: proteinValues,
                    y: proteinLabels,
                    type: 'bar',
                    orientation: 'h',
                    text: proteinHoverText,
                    hoverinfo: 'text',
                    marker: {{
                        color: color,
                        line: {{
                            color: color.replace('0.8', '1'),
                            width: 1
                        }}
                    }}
                }}], {{
                    margin: {{l: 80, r: 20, t: 20, b: 40}},
                    height: 200,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: 'white', size: 9}},
                    xaxis: {{color: 'white', gridcolor: 'rgba(255,255,255,0.2)', title: 'Count'}},
                    yaxis: {{color: 'white'}}
                }}, {{
                    responsive: true,
                    displayModeBar: false,
                    hovermode: 'closest',
                    hoverlabel: {{
                        bgcolor: 'rgba(0,0,0,0.9)',
                        bordercolor: 'white',
                        font: {{color: 'white', size: 11, family: 'monospace'}},
                        align: 'left',
                        namelength: -1
                    }}
                }});
            }}

            // Organism histogram
            const organismLabels = Object.keys(panelData.organism_histogram);
            const organismValues = Object.values(panelData.organism_histogram);
            const organismHoverText = createHoverText(panelData.variations, 'organism', organismLabels);

            if (organismLabels.length > 0) {{
                Plotly.newPlot(`organism-${{panelKey}}`, [{{
                    x: organismValues,
                    y: organismLabels,
                    type: 'bar',
                    orientation: 'h',
                    text: organismHoverText,
                    hoverinfo: 'text',
                    marker: {{
                        color: color,
                        line: {{
                            color: color.replace('0.8', '1'),
                            width: 1
                        }}
                    }}
                }}], {{
                    margin: {{l: 80, r: 20, t: 20, b: 40}},
                    height: 200,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{color: 'white', size: 9}},
                    xaxis: {{color: 'white', gridcolor: 'rgba(255,255,255,0.2)', title: 'Count'}},
                    yaxis: {{color: 'white'}}
                }}, {{
                    responsive: true,
                    displayModeBar: false,
                    hovermode: 'closest',
                    hoverlabel: {{
                        bgcolor: 'rgba(0,0,0,0.9)',
                        bordercolor: 'white',
                        font: {{color: 'white', size: 11, family: 'monospace'}},
                        align: 'left',
                        namelength: -1
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""

        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_file

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate 3D interactive visualization of prompt optimization landscape'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='prompt_refiner_results.db',
        help='Path to results database'
    )
    parser.add_argument(
        '--run',
        type=str,
        help='Specific run timestamp to visualize'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Visualize the most recent run'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output HTML file path (default: auto-generated)'
    )

    args = parser.parse_args()

    try:
        visualizer = PromptLandscapeVisualizer(db_path=args.db)

        # Determine which run to visualize
        if args.latest or (not args.run):
            run_timestamp = visualizer.get_latest_run()
            if not run_timestamp:
                print("❌ No runs found in database")
                print("\n💡 Run the prompt refiner suite first:")
                print("   python prompt_refiner_suite.py --prompt 'Your prompt here'")
                return
            print(f"📊 Visualizing latest run: {run_timestamp}")
        else:
            run_timestamp = args.run

        # Generate HTML
        output_file = visualizer.generate_html(run_timestamp, args.output)

        print(f"\n✅ Generated 3D visualization: {output_file}")
        print(f"\n🌐 Open in browser:")
        print(f"   file://{Path(output_file).absolute()}")

        # Try to open in default browser
        import webbrowser
        webbrowser.open(f"file://{Path(output_file).absolute()}")
        print(f"\n🚀 Opening in default browser...")

        visualizer.close()

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Run the prompt refiner suite first to generate results:")
        print("   python prompt_refiner_suite.py --prompt 'Your prompt here'")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
