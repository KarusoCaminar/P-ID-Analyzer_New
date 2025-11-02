"""
Vollständige Auswertung aller Test-Ergebnisse:
- KPIs aus JSON-Dateien
- Visualisierungen (PNG-Dateien)
- Trends über Zeit
- Muster und Probleme identifizieren
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

def analyze_all_results():
    """Analysiert alle verfügbaren Test-Ergebnisse."""
    
    results = {
        'kpi_analysis': analyze_kpis(),
        'visualization_analysis': analyze_visualizations(),
        'trend_analysis': analyze_trends(),
        'problem_patterns': identify_problems(),
        'recommendations': generate_recommendations()
    }
    
    # Report schreiben
    report_path = Path('outputs/debug/comprehensive_test_analysis.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Markdown Report
    markdown_report = generate_markdown_report(results)
    markdown_path = Path('outputs/debug/comprehensive_test_analysis.md')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"\n[OK] Analyse abgeschlossen:")
    print(f"  - JSON Report: {report_path}")
    print(f"  - Markdown Report: {markdown_path}")
    
    return results

def analyze_kpis():
    """Analysiert alle KPI-JSON-Dateien."""
    
    kpi_files = list(Path('outputs').rglob('*kpis.json'))
    print(f"\nAnalysiere {len(kpi_files)} KPI-Dateien...")
    
    all_kpis = []
    metrics = {
        'quality_scores': [],
        'total_elements': [],
        'total_connections': [],
        'avg_element_confidence': [],
        'avg_connection_confidence': [],
        'graph_density': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for kpi_file in kpi_files:
        try:
            with open(kpi_file, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                all_kpis.append({
                    'file': str(kpi_file),
                    'timestamp': kpi_file.stat().st_mtime,
                    'data': kpi_data
                })
                
                # Metriken sammeln
                if 'quality_score' in kpi_data:
                    metrics['quality_scores'].append(kpi_data['quality_score'])
                if 'total_elements' in kpi_data:
                    metrics['total_elements'].append(kpi_data['total_elements'])
                if 'total_connections' in kpi_data:
                    metrics['total_connections'].append(kpi_data['total_connections'])
                if 'avg_element_confidence' in kpi_data:
                    metrics['avg_element_confidence'].append(kpi_data['avg_element_confidence'])
                if 'avg_connection_confidence' in kpi_data:
                    metrics['avg_connection_confidence'].append(kpi_data['avg_connection_confidence'])
                if 'graph_density' in kpi_data:
                    metrics['graph_density'].append(kpi_data['graph_density'])
                if 'element_precision' in kpi_data:
                    metrics['precision'].append(kpi_data['element_precision'])
                if 'element_recall' in kpi_data:
                    metrics['recall'].append(kpi_data['element_recall'])
                if 'element_f1' in kpi_data:
                    metrics['f1'].append(kpi_data['element_f1'])
        except Exception as e:
            print(f"  ⚠ Fehler beim Lesen {kpi_file}: {e}")
    
    # Statistik berechnen
    stats = {}
    for key, values in metrics.items():
        if values:
            stats[key] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0
            }
    
    return {
        'total_files': len(all_kpis),
        'metrics_summary': stats,
        'all_kpis': all_kpis[:10],  # Nur erste 10 für Übersicht
        'problematic_files': [
            k for k in all_kpis 
            if k['data'].get('quality_score', 100) < 50 or
               k['data'].get('hallucinated_elements', 0) > 10 or
               k['data'].get('missed_elements', 0) > 10
        ]
    }

def analyze_visualizations():
    """Analysiert verfügbare Visualisierungen."""
    
    viz_types = {
        'debug_map': list(Path('outputs').rglob('*debug_map.png')),
        'kpi_dashboard': list(Path('outputs').rglob('*kpi_dashboard.png')),
        'confidence_map': list(Path('outputs').rglob('*confidence_map.png')),
        'score_curve': list(Path('outputs').rglob('*score_curve.png'))
    }
    
    analysis = {}
    for viz_type, files in viz_types.items():
        analysis[viz_type] = {
            'count': len(files),
            'recent_files': sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        }
    
    return analysis

def analyze_trends():
    """Analysiert Trends über Zeit."""
    
    kpi_files = list(Path('outputs').rglob('*kpis.json'))
    
    timeline = []
    for kpi_file in kpi_files:
        try:
            mtime = kpi_file.stat().st_mtime
            with open(kpi_file, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                timeline.append({
                    'timestamp': mtime,
                    'quality_score': kpi_data.get('quality_score', None),
                    'total_elements': kpi_data.get('total_elements', None),
                    'avg_confidence': kpi_data.get('avg_element_confidence', None)
                })
        except:
            continue
    
    timeline.sort(key=lambda x: x['timestamp'])
    
    # Zeitbasierte Gruppen
    if timeline:
        recent = timeline[-10:]
        older = timeline[:-10] if len(timeline) > 10 else []
        
        return {
            'total_data_points': len(timeline),
            'recent_performance': {
                'avg_quality': statistics.mean([t['quality_score'] for t in recent if t['quality_score'] is not None]) if any(t['quality_score'] is not None for t in recent) else None,
                'avg_elements': statistics.mean([t['total_elements'] for t in recent if t['total_elements'] is not None]) if any(t['total_elements'] is not None for t in recent) else None
            },
            'older_performance': {
                'avg_quality': statistics.mean([t['quality_score'] for t in older if t['quality_score'] is not None]) if older and any(t['quality_score'] is not None for t in older) else None,
                'avg_elements': statistics.mean([t['total_elements'] for t in older if t['total_elements'] is not None]) if older and any(t['total_elements'] is not None for t in older) else None
            } if older else None
        }
    
    return {'total_data_points': 0}

def identify_problems():
    """Identifiziert problematische Muster."""
    
    problems = []
    
    kpi_files = list(Path('outputs').rglob('*kpis.json'))
    
    # Problem 1: Niedrige Quality Scores
    low_scores = []
    zero_scores = 0
    for kpi_file in kpi_files:
        try:
            with open(kpi_file, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                score = kpi_data.get('quality_score', 100)
                if score == 0:
                    zero_scores += 1
                elif score < 30:
                    low_scores.append((str(kpi_file), score))
        except:
            continue
    
    if zero_scores > 0:
        problems.append({
            'type': 'zero_quality_scores',
            'severity': 'high',
            'count': zero_scores,
            'description': f'{zero_scores} Analysen haben quality_score = 0.0 (wahrscheinlich ohne Ground Truth)'
        })
    
    if low_scores:
        problems.append({
            'type': 'low_quality_scores',
            'severity': 'medium',
            'count': len(low_scores),
            'examples': low_scores[:5]
        })
    
    # Problem 2: Halluzinationen
    hallucination_files = []
    for kpi_file in kpi_files:
        try:
            with open(kpi_file, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                hall = kpi_data.get('hallucinated_elements', 0)
                if hall > 20:
                    hallucination_files.append((str(kpi_file), hall))
        except:
            continue
    
    if hallucination_files:
        problems.append({
            'type': 'high_hallucinations',
            'severity': 'high',
            'count': len(hallucination_files),
            'description': 'Viele erkannte Elemente die nicht in Ground Truth vorhanden sind',
            'examples': hallucination_files[:5]
        })
    
    # Problem 3: Niedrige Confidence
    low_confidence = []
    for kpi_file in kpi_files:
        try:
            with open(kpi_file, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                conf = kpi_data.get('avg_element_confidence', 1.0)
                if conf < 0.75:
                    low_confidence.append((str(kpi_file), conf))
        except:
            continue
    
    if low_confidence:
        problems.append({
            'type': 'low_confidence',
            'severity': 'medium',
            'count': len(low_confidence),
            'examples': low_confidence[:5]
        })
    
    return problems

def generate_recommendations():
    """Generiert Empfehlungen basierend auf Analyse."""
    
    recommendations = []
    
    # Empfehlung 1: Ground Truth Integration
    recommendations.append({
        'priority': 'high',
        'category': 'data_quality',
        'title': 'Ground Truth Integration verbessern',
        'description': 'Viele Analysen haben quality_score = 0.0, was darauf hindeutet, dass Ground Truth nicht korrekt geladen oder verglichen wird.',
        'action_items': [
            'Ground Truth Dateien validieren',
            'Vergleichslogik in KPI Calculator prüfen',
            'Fallback für fehlende Ground Truth implementieren'
        ]
    })
    
    # Empfehlung 2: Halluzination-Reduktion
    recommendations.append({
        'priority': 'high',
        'category': 'accuracy',
        'title': 'Halluzinationen reduzieren',
        'description': 'System erkennt viele Elemente die nicht in Ground Truth vorhanden sind.',
        'action_items': [
            'Prompt Engineering zur besseren Präzision',
            'Confidence Threshold erhöhen',
            'Post-Processing Filter implementieren'
        ]
    })
    
    # Empfehlung 3: Confidence Calibration
    recommendations.append({
        'priority': 'medium',
        'category': 'reliability',
        'title': 'Confidence Calibration',
        'description': 'Confidence-Werte sollten besser kalibriert werden um echte Unsicherheit widerzuspiegeln.',
        'action_items': [
            'Confidence Scores gegen Ground Truth validieren',
            'Calibration Curve erstellen',
            'Threshold-basierte Filterung implementieren'
        ]
    })
    
    return recommendations

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generiert einen Markdown Report."""
    
    md = "# Umfassende Test-Ergebnis Analyse\n\n"
    md += f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # KPI Analysis
    md += "## 1. KPI Analyse\n\n"
    kpi_analysis = results.get('kpi_analysis', {})
    md += f"- **Gesamt KPI-Dateien**: {kpi_analysis.get('total_files', 0)}\n\n"
    
    metrics = kpi_analysis.get('metrics_summary', {})
    if metrics:
        md += "### Metrik-Statistiken\n\n"
        md += "| Metrik | Count | Mean | Median | Min | Max | StdDev |\n"
        md += "|--------|-------|------|--------|-----|-----|--------|\n"
        for key, stats in metrics.items():
            md += f"| {key} | {stats['count']} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['std_dev']:.2f} |\n"
        md += "\n"
    
    # Visualization Analysis
    md += "## 2. Visualisierungs-Analyse\n\n"
    viz_analysis = results.get('visualization_analysis', {})
    for viz_type, data in viz_analysis.items():
        md += f"- **{viz_type}**: {data['count']} Dateien\n"
    md += "\n"
    
    # Trend Analysis
    md += "## 3. Trend-Analyse\n\n"
    trend_analysis = results.get('trend_analysis', {})
    md += f"- **Gesamt Datenpunkte**: {trend_analysis.get('total_data_points', 0)}\n"
    recent = trend_analysis.get('recent_performance', {})
    if recent.get('avg_quality') is not None:
        md += f"- **Durchschnittlicher Quality Score (recent)**: {recent['avg_quality']:.2f}\n"
    md += "\n"
    
    # Problems
    md += "## 4. Identifizierte Probleme\n\n"
    problems = results.get('problem_patterns', [])
    if problems:
        for i, problem in enumerate(problems, 1):
            md += f"### Problem {i}: {problem.get('type', 'unknown')}\n\n"
            md += f"- **Severity**: {problem.get('severity', 'unknown')}\n"
            md += f"- **Count**: {problem.get('count', 0)}\n"
            if 'description' in problem:
                md += f"- **Description**: {problem['description']}\n"
            md += "\n"
    else:
        md += "Keine kritischen Probleme identifiziert.\n\n"
    
    # Recommendations
    md += "## 5. Empfehlungen\n\n"
    recommendations = results.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        md += f"### {i}. {rec.get('title', 'Unbenannt')}\n\n"
        md += f"- **Priority**: {rec.get('priority', 'unknown')}\n"
        md += f"- **Category**: {rec.get('category', 'unknown')}\n"
        md += f"- **Description**: {rec.get('description', '')}\n\n"
        md += "**Action Items**:\n"
        for item in rec.get('action_items', []):
            md += f"- {item}\n"
        md += "\n"
    
    return md

if __name__ == '__main__':
    results = analyze_all_results()
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    print(f"\nKPI-Dateien analysiert: {results['kpi_analysis']['total_files']}")
    print(f"Probleme identifiziert: {len(results['problem_patterns'])}")
    print(f"Empfehlungen: {len(results['recommendations'])}")

