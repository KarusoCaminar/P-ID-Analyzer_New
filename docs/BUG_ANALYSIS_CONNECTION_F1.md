# 🐛 Bug Analysis: Connection F1 = 0.0

**Datum:** 2025-11-07  
**Status:** ✅ Bug gefunden!

---

## 🔍 Problem

**Connection F1 = 0.0 für ALLE Parameter-Kombinationen**

Alle 36 Tests haben Connection F1 = 0.0, obwohl:
- ✅ Element F1 = 1.0 (perfekt!)
- ✅ 10 Connections im Analysis-Result
- ✅ 8 Connections im Ground Truth

---

## 🐛 BUG #1: Ground Truth Format Mismatch

### Problem

**Ground Truth Format:**
```json
{
  "connections": [
    {
      "name": "Conn_P201_to_Fv3040",
      "from_converter_ports": [{"unit_name": "P-201", "port": "Out"}],
      "to_converter_ports": [{"unit_name": "Fv-3-3040", "port": "In"}]
    }
  ]
}
```

**KPI Calculator erwartet:**
```python
truth_connections = {(c.get('from_id'), c.get('to_id')) for c in truth_data.get('connections', [])
                    if c.get('from_id') and c.get('to_id')}
```

**Ergebnis:** `truth_connections` ist **leer**, weil `from_id` und `to_id` nicht existieren!

---

### Location

**File:** `src/analyzer/evaluation/kpi_calculator.py`  
**Line:** 550-551

```python
truth_connections = {(c.get('from_id'), c.get('to_id')) for c in truth_data.get('connections', [])
                    if c.get('from_id') and c.get('to_id')}
```

---

### Fix

**Wir müssen das Ground Truth Format konvertieren:**

```python
def extract_connections_from_truth(truth_connections: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """
    Extract connections from ground truth format.
    
    Handles two formats:
    1. Simple format: {"from_id": "P-201", "to_id": "Fv-3-3040"}
    2. Converter ports format: {
        "from_converter_ports": [{"unit_name": "P-201", "port": "Out"}],
        "to_converter_ports": [{"unit_name": "Fv-3-3040", "port": "In"}]
    }
    """
    connections = set()
    
    for conn in truth_connections:
        # Format 1: Simple format (from_id, to_id)
        if 'from_id' in conn and 'to_id' in conn:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            if from_id and to_id:
                connections.add((from_id, to_id))
        
        # Format 2: Converter ports format
        elif 'from_converter_ports' in conn and 'to_converter_ports' in conn:
            from_ports = conn.get('from_converter_ports', [])
            to_ports = conn.get('to_converter_ports', [])
            
            # Extract unit names from ports
            from_ids = [p.get('unit_name') for p in from_ports if p.get('unit_name')]
            to_ids = [p.get('unit_name') for p in to_ports if p.get('unit_name')]
            
            # Handle multiple from/to (merges/splits)
            for from_id in from_ids:
                for to_id in to_ids:
                    if from_id and to_id:
                        connections.add((from_id, to_id))
    
    return connections
```

---

## 🎯 Lösung

### Step 1: Fix KPI Calculator

Update `_calculate_quality_metrics` in `kpi_calculator.py`:

```python
# Map truth connections to analysis connection space
analysis_connections = {(c.get('from_id'), c.get('to_id')) for c in analysis_data.get('connections', [])
                       if c.get('from_id') and c.get('to_id')}

# CRITICAL FIX: Extract connections from ground truth (handle both formats)
truth_connections = extract_connections_from_truth(truth_data.get('connections', []))
```

---

## 📊 Erwartetes Ergebnis

Nach dem Fix:
- ✅ Ground Truth Connections werden korrekt extrahiert (8 Connections)
- ✅ Analysis Connections werden korrekt gematched (10 Connections)
- ✅ Connection F1 sollte > 0.0 sein (abhängig von Match-Qualität)

---

## 🔍 Weitere mögliche Bugs

### Bug #2: Merge/Split Connections

Ground Truth hat Merge/Split Nodes:
```json
{
  "name": "Node_Merge_before_Mixer",
  "from_converter_ports": [
    {"unit_name": "FT-10", "port": "Out"},
    {"unit_name": "FT-11", "port": "Out"}
  ],
  "to_converter_ports": [{"unit_name": "M-08", "port": "In"}]
}
```

Das bedeutet: **2 Connections** (FT-10 → M-08, FT-11 → M-08)

**Lösung:** `extract_connections_from_truth` muss bereits Merge/Split Nodes handhaben (siehe oben).

---

### Bug #3: Control Connections

Ground Truth hat Control Connections:
```json
{
  "name": "Supply_ISA_to_Fv3040",
  "from_converter_ports": [{"unit_name": "ISA-Supply", "port": "Out"}],
  "to_converter_ports": [{"unit_name": "Fv-3-3040", "port": "Control"}]
}
```

**Frage:** Sollen Control Connections auch gematched werden?

**Aktuell:** Analysis hat `kind: 'process'` für alle Connections. Control Connections haben `kind: 'control'`.

**Lösung:** Beide Types matchen, oder nur Process Connections matchen?

---

## ✅ Nächste Schritte

1. ✅ **Bug #1 fixen** (Ground Truth Format)
2. ⏳ **Bug #2 testen** (Merge/Split)
3. ⏳ **Bug #3 entscheiden** (Control Connections)
4. ⏳ **Parameter-Tuning erneut laufen** (für Validierung)

