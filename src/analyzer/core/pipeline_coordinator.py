    def _run_hybrid_validation(
        self,
        image_path: str
    ) -> None:
        """
        Hybrid validation combining CV line detection with semantic validation.
        
        CRITICAL FIX: This solves the "Blinde CV-Kopplung" problem by:
        1. Extracting physical polylines from CV (line_extractor)
        2. Validating connections semantically (sensors not as sources, etc.)
        3. Using CV polylines to detect and correct direction errors in LLM connections
        
        Example:
        - LLM says: FT-10 -> Fv-3-3040 (wrong direction)
        - CV finds physical line from Fv-3-3040 -> FT-10
        - Semantic validation: FT-10 is a sensor, can't be source
        - Result: Correct to Fv-3-3040 -> FT-10
        
        Args:
            image_path: Path to the image
        """
        logger.info("=== Starting Hybrid CV + Semantic Validation ===")
        
        elements = self._analysis_results.get('elements', [])
        connections = self._analysis_results.get('connections', [])
        
        if not connections:
            logger.info("No connections to validate. Skipping hybrid validation.")
            return
        
        # Step 1: Extract physical polylines using CV
        from src.analyzer.analysis.line_extractor import LineExtractor
        
        line_extractor = LineExtractor(self.active_logic_parameters)
        excluded_zones = self._excluded_zones
        legend_data = self._analysis_results.get('legend_data', {})
        
        cv_result = line_extractor.extract_pipeline_lines(
            image_path=image_path,
            elements=elements,
            excluded_zones=excluded_zones,
            legend_data=legend_data if legend_data.get('has_legend') else None
        )
        
        # Get physical polylines and line segments
        pipeline_lines = cv_result.get('pipeline_lines', [])
        line_segments = cv_result.get('line_segments', [])
        
        logger.info(f"CV extracted {len(pipeline_lines)} pipeline lines, {len(line_segments)} line segments")
        
        # Step 2: Build mapping from element pairs to physical lines
        # Key: (from_id, to_id) -> list of polylines
        physical_lines_map = {}
        elements_map = {el.get('id'): el for el in elements if el.get('id')}
        
        for line in pipeline_lines:
            from_id = line.get('from_id')
            to_id = line.get('to_id')
            polyline = line.get('polyline', [])
            
            if from_id and to_id and polyline:
                key = (from_id, to_id)
                if key not in physical_lines_map:
                    physical_lines_map[key] = []
                physical_lines_map[key].append(polyline)
        
        # Step 3: Validate connections semantically (removes invalid, reverses wrong direction)
        validated_connections = self._validate_connection_semantics(connections, elements)
        
        # Step 4: Cross-validate with CV polylines
        corrected_connections = []
        corrected_count = 0
        removed_count = 0
        
        for conn in validated_connections:
            from_id = conn.get('from_id')
            to_id = conn.get('to_id')
            
            if not from_id or not to_id:
                continue
            
            # Check if CV found a physical line for this connection
            forward_key = (from_id, to_id)
            reverse_key = (to_id, from_id)
            
            has_forward_line = forward_key in physical_lines_map
            has_reverse_line = reverse_key in physical_lines_map
            
            # Case 1: CV found line in correct direction -> keep connection
            if has_forward_line:
                conn['polyline'] = physical_lines_map[forward_key][0]  # Use first polyline
                conn['cv_verified'] = True
                corrected_connections.append(conn)
                continue
            
            # Case 2: CV found line in reverse direction -> reverse connection if semantically valid
            if has_reverse_line:
                from_el = elements_map.get(from_id)
                to_el = elements_map.get(to_id)
                
                if from_el and to_el:
                    from_type = from_el.get('type', '')
                    to_type = to_el.get('type', '')
                    
                    # Check if reverse is semantically valid
                    # (e.g., Sensor can't be source, but Pump -> Sensor is valid)
                    sensor_types = {'Flow Transmitter', 'Volume Flow Sensor', 'FT', 'PT', 'TT'}
                    source_types = {'Source', 'Pump', 'CHP', 'HP'}
                    
                    # If current direction is semantically invalid (Sensor -> Source),
                    # and reverse would be valid (Source -> Sensor), reverse it
                    if from_type in sensor_types and to_type in source_types:
                        logger.info(f"CV correction: Reversing {from_id} -> {to_id} to {to_id} -> {from_id} (CV found physical line in reverse direction)")
                        conn_copy = conn.copy()
                        conn_copy['from_id'] = to_id
                        conn_copy['to_id'] = from_id
                        conn_copy['polyline'] = physical_lines_map[reverse_key][0]
                        conn_copy['cv_verified'] = True
                        conn_copy['cv_corrected'] = True
                        corrected_connections.append(conn_copy)
                        corrected_count += 1
                        continue
            
            # Case 3: No CV line found -> keep connection if semantically valid, but mark as unverified
            # (LLM might have detected a connection that CV missed due to threshold issues)
            conn['cv_verified'] = False
            corrected_connections.append(conn)
        
        # Update analysis results
        original_count = len(connections)
        self._analysis_results['connections'] = corrected_connections
        
        logger.info(f"Hybrid validation complete: {original_count} -> {len(corrected_connections)} connections "
                   f"({corrected_count} CV-corrected, {removed_count} removed)")
        
        # Store CV results for later use
        self._analysis_results['cv_pipeline_lines'] = pipeline_lines
        self._analysis_results['cv_line_segments'] = line_segments
