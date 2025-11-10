# Live Test Monitoring Script
# Überwacht den laufenden Test und zeigt Performance-Metriken

$logFile = (Get-ChildItem outputs\live_test -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\logs\test.log"

if (-not (Test-Path $logFile)) {
    Write-Host "[ERROR] Log file not found. Start the test first!" -ForegroundColor Red
    exit 1
}

Write-Host "=== LIVE TEST MONITORING ===" -ForegroundColor Cyan
Write-Host "Log File: $logFile" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

$lastPos = 0
$metrics = @{
    'api_calls' = 0
    'rate_limits' = 0
    'dsq_throttles' = 0
    'errors' = 0
    'start_time' = Get-Date
}

while ($true) {
    if (Test-Path $logFile) {
        $file = Get-Item $logFile
        if ($file.Length -gt $lastPos) {
            $content = Get-Content $logFile -Raw
            $newContent = $content.Substring($lastPos)
            
            # Update metrics
            $metrics['api_calls'] += ([regex]::Matches($newContent, 'REQUEST|call_llm', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
            $metrics['rate_limits'] += ([regex]::Matches($newContent, '429|rate.*limit|resource.*exhausted|DSQ.*Rate.*Limit', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
            $metrics['dsq_throttles'] += ([regex]::Matches($newContent, 'DSQ.*Request.*Smoothing|Throttling', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
            $metrics['errors'] += ([regex]::Matches($newContent, 'ERROR|FAIL', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
            
            # Show new content
            Write-Host $newContent -NoNewline
            
            $lastPos = $file.Length
            
            # Print metrics every 30 seconds
            $elapsed = (Get-Date) - $metrics['start_time']
            if ($elapsed.TotalSeconds % 30 -lt 3) {
                Write-Host "`n=== METRICS ===" -ForegroundColor Cyan
                Write-Host "API Calls: $($metrics['api_calls'])" -ForegroundColor Green
                Write-Host "Rate Limits: $($metrics['rate_limits'])" -ForegroundColor $(if ($metrics['rate_limits'] -gt 0) { 'Red' } else { 'Green' })
                Write-Host "DSQ Throttles: $($metrics['dsq_throttles'])" -ForegroundColor Yellow
                Write-Host "Errors: $($metrics['errors'])" -ForegroundColor $(if ($metrics['errors'] -gt 5) { 'Red' } else { 'Yellow' })
                if ($metrics['api_calls'] -gt 0) {
                    $rate = [math]::Round(($metrics['api_calls'] / $elapsed.TotalMinutes), 2)
                    Write-Host "API Call Rate: $rate calls/min" -ForegroundColor Cyan
                }
                Write-Host "===============`n" -ForegroundColor Cyan
            }
        }
    }
    
    Start-Sleep -Seconds 2
}

