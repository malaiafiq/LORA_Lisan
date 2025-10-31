# PowerShell script to activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the enhanced training script:" -ForegroundColor Yellow
Write-Host "  python train_whisper_lora_enhanced_simple.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test the installation:" -ForegroundColor Yellow
Write-Host "  python test_simple.py" -ForegroundColor Cyan
Write-Host ""
