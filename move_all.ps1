param(
    [string]$Root = "DroneRF"
)

$ErrorActionPreference = "Stop"

$HighDir = Join-Path $Root "High"
$LowDir  = Join-Path $Root "Low"
New-Item -ItemType Directory -Path $HighDir -Force | Out-Null
New-Item -ItemType Directory -Path $LowDir  -Force | Out-Null

function Move-WithUniqueName {
    param(
        [Parameter(Mandatory=$true)][string]$Src,
        [Parameter(Mandatory=$true)][string]$TgtDir
    )

    $base = [IO.Path]::GetFileName($Src)
    $name = [IO.Path]::GetFileNameWithoutExtension($base)
    $ext  = [IO.Path]::GetExtension($base)  # includes leading dot or empty

    if ([string]::IsNullOrEmpty($ext)) {
        $out = Join-Path $TgtDir $name
    } else {
        $out = Join-Path $TgtDir ($name + $ext)
    }

    $i = 1
    while (Test-Path -LiteralPath $out) {
        if ([string]::IsNullOrEmpty($ext)) {
            $out = Join-Path $TgtDir ("{0}_{1}" -f $name, $i)
        } else {
            $out = Join-Path $TgtDir ("{0}_{1}{2}" -f $name, $i, $ext)
        }
        $i++
    }

    Move-Item -LiteralPath $Src -Destination $out
}

# Find all directories under $Root whose names end with _h or _l (case-insensitive),
# excluding the destination folders themselves.
$dirs = Get-ChildItem -LiteralPath $Root -Recurse -Directory |
    Where-Object {
        # Exclude the destination paths (and their children)
        ($_.FullName -notlike ($HighDir + "*")) -and
        ($_.FullName -notlike ($LowDir + "*")) -and
        # Name ends with _h or _l (case-insensitive)
        ($_.Name -match '_(?i:[hl])$')
    }

foreach ($d in $dirs) {
    $folderName = $d.Name
    if ($folderName -match '(?i)_h$') {
        $dest = $HighDir
    } else {
        # If it matched _l$, route to Low
        $dest = $LowDir
    }

    # Move all CSVs found under this directory (recursively)
    Get-ChildItem -LiteralPath $d.FullName -Recurse -File -Filter *.csv | ForEach-Object {
        Move-WithUniqueName -Src $_.FullName -TgtDir $dest
    }
}

Write-Host "✅ Done. All CSVs moved into:"
Write-Host ("   High -> {0}" -f $HighDir)
Write-Host ("   Low  -> {0}" -f $LowDir)
