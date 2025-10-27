param(
    [string]$RootDir = "DroneRF"
)

$ErrorActionPreference = "Stop"

function Invoke-Extract {
    param(
        [Parameter(Mandatory=$true)][string]$Archive,
        [Parameter(Mandatory=$true)][string]$OutDir
    )

    $unrar = Get-Command unrar -ErrorAction SilentlyContinue
    $sz = Get-Command 7z -ErrorAction SilentlyContinue

    if ($unrar) {
        # unrar x -o+ -idq -inul -- "$archive" "$outdir/"
        & $unrar.Source $Archive "x" "-o+" "-idq" "-inul" "--" $Archive ("{0}{1}" -f ($OutDir), [IO.Path]::DirectorySeparatorChar)
    }
    elseif ($sz) {
        # 7z x -y -o"$outdir" -- "$archive" >/dev/null
        & $sz.Source "x" "-y" ("-o{0}" -f $OutDir) "--" $Archive | Out-Null
    }
    else {
        throw "Error: need 'unrar' or '7z' installed and in PATH."
    }
}

# Walk all .rar files under RootDir
Get-ChildItem -LiteralPath $RootDir -Recurse -File -Filter *.rar | ForEach-Object {
    $f = $_.FullName
    $dn = $_.DirectoryName
    $bn = $_.Name

    # Detect multi-part archives: something.partN.rar
    $isMultipart = $false
    $partnum = $null
    if ($bn -match '\.part(\d+)\.rar$') {
        $isMultipart = $true
        $partnum = [int]$Matches[1]
        if ($partnum -ne 1) {
            # Only extract from .part1.rar; skip the rest
            return
        }
        $base = $bn -replace '\.part\d+\.rar$', ''
    }
    else {
        $base = $bn -replace '\.rar$', ''
    }

    $out = Join-Path $dn $base
    if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

    Write-Host ("Extracting: {0} -> {1}" -f $f, $out)
    Invoke-Extract -Archive $f -OutDir $out
}

Write-Host "Done."
