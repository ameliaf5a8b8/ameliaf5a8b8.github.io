content/ → what you write

static/ → images, PDFs

themes/ → look & feel

hugo.toml


function tex2md {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Filename
    )

    # Check if input file exists
    if (-not (Test-Path $Filename)) {
        Write-Host "File not found: $Filename" -ForegroundColor Red
        return
    }

    # Output name
    $baseName   = [System.IO.Path]::GetFileNameWithoutExtension($Filename)
    $outputName = "$baseName.md"

    # Dates
    $today = Get-Date -Format "yyyy-MM-dd"

    # Execute Pandoc
    pandoc $Filename `
        --from=latex `
        --to=markdown `
        --lua-filter="C:\Users\User\Coding_projects\blog\filters\latex_cleanup.lua" `
        --wrap=none `
        -o $outputName

    # Abort if pandoc failed
    if (-not (Test-Path $outputName)) {
        Write-Host "Pandoc failed - no output created." -ForegroundColor Red
        return
    }

    # YAML front matter (no here-strings)
    $frontMatter = @(
        '---'
        "title: `"$baseName`""
        "date: $today"
        "lastmod: $today"
        'tags: []'
        'categories: []'
        'math: true'
        '---'
        ''
    ) -join "`n"

    # Prepend front matter
    $content = Get-Content $outputName -Raw
    Set-Content $outputName ($frontMatter + $content)

    Write-Host "Converted $Filename to $outputName" -ForegroundColor Green
}
