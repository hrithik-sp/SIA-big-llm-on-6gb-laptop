# Count open browser tabs (Chrome, Edge, Brave, Firefox) through Windows UI Automation.
# Read-only: it only inspects the tab strip of visible browser windows. Prints JSON.
param([switch]$Titles)
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$root = [System.Windows.Automation.AutomationElement]::RootElement
$tabCond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::TabItem)
$browsers = @{ chrome = "Chrome"; msedge = "Edge"; brave = "Brave"; firefox = "Firefox" }
$result = @{}
$windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition)
foreach ($w in $windows) {
    try { $proc = Get-Process -Id $w.Current.ProcessId -ErrorAction Stop } catch { continue }
    $name = $browsers[$proc.ProcessName.ToLower()]
    if (-not $name) { continue }
    # Edge app windows (like Sprout itself) have no tab strip, so they add nothing.
    $tabs = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, $tabCond)
    if ($tabs.Count -eq 0) { continue }
    if (-not $result.ContainsKey($name)) { $result[$name] = @{ windows = 0; tabs = 0; titles = @() } }
    $result[$name].windows += 1
    $result[$name].tabs += $tabs.Count
    if ($Titles) { foreach ($t in $tabs) { $result[$name].titles += $t.Current.Name } }
}
$result | ConvertTo-Json -Depth 4 -Compress
