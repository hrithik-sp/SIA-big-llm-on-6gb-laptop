# Read-only device queries for Sprout's device_info tool. Prints one JSON object.
#   powershell -NoProfile -File devices.ps1 -Category battery|display|audio|bluetooth|usb|printers|startup|internet|model|location
# Nothing here changes a setting; the battery report is written to a temp file and deleted.
param([Parameter(Mandatory)][string]$Category)
$ErrorActionPreference = "SilentlyContinue"
$out = @{}

switch ($Category) {
  "battery" {
    $x = Join-Path $env:TEMP ("sprout-battery-" + [guid]::NewGuid().ToString("N") + ".xml")
    powercfg /batteryreport /xml /output $x | Out-Null
    if (Test-Path $x) {
      [xml]$r = Get-Content $x -Raw
      Remove-Item $x -Force
      $b = @($r.BatteryReport.Batteries.Battery)[0]
      if ($b) {
        $out.design_capacity_mwh = [int64]$b.DesignCapacity
        $out.full_charge_capacity_mwh = [int64]$b.FullChargeCapacity
        if ([int64]$b.DesignCapacity -gt 0) { $out.health_pct = [math]::Round(100 * [int64]$b.FullChargeCapacity / [int64]$b.DesignCapacity) }
        if ([int]$b.CycleCount -gt 0) { $out.cycle_count = [int]$b.CycleCount }
        $out.manufacturer = "$($b.Manufacturer)"; $out.chemistry = "$($b.Chemistry)"
      }
    }
  }
  "display" {
    $br = Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness | Where-Object Active | Select-Object -First 1
    if ($br) { $out.brightness_pct = [int]$br.CurrentBrightness }
    $out.monitors = @(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorID | ForEach-Object {
      ($_.UserFriendlyName | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ }) -join "" } | Where-Object { $_ })
  }
  "audio" {
    Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {  // full vtable in Windows' order (endpointvolume.h); only the two getters are called
  int RegisterControlChangeNotify(IntPtr p); int UnregisterControlChangeNotify(IntPtr p); int GetChannelCount(out uint n);
  int SetMasterVolumeLevel(float l, IntPtr g); int SetMasterVolumeLevelScalar(float l, IntPtr g);
  int GetMasterVolumeLevel(out float l); int GetMasterVolumeLevelScalar(out float l);
  int SetChannelVolumeLevel(uint c, float l, IntPtr g); int SetChannelVolumeLevelScalar(uint c, float l, IntPtr g);
  int GetChannelVolumeLevel(uint c, out float l); int GetChannelVolumeLevelScalar(uint c, out float l);
  int SetMute([MarshalAs(UnmanagedType.Bool)] bool m, IntPtr g); int GetMute([MarshalAs(UnmanagedType.Bool)] out bool m); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int Activate(ref Guid id, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int f(); int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}
public static class SproutVolume {
  public static float[] Get() {
    var en = (IMMDeviceEnumerator)(new MMDeviceEnumerator()); IMMDevice dev; en.GetDefaultAudioEndpoint(0, 1, out dev);
    var iid = typeof(IAudioEndpointVolume).GUID; object o; dev.Activate(ref iid, 23, IntPtr.Zero, out o);
    var v = (IAudioEndpointVolume)o; float lvl; bool mute; v.GetMasterVolumeLevelScalar(out lvl); v.GetMute(out mute);
    return new float[] { lvl, mute ? 1f : 0f }; } }
"@
    $v = [SproutVolume]::Get()
    if ($v) { $out.volume_pct = [math]::Round(100 * $v[0]); $out.muted = [bool]$v[1] }
    $out.devices = @(Get-PnpDevice -Class AudioEndpoint -PresentOnly -Status OK | Select-Object -ExpandProperty FriendlyName)
  }
  "bluetooth" {
    $all = @(Get-PnpDevice -Class Bluetooth)
    $out.adapter = @($all | Where-Object { $_.InstanceId -like "USB\*" -or $_.InstanceId -like "PCI\*" } | Select-Object -ExpandProperty FriendlyName) -join "; "
    $out.devices = @($all | Where-Object { $_.InstanceId -like "BTHENUM\*" -or $_.InstanceId -like "BTHLE\*" -or $_.InstanceId -like "BTHLEDEVICE\*" } |
      Where-Object { $_.FriendlyName -and $_.FriendlyName -notmatch "Service|Enumerator|Transport|Attribute|Generic Access|Generic Attribute" } |
      ForEach-Object {
        $con = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName "{83DA6326-97A6-4088-9453-A1923F573B29} 15").Data
        @{ name = $_.FriendlyName; connected = [bool]$con }
      } | Sort-Object { $_.name } -Unique)
  }
  "usb" {
    $out.devices = @(Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "USB\VID*" -and $_.Class -ne "USB" -and $_.FriendlyName } |
      Select-Object -ExpandProperty FriendlyName -Unique)
  }
  "printers" {
    $out.printers = @(Get-CimInstance Win32_Printer | ForEach-Object { @{ name = $_.Name; default = [bool]$_.Default } })
  }
  "startup" {
    $out.startup_apps = @(Get-CimInstance Win32_StartupCommand | Select-Object -ExpandProperty Name -Unique)
  }
  "internet" {
    $nlm = [Activator]::CreateInstance([Type]::GetTypeFromCLSID([Guid]"DCB00C01-570F-4A9B-8D69-199FDBA5723B"))
    $out.internet_connected = [bool]$nlm.IsConnectedToInternet
    $out.network_connected = [bool]$nlm.IsConnected
    $cfg = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -First 1
    if ($cfg) { $out.adapter = $cfg.InterfaceAlias; $out.local_ip = "$($cfg.IPv4Address.IPAddress)"
                $out.gateway = "$($cfg.IPv4DefaultGateway.NextHop)"; $out.dns = @($cfg.DNSServer.ServerAddresses) }
  }
  "model" {
    $cs = Get-CimInstance Win32_ComputerSystem; $bios = Get-CimInstance Win32_BIOS
    $out.manufacturer = $cs.Manufacturer; $out.model = $cs.Model
    $out.bios = "$($bios.Manufacturer) $($bios.SMBIOSBIOSVersion)"
  }
  "location" {
    Add-Type -AssemblyName System.Device
    $w = New-Object System.Device.Location.GeoCoordinateWatcher([System.Device.Location.GeoPositionAccuracy]::Default)
    $w.Start()
    for ($i = 0; $i -lt 40; $i++) {
      if ($w.Status -eq "Ready" -and -not $w.Position.Location.IsUnknown) { break }
      Start-Sleep -Milliseconds 400
    }
    $out.permission = "$($w.Permission)"; $out.status = "$($w.Status)"
    $c = $w.Position.Location
    if (-not $c.IsUnknown) { $out.lat = $c.Latitude; $out.lon = $c.Longitude; $out.accuracy_m = [math]::Round($c.HorizontalAccuracy) }
    $w.Stop()
  }
  default { $out.error = "unknown category" }
}
$out | ConvertTo-Json -Depth 4 -Compress
