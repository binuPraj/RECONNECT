class AppConfig {
  // Toggle this when testing on a physical Android device vs emulator.
  // - Emulator: useSimulator = true  → connects to 127.0.0.1 (host loopback via ADB tunnel)
  // - Physical device: useSimulator = false → connects to Windows Wi-Fi LAN IP
  //
  // To find your current IP: run `ipconfig` and look for "Wireless LAN adapter Wi-Fi".
  // Update _deviceHost below whenever your router assigns a different IP.
  static const bool useSimulator = false;

  static const String _deviceHost = '192.168.1.95'; // ← Windows Wi-Fi LAN IP

  static String get serverHost => useSimulator ? '127.0.0.1' : _deviceHost;

  static String get wsAudioUrl => 'ws://$serverHost:8000/ws/audio';
  static String get httpBaseUrl => 'http://$serverHost:8000';
}
